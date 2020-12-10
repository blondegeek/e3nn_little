# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except, abstract-method, arguments-differ
import argparse
import datetime
import itertools
import pickle
import subprocess
import time

import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import SchNet

from e3nn_little.nn.models import Network


def execute(args):
    path = 'QM9'
    dataset = QM9(path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Report meV instead of eV.
    units = 1000 if args.target in [2, 3, 4, 6, 7, 8, 9, 10] else 1

    _, datasets = SchNet.from_qm9_pretrained(path, dataset, args.target)
    train_dataset, val_dataset, _test_dataset = datasets

    model = Network(
        muls=(args.mul0, args.mul1, args.mul2),
        ps=(1,) if 'shp' in args.opts else (1, -1),
        lmax=args.lmax,
        num_layers=args.num_layers,
        rad_gaussians=args.rad_gaussians,
        rad_hs=(args.rad_h,) * args.rad_layers + (args.rad_bottleneck,),
        mean=args.mean, std=args.std,
        atomref=dataset.atomref(args.target),
        options=args.opts
    )
    model = model.to(device)

    # profile
    loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False)
    for step, data in enumerate(loader):
        with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
            data = data.to(device)
            pred = model(data.z, data.pos, data.batch)
            mse = (pred.view(-1) - data.y[:, args.target]).pow(2)
            mse.mean().backward()
        if step == 5:
            break
    prof.export_chrome_trace(f"{datetime.datetime.now()}.json")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=25, factor=0.5, verbose=True)

    dynamics = []
    wall = time.perf_counter()
    wall_print = time.perf_counter()

    for epoch in itertools.count():

        errs = []
        loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        for step, data in enumerate(loader):
            data = data.to(device)

            pred = model(data.z, data.pos, data.batch)
            optim.zero_grad()
            (pred.view(-1) - data.y[:, args.target]).pow(2).mean().backward()
            optim.step()

            err = pred.view(-1) - data.y[:, args.target]
            errs += [err.cpu().detach()]

            if time.perf_counter() - wall_print > 15:
                wall_print = time.perf_counter()
                w = time.perf_counter() - wall
                e = epoch + (step + 1) / len(loader)
                print((
                    f'[{e:.1f}] ['
                    f'wall={w / 3600:.2f}h '
                    f'wall/epoch={w / e:.0f}s '
                    f'wall/step={1e3 * w / e / len(loader):.0f}ms '
                    f'step={step}/{len(loader)} '
                    f'mae={units * torch.cat(errs)[-200:].abs().mean():.5f} '
                    f'lr={optim.param_groups[0]["lr"]:.1e}]'
                ), flush=True)

        train_err = torch.cat(errs)

        errs = []
        loader = DataLoader(val_dataset, batch_size=256)
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                pred = model(data.z, data.pos, data.batch)

            err = pred.view(-1) - data.y[:, args.target]
            errs += [err.cpu().detach()]
        val_err = torch.cat(errs)

        dynamics += [{
            'epoch': epoch,
            'wall': time.perf_counter() - wall,
            'train': {
                'mae': {
                    'mean': units * train_err.abs().mean().item(),
                    'std': units * train_err.abs().std().item(),
                },
                'mse': {
                    'mean': units * train_err.pow(2).mean().item(),
                    'std': units * train_err.pow(2).std().item(),
                }
            },
            'val': {
                'mae': {
                    'mean': units * val_err.abs().mean().item(),
                    'std': units * val_err.abs().std().item(),
                },
                'mse': {
                    'mean': units * val_err.pow(2).mean().item(),
                    'std': units * val_err.pow(2).std().item(),
                }
            },
            'lr': optim.param_groups[0]["lr"],
        }]

        print(f'[{epoch}] Target: {args.target:02d}, MAE TRAIN: {units * train_err.abs().mean():.5f} ± {units * train_err.abs().std():.5f}, MAE VAL: {units * val_err.abs().mean():.5f} ± {units * val_err.abs().std():.5f}', flush=True)

        scheduler.step(val_err.pow(2).mean())

        yield {
            'args': args,
            'dynamics': dynamics,
            'state': {k: v.cpu() for k, v in model.state_dict().items()},
        }


def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--mul0", type=int, default=64)
    parser.add_argument("--mul1", type=int, default=8)
    parser.add_argument("--mul2", type=int, default=0)
    parser.add_argument("--lmax", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--rad_gaussians", type=int, default=50)
    parser.add_argument("--rad_h", type=int, default=128)
    parser.add_argument("--rad_bottleneck", type=int, default=128)
    parser.add_argument("--rad_layers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--opts", type=str, default="res")
    parser.add_argument("--target", type=int, default=7)
    parser.add_argument("--mean", type=float, default=0)
    parser.add_argument("--std", type=float, default=1)

    args = parser.parse_args()

    with open(args.output, 'wb') as handle:
        pickle.dump(args, handle)

    for data in execute(args):
        data['git'] = git
        with open(args.output, 'wb') as handle:
            pickle.dump(args, handle)
            pickle.dump(data, handle)


if __name__ == "__main__":
    main()
