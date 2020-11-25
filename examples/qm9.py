# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except, abstract-method, arguments-differ
import argparse
import pickle
import subprocess
import time

import torch
from torch.autograd import profiler
from torch_geometric.data import DataLoader
from torch_geometric.datasets import QM9
from torch_geometric.nn import SchNet

from e3nn_little.nn.models import Network


def execute(args):
    path = 'QM9'
    dataset = QM9(path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    target = 7
    # Report meV instead of eV.
    units = 1000 if target in [2, 3, 4, 6, 7, 8, 9, 10] else 1

    _, datasets = SchNet.from_qm9_pretrained(path, dataset, target)
    train_dataset, val_dataset, _test_dataset = datasets

    model = Network(
        muls=(args.mul0, args.mul1, args.mul2), lmax=args.lmax, num_layers=args.num_layers, rad_gaussians=args.rad_gaussians,
        rad_hs=(args.rad_h,) * args.rad_layers + (args.rad_bottleneck,),
        mean=0, std=1, atomref=dataset.atomref(target),
        options=args.arch
    )
    model = model.to(device)

    # profile
    loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False)
    for step, data in enumerate(loader):
        with profiler.profile(use_cuda=True) as prof:
            data = data.to(device)
            pred = model(data.z, data.pos, data.batch)
            mse = (pred.view(-1) - data.y[:, target]).pow(2)
            mse.mean().backward()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10), flush=True)
        prof.export_chrome_trace("trace.json")
        if step == 2:
            break

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=25, factor=0.8, min_lr=1e-6)

    dynamics = []
    wall = time.perf_counter()
    wall_print = time.perf_counter()

    for epoch in range(args.num_epochs):

        maes = []
        loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        for step, data in enumerate(loader):
            data = data.to(device)

            pred = model(data.z, data.pos, data.batch)
            optim.zero_grad()
            (pred.view(-1) - data.y[:, target]).pow(2).mean().backward()
            optim.step()

            mae = (pred.view(-1) - data.y[:, target]).abs()
            maes += [mae.cpu().detach()]

            if time.perf_counter() - wall_print > 15:
                wall_print = time.perf_counter()
                print((
                    f'[{epoch}] ['
                    f'wall={time.perf_counter() - wall:.0f} step={step}/{len(loader)} '
                    f'mae={units * torch.cat(maes)[-200:].mean():.5f} '
                    f'lr={optim.param_groups[0]["lr"]:.1e}]'
                ), flush=True)

        train_mae = torch.cat(maes)

        maes = []
        loader = DataLoader(val_dataset, batch_size=256)
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                pred = model(data.z, data.pos, data.batch)

            mae = (pred.view(-1) - data.y[:, target]).abs()
            maes += [mae.cpu().detach()]
        val_mae = torch.cat(maes)

        dynamics += [{
            'epoch': epoch,
            'wall': time.perf_counter() - wall,
            'train_mae': units * train_mae,
            'val_mae': units * val_mae,
        }]

        print(f'[{epoch}] Target: {target:02d}, MAE TRAIN: {units * train_mae.mean():.5f} ± {units * train_mae.std():.5f}, MAE VAL: {units * val_mae.mean():.5f} ± {units * val_mae.std():.5f}', flush=True)

        scheduler.step(val_mae.pow(2).mean())

        yield {
            'args': args,
            'dynamics': dynamics,
        }


def main():
    git = {
        'log': subprocess.getoutput('git log --format="%H" -n 1 -z'),
        'status': subprocess.getoutput('git status -z'),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--mul0", type=int, default=30)
    parser.add_argument("--mul1", type=int, default=10)
    parser.add_argument("--mul2", type=int, default=0)
    parser.add_argument("--lmax", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--rad_gaussians", type=int, default=40)
    parser.add_argument("--rad_h", type=int, default=200)
    parser.add_argument("--rad_bottleneck", type=int, default=50)
    parser.add_argument("--rad_layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--arch", type=str, default="")

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
