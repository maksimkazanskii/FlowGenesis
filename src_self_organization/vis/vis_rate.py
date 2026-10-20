#!/usr/bin/env python3
"""Plot FashionMNIST rate ablation: accuracy and neuron count.

For N0=160, overlay rates 1,2.5,5,10 with mean +/- SD across seeds.
Supports partial runs by interpolating onto the shared observed epoch grid.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

COLORS = {1.0: 'black', 2.5: 'steelblue', 5.0: 'orange', 10.0: 'red'}

def array(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    try:
        a = np.asarray(x, dtype=float).reshape(-1)
        return a if a.size else None
    except (ValueError, TypeError):
        return None

def neurons(run, curves):
    for key in ('num_neurons', 'neurons', 'neuron_history', 'population'):
        a = array(curves.get(key))
        if a is not None:
            return a
    path = run / 'statistics.pt'
    if not path.exists():
        return None
    stats = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(stats, dict):
        return None
    for key in ('neurons', 'neuron_history', 'num_neurons_history', 'population',
                'population_history', 'alive_history'):
        a = array(stats.get(key))
        if a is not None:
            return a
    for key in ('history', 'epoch_history', 'development_history'):
        rows = stats.get(key)
        if isinstance(rows, (list, tuple)) and rows and isinstance(rows[0], dict):
            for field in ('neurons', 'num_neurons', 'population', 'alive'):
                if all(field in row for row in rows):
                    return array([row[field] for row in rows])
    return None

def collect(root, rate, n0, metric):
    folder = root / f'value_{rate:g}' / f'neurons_{n0}'
    series = []
    for run in sorted(folder.glob('seed_*')):
        path = run / 'learning_curves.pt'
        if not path.exists():
            continue
        try:
            curves = torch.load(path, map_location='cpu', weights_only=False)
            if not isinstance(curves, dict):
                raise ValueError('learning_curves.pt is not a dictionary')
            y = array(curves.get('test_accuracy')) if metric == 'accuracy' else neurons(run, curves)
            if y is None:
                continue
            x = array(curves.get('epoch'))
            if x is None:
                x = np.arange(len(y), dtype=float)
            n = min(len(x), len(y))
            x, y = x[:n], y[:n]
            good = np.isfinite(x) & np.isfinite(y)
            x, y = x[good], y[good]
            if not len(x):
                continue
            order = np.argsort(x)
            x, y = x[order], y[order]
            x, indices = np.unique(x, return_index=True)
            y = y[indices]
            series.append((x, y))
        except Exception as exc:
            print(f'[WARN] {run}: {exc}')
    if not series:
        return None
    # Use epochs actually present in the runs; retain partial runs rather than truncating all.
    grid = np.unique(np.concatenate([x for x, _ in series]))
    vals = np.full((len(series), len(grid)), np.nan)
    for i, (x, y) in enumerate(series):
        if len(x) == 1:
            vals[i, grid == x[0]] = y[0]
        else:
            inside = (grid >= x[0]) & (grid <= x[-1])
            vals[i, inside] = np.interp(grid[inside], x, y)
    counts = np.sum(np.isfinite(vals), axis=0)
    valid = counts > 0
    grid, vals, counts = grid[valid], vals[:, valid], counts[valid]
    mean = np.nansum(vals, axis=0) / counts
    std = np.zeros(len(grid))
    for j in range(len(grid)):
        if counts[j] > 1:
            std[j] = np.std(vals[np.isfinite(vals[:, j]), j], ddof=1)
    return grid, mean, std, len(series)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', '--root', type=Path,
                   default=Path('data/neurogenesis/rate_ablations/fashionmnist/rate'))
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--n0', nargs='+', type=int, default=[160],
                   help='Initial neuron counts; default: 20. E.g. --n0 10 20 40 80 160')
    p.add_argument('--rates', nargs='+', type=float, default=[1, 2.5, 5, 10])
    args = p.parse_args()
    root = args.folder.expanduser()
    if not root.is_dir():
        p.error(f'Experiment directory does not exist: {root}')
    out = (args.out or root / 'plots').expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for n0 in args.n0:
        for metric, ylabel, suffix in (
            ('accuracy', 'Test accuracy (%)', 'test_accuracy'),
            ('neurons', 'Number of neurons', 'neurons'),
        ):
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            plotted = 0
            for rate in args.rates:
                result = collect(root, rate, n0, metric)
                if result is None:
                    print(f'[WARN] rate={rate:g}, N0={n0}: no {metric} trajectories')
                    continue
                x, mean, sd, n = result
                color = COLORS.get(rate, None)
                ax.plot(x, mean, lw=1.3, color=color, label=rf'$r={rate:g}$ (n={n})')
                ax.fill_between(x, mean-sd, mean+sd, color=color, alpha=0.065, linewidth=0)
                print(f'N0={n0}, rate={rate:g}, {metric}: {n} seed(s), {len(x)} epochs')
                plotted += 1
            if not plotted:
                plt.close(fig)
                continue
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.20, linestyle='--')
            ax.legend(frameon=False)
            fig.tight_layout()
            for ext in ('png', 'pdf'):
                dest = out / f'fashionmnist_rate_n0_{n0}_{suffix}.{ext}'
                fig.savefig(dest, dpi=300 if ext == 'png' else None, bbox_inches='tight')
                print(f'Saved: {dest}')
            plt.close(fig)

if __name__ == '__main__':
    main()
