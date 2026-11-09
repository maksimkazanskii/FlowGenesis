#!/usr/bin/env python3
"""Plot FlowGenesis pruning-signal ablations, including incomplete runs.

Example:
 python -m src_self_organization.vis.vis_death_signals \
   --folder results/death_selection --dataset fashionmnist --n0 20

Outputs two figures (PNG and PDF): neuron population and test accuracy.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

REGIMES = (
    'original', 'unrestricted_random', 'shuffle_activity',
    'shuffle_flux',
)
LABELS = {
    'original': 'Original',
    'unrestricted_random': 'Random pruning',
    'shuffle_activity': 'Shuffled activity',
    'shuffle_flux': 'Shuffled Flux',
}
COLORS = {
    'original': 'black',
    'unrestricted_random': 'steelblue',
    'shuffle_activity': 'darkorange',
    'shuffle_flux': 'forestgreen',
}


def vector(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return arr if arr.size else None


def locate(root, dataset, regime, n0, seed):
    suffix = Path(f'value_{regime}') / f'neurons_{n0}' / f'seed_{seed}'
    for base in (root / dataset / 'death_selection', root / 'death_selection', root):
        run = base / suffix
        if (run / 'learning_curves.pt').exists():
            return run
    return None


def load_dict(path):
    if not path.exists():
        return {}
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f'Expected dict in {path}')
    return obj


def extract_neurons(run, curves):
    for key in ('num_neurons', 'neurons', 'neuron_history', 'population'):
        arr = vector(curves.get(key))
        if arr is not None:
            return arr
    stats = load_dict(run / 'statistics.pt')
    for key in ('neurons', 'neuron_history', 'num_neurons_history',
                'population', 'population_history', 'alive_history'):
        arr = vector(stats.get(key))
        if arr is not None:
            return arr
    return None


def load_run(run):
    curves = load_dict(run / 'learning_curves.pt')
    epochs = vector(curves.get('epoch'))
    result = {}
    for metric, keys in (
        ('accuracy', ('test_accuracy', 'test_acc', 'test_accuracy_history')),
        ('neurons', ('num_neurons', 'neurons', 'neuron_history', 'population')),
    ):
        values = extract_neurons(run, curves) if metric == 'neurons' else None
        if metric == 'accuracy':
            for key in keys:
                values = vector(curves.get(key))
                if values is not None:
                    break
        if values is None:
            continue
        x = epochs if epochs is not None else np.arange(1, len(values) + 1, dtype=float)
        n = min(len(x), len(values))
        x, y = x[:n], values[:n]
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if not len(x):
            continue
        order = np.argsort(x)
        x, y = x[order], y[order]
        unique_x, idx = np.unique(x, return_index=True)
        result[metric] = (unique_x, y[idx])
    return result


def aggregate(runs, metric):
    series = [run[metric] for run in runs if metric in run]
    if not series:
        return None
    grid = np.unique(np.concatenate([x for x, _ in series]))
    values = np.full((len(series), len(grid)), np.nan)
    for i, (x, y) in enumerate(series):
        if len(x) == 1:
            values[i, grid == x[0]] = y[0]
        else:
            valid = (grid >= x[0]) & (grid <= x[-1])
            values[i, valid] = np.interp(grid[valid], x, y)
    counts = np.isfinite(values).sum(axis=0)
    keep = counts > 0
    grid, values, counts = grid[keep], values[:, keep], counts[keep]
    mean = np.nansum(values, axis=0) / counts
    sd = np.array([np.std(values[np.isfinite(values[:, j]), j], ddof=1)
                   if counts[j] > 1 else 0.0 for j in range(len(grid))])
    return grid, mean, sd, counts, len(series)


def plot(results, regimes, metric, ylabel, out, n0, dataset):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    found = False
    for regime in regimes:
        summary = aggregate(results.get(regime, []), metric)
        if summary is None:
            print(f'[WARN] No {metric} for {regime}')
            continue
        x, mean, sd, counts, n = summary
        color = COLORS.get(regime)
        ax.plot(x, mean, lw=1.3, color=color,
                label=f'{LABELS.get(regime, regime)} (n={n})')
        ax.fill_between(x, mean - sd, mean + sd, color=color,
                        alpha=0.065, linewidth=0)
        print(f'{regime}, {metric}: {n} seed(s), {len(x)} epochs; '
              f'final epoch uses {counts[-1]} seed(s)')
        found = True
    if not found:
        plt.close(fig)
        print(f'[WARN] No {metric} histories; plot skipped')
        return
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(rf'{dataset.upper()} — $N_0={n0}$')
    ax.grid(alpha=0.2, linestyle='--')
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        target = out / f'death_selection_n0_{n0}_{metric}.{ext}'
        fig.savefig(target, dpi=300 if ext == 'png' else None,
                    bbox_inches='tight')
        print(f'Saved: {target}')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', '--root', type=Path,
                   default=Path('results/death_selection'))
    p.add_argument('--dataset', default='fashionmnist')
    p.add_argument('--n0', type=int, default=20)
    p.add_argument('--regimes', nargs='+', choices=REGIMES, default=list(REGIMES))
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()
    root = args.folder.expanduser()
    out = args.out.expanduser() if args.out else root / 'plots'
    results = {}
    for regime in args.regimes:
        runs = []
        for seed in args.seeds:
            run = locate(root, args.dataset, regime, args.n0, seed)
            if run is None:
                print(f'[MISSING] {regime}, N0={args.n0}, seed={seed}')
                continue
            try:
                data = load_run(run)
                if data:
                    runs.append(data)
                    print(f'[OK] {regime}, seed={seed}: {run}')
                else:
                    print(f'[WARN] No metrics in {run}')
            except Exception as exc:
                print(f'[WARN] Cannot load {run}: {exc}')
        results[regime] = runs
    if not any(results.values()):
        raise RuntimeError(f'No learning_curves.pt found under {root}; expected '
                           '<dataset>/death_selection/value_<regime>/neurons_<N0>/seed_<seed>/')
    out.mkdir(parents=True, exist_ok=True)
    plot(results, args.regimes, 'neurons', 'Number of neurons', out, args.n0, args.dataset)
    plot(results, args.regimes, 'accuracy', 'Test accuracy (%)', out, args.n0, args.dataset)

if __name__ == '__main__':
    main()
