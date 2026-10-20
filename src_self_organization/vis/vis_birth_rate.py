#!/usr/bin/env python3
"""Visualize FlowGenesis birth-threshold ablation (partial runs supported).

Example:
 python -m src_self_organization.vis.vis_birth_rate \
   --folder data/neurogenesis/birth_ablations --n0 20
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt


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


def locate(root, dataset, threshold, n0, seed):
    suffix = Path(f'value_{threshold:g}') / f'neurons_{n0}' / f'seed_{seed}'
    for base in (root / dataset / 'birth', root / 'birth', root):
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
    for key in ('history', 'epoch_history', 'development_history'):
        rows = stats.get(key)
        if isinstance(rows, (tuple, list)) and rows and isinstance(rows[0], dict):
            for field in ('neurons', 'num_neurons', 'population', 'alive'):
                if all(field in row for row in rows):
                    return vector([row[field] for row in rows])
    return None


def load_run(run):
    curves = load_dict(run / 'learning_curves.pt')
    accuracy = None
    for key in ('test_accuracy', 'test_acc', 'test_accuracy_history'):
        accuracy = vector(curves.get(key))
        if accuracy is not None:
            break
    neurons = extract_neurons(run, curves)
    epochs = vector(curves.get('epoch'))
    if epochs is None:
        length = len(accuracy) if accuracy is not None else (len(neurons) if neurons is not None else 0)
        epochs = np.arange(length, dtype=float)
    result = {}
    for metric, values in (('accuracy', accuracy), ('neurons', neurons)):
        if values is None:
            continue
        n = min(len(epochs), len(values))
        x, y = epochs[:n], values[:n]
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if not len(x):
            continue
        order = np.argsort(x)
        x, y = x[order], y[order]
        x, unique = np.unique(x, return_index=True)
        result[metric] = (x, y[unique])
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
    counts = np.sum(np.isfinite(values), axis=0)
    keep = counts > 0
    grid, values, counts = grid[keep], values[:, keep], counts[keep]
    means = np.nansum(values, axis=0) / counts
    sd = np.array([np.std(values[np.isfinite(values[:, j]), j], ddof=1)
                   if counts[j] > 1 else 0.0 for j in range(len(grid))])
    return grid, means, sd, counts, len(series)


def plot(results, thresholds, metric, ylabel, out, n0, dataset):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    COLORS = {0.15: 'black', 0.25: 'steelblue', 0.35: 'orange', 0.45: 'red'}
    found = False
    for threshold in thresholds:
        color = COLORS.get(round(threshold, 8), None)
        summary = aggregate(results.get(threshold, []), metric)
        if summary is None:
            print(f'[WARN] No {metric} for birth threshold={threshold:g}')
            continue
        x, mean, sd, counts, n = summary
        ax.plot(x, mean, lw=1.3, color=color,
                label=rf'$\theta_b={threshold:g}$ ($n={n}$)')
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.065, linewidth=0)
        print(f'Birth={threshold:g}, {metric}: {n} seed(s), {len(x)} epochs')
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
        target = out / f'birth_threshold_n0_{n0}_{metric}.{ext}'
        fig.savefig(target, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f'Saved: {target}')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', '--root', type=Path,
                   default=Path('data/neurogenesis/birth_ablations'))
    p.add_argument('--dataset', default='fashionmnist')
    p.add_argument('--n0', type=int, default=20)
    p.add_argument('--thresholds', nargs='+', type=float, default=[0.15, 0.25, 0.35, 0.45])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()
    root = args.folder.expanduser()
    out = args.out.expanduser() if args.out else root / 'plots'
    results = {}
    for threshold in args.thresholds:
        runs = []
        for seed in args.seeds:
            run = locate(root, args.dataset, threshold, args.n0, seed)
            if run is None:
                print(f'[MISSING] birth={threshold:g}, N0={args.n0}, seed={seed}')
                continue
            try:
                data = load_run(run)
                if data:
                    runs.append(data)
                    print(f'[OK] birth={threshold:g}, seed={seed}: {run}')
                else:
                    print(f'[WARN] No metrics in {run}')
            except Exception as exc:
                print(f'[WARN] Cannot load {run}: {exc}')
        results[threshold] = runs
    if not any(results.values()):
        raise RuntimeError(f'No learning_curves.pt found under {root}; expected '
                           '<dataset>/birth/value_<threshold>/neurons_<N0>/seed_<seed>/')
    out.mkdir(parents=True, exist_ok=True)
    plot(results, args.thresholds, 'neurons', 'Number of neurons', out, args.n0, args.dataset)
    plot(results, args.thresholds, 'accuracy', 'Test accuracy (%)', out, args.n0, args.dataset)

if __name__ == '__main__':
    main()

