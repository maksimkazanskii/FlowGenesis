#!/usr/bin/env python3
"""Visualize warmup ablation at N0=80 using all saved partial seed curves.

Each epoch aggregates only seeds with an available saved measurement.
Running experiments must write checkpoints to disk for live partial plots.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

WARMUPS = (50000, 100000, 200000, 400000, 800000)
N0 = 80
COLORS = dict(zip(WARMUPS, ('#2166ac', '#67a9cf', '#1b9e77', '#e6ab02', '#d95f02')))


def numeric(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        result = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return result if result.size else None


def load_pt(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def neuron_history(folder, curve):
    for key in ('num_neurons', 'neurons', 'neuron_history', 'population'):
        value = numeric(curve.get(key))
        if value is not None:
            return value
    path = folder / 'statistics.pt'
    if not path.exists():
        return None
    stats = load_pt(path)
    if not isinstance(stats, dict):
        return None
    for key in ('neurons', 'neuron_history', 'num_neurons_history',
                'population', 'population_history', 'alive_history'):
        value = numeric(stats.get(key))
        if value is not None:
            return value
    for key in ('history', 'epoch_history', 'development_history'):
        rows = stats.get(key)
        if isinstance(rows, (list, tuple)) and rows and isinstance(rows[0], dict):
            for field in ('num_neurons', 'neurons', 'population', 'alive'):
                if field in rows[0]:
                    return numeric([row.get(field, np.nan) for row in rows])
    return None


def extract_metadata(path):
    """Extract parameters from any directory component, not only fixed ordering."""
    components = [p.lower() for p in path.parts]
    text = '/'.join(components)
    patterns = {
        'warmup': r'(?:warm[_-]?up|warmup)[_=-]?(\d+)',
        'n0': r'(?:neurons|n0|hidden[_-]?neurons)[_=-]?(\d+)',
        'seed': r'seed[_=-]?(\d+)',
    }
    found = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        found[name] = int(matches[-1]) if matches else None
    return found


def discover(root):
    files = sorted(root.rglob('learning_curves.pt'))
    print(f'[INFO] Found {len(files)} learning_curves.pt file(s) under {root}')
    selected = {w: [] for w in WARMUPS}
    unmatched = []
    for path in files:
        meta = extract_metadata(path.parent)
        if meta['warmup'] in selected and meta['n0'] == N0 and meta['seed'] is not None:
            selected[meta['warmup']].append((meta['seed'], path))
        else:
            unmatched.append((path, meta))
    for path, meta in unmatched[:15]:
        print(f'[UNMATCHED] {path} | parsed={meta}')
    if unmatched:
        print(f'[INFO] {len(unmatched)} file(s) did not match warmup/N0/seed identifiers.')
    return selected


def load_runs(paths):
    records = {'accuracy': [], 'neurons': []}
    seen_seeds = set()
    for seed, path in sorted(paths):
        if seed in seen_seeds:
            print(f'[WARN] Duplicate seed {seed}: skipping {path}')
            continue
        try:
            curve = load_pt(path)
            if not isinstance(curve, dict):
                raise ValueError('learning_curves.pt is not a dictionary')
            accuracy = numeric(curve.get('test_accuracy'))
            if accuracy is None:
                raise ValueError(f'missing test_accuracy; available keys={list(curve)}')
            epochs = numeric(curve.get('epoch'))
            if epochs is None:
                epochs = np.arange(1, len(accuracy) + 1, dtype=float)
                print(f'[WARN] No epoch field in {path}; using 1..{len(accuracy)}')
            length = min(len(epochs), len(accuracy))
            if not length:
                continue
            records['accuracy'].append((seed, epochs[:length], accuracy[:length]))
            neurons = neuron_history(path.parent, curve)
            if neurons is not None:
                n = min(len(epochs), len(neurons))
                records['neurons'].append((seed, epochs[:n], neurons[:n]))
            seen_seeds.add(seed)
        except Exception as exc:
            print(f'[WARN] Cannot load {path}: {exc}')
    return records


def aggregate(runs):
    """Union of saved epochs: incomplete seeds contribute wherever available."""
    if not runs:
        return None
    epoch_values = {}
    for seed, xs, ys in runs:
        for x, y in zip(xs, ys):
            if np.isfinite(x) and np.isfinite(y):
                epoch_values.setdefault(float(x), []).append(float(y))
    if not epoch_values:
        return None
    epochs = np.asarray(sorted(epoch_values))
    mean = np.asarray([np.mean(epoch_values[e]) for e in epochs])
    std = np.asarray([np.std(epoch_values[e], ddof=1) if len(epoch_values[e]) > 1
                      else 0.0 for e in epochs])
    counts = np.asarray([len(epoch_values[e]) for e in epochs])
    return epochs, mean, std, counts


def plot(data, metric, ylabel, output):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for warmup in WARMUPS:
        result = aggregate(data[warmup][metric])
        if result is None:
            continue
        x, mean, std, counts = result
        color = COLORS[warmup]
        ax.plot(x, mean, color=color, linewidth=1.7,
                label=f'{warmup:,} steps (seeds={counts.max()}, last={counts[-1]})')
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.13)
        plotted = True
    if plotted:
        ax.set(xlabel='Epoch', ylabel=ylabel,
               title=f'Fashion-MNIST | N₀={N0} | Warmup ablation')
        ax.grid(alpha=0.2, linestyle='--')
        ax.legend(frameon=False, title='Warmup')
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            dest = output.with_suffix('.' + extension)
            fig.savefig(dest, dpi=300, bbox_inches='tight')
            print(f'[SAVED] {dest}')
    plt.close(fig)
    return plotted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', '--root', type=Path,
                        default=Path('data/neurogenesis/warmup_ablations'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    root = args.folder.expanduser()
    if not root.is_dir():
        parser.error(f'Directory not found: {root}')
    output = args.out.expanduser() if args.out else root / 'plots'
    output.mkdir(parents=True, exist_ok=True)
    found = discover(root)
    data = {warmup: load_runs(paths) for warmup, paths in found.items()}
    for warmup in WARMUPS:
        print(f'[INFO] warmup={warmup}: '
              f'{len(data[warmup]["accuracy"])} accuracy seed(s), '
              f'{len(data[warmup]["neurons"])} neuron seed(s)')
    if not any(data[w]['accuracy'] for w in WARMUPS):
        print('\n[ERROR] No matching saved curves could be loaded.')
        print('Run: find data/neurogenesis -name learning_curves.pt | head -30')
        print('Also inspect output directories and training logs for save locations.')
        return 1
    plot(data, 'accuracy', 'Test accuracy (%)', output / 'warmup_test_accuracy_N80')
    if not plot(data, 'neurons', 'Number of neurons', output / 'warmup_neurons_N80'):
        print('[WARN] No per-epoch neuron histories available.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
