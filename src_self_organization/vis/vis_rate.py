"""Plot FlowGenesis rate ablation: one accuracy and one population figure per N0.

Expected layout:
  ROOT/fashionmnist/rate/value_<rate>/neurons_<N0>/seed_<seed>/learning_curves.pt

Each rate is plotted as mean +/- 1 sample SD across available seeds.
Incomplete runs are supported: each epoch uses all seeds that reached it.
"""
from pathlib import Path
import argparse
import re
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

RATES = (1.0, 2.5, 5.0, 10.0)
N0_VALUES = (20, 40, 80, 160)
COLORS = {1.0: 'steelblue', 2.5: 'darkorange', 5.0: 'seagreen', 10.0: 'firebrick'}


def numeric_array(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    try:
        a = np.asarray(value, dtype=float).reshape(-1)
        return a if len(a) else None
    except (TypeError, ValueError):
        return None


def read_curve(path):
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError('not a dictionary')
        epochs = numeric_array(payload.get('epoch'))
        acc = numeric_array(payload.get('test_accuracy'))
        neurons = None
        for key in ('num_neurons', 'neurons', 'neuron_history', 'population'):
            neurons = numeric_array(payload.get(key))
            if neurons is not None:
                break
        if epochs is None or acc is None:
            raise ValueError('missing epoch or test_accuracy')
        return epochs, acc, neurons
    except Exception as exc:
        warnings.warn(f'Skipping {path}: {exc}')
        return None


def collect(root, dataset, n0, rate):
    base = root / dataset / 'rate'
    # Match numeric value rather than depending on whether folder is value_5 or value_5.0.
    folders = []
    if base.exists():
        for p in base.glob('value_*'):
            try:
                if abs(float(p.name.removeprefix('value_')) - rate) < 1e-9:
                    folders.append(p)
            except ValueError:
                continue
    by_metric = {'accuracy': [], 'neurons': []}
    seen_seeds = set()
    for folder in sorted(folders):
        for run in sorted((folder / f'neurons_{n0}').glob('seed_*')):
            if not re.fullmatch(r'seed_\d+', run.name):
                continue
            seed = int(run.name.split('_')[-1])
            if seed in seen_seeds:
                warnings.warn(f'Duplicate seed {seed} for N0={n0}, rate={rate}; using first')
                continue
            file = run / 'learning_curves.pt'
            if not file.is_file():
                continue
            loaded = read_curve(file)
            if loaded is None:
                continue
            seen_seeds.add(seed)
            epochs, acc, neurons = loaded
            for metric, vals in (('accuracy', acc), ('neurons', neurons)):
                if vals is None:
                    continue
                length = min(len(epochs), len(vals))
                # Map epochs explicitly: partial runs need not have identical lengths.
                samples = {}
                for ep, val in zip(epochs[:length], vals[:length]):
                    if np.isfinite(ep) and np.isfinite(val) and ep >= 1 and ep == int(ep):
                        samples[int(ep)] = float(val)
                if samples:
                    by_metric[metric].append(samples)
    return by_metric


def summarize(runs):
    if not runs:
        return None
    all_epochs = sorted(set().union(*(r.keys() for r in runs)))
    means, stds, counts = [], [], []
    for epoch in all_epochs:
        values = np.asarray([r[epoch] for r in runs if epoch in r], dtype=float)
        means.append(float(values.mean()))
        stds.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
        counts.append(len(values))
    return np.array(all_epochs), np.array(means), np.array(stds), np.array(counts)


def make_plot(n0, metric, results, out, dataset):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plotted = False
    for rate in RATES:
        runs = results[rate][metric]
        s = summarize(runs)
        if s is None:
            print(f'[WARN] N0={n0}, rate={rate:g}: no {metric} data')
            continue
        x, mean, std, counts = s
        color = COLORS[rate]
        ax.plot(x, mean, lw=1.5, color=color, label=f'Rate={rate:g} (seeds={len(runs)})')
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        print(f'N0={n0} rate={rate:g} {metric}: {len(runs)} seeds, last epoch={x[-1]:g}, seeds at last epoch={counts[-1]}')
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test accuracy (%)' if metric == 'accuracy' else 'Number of neurons')
    ax.set_title(f'{dataset.upper()} — $N_0={n0}$')
    ax.grid(alpha=0.20, linestyle='--')
    ax.legend(frameon=False)
    fig.tight_layout()
    stem = f'{dataset}_rate_N{n0}_{"test_accuracy" if metric == "accuracy" else "neurons"}'
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{stem}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {stem}.png/.pdf')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', '--root', dest='root', type=Path,
                        default=Path('data/neurogenesis/rate_ablations'))
    parser.add_argument('--dataset', default='fashionmnist')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    root = args.root.expanduser()
    out = args.out.expanduser() if args.out else root / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    for n0 in N0_VALUES:
        results = {rate: collect(root, args.dataset, n0, rate) for rate in RATES}
        for metric in ('accuracy', 'neurons'):
            make_plot(n0, metric, results, out, args.dataset)
    print(f'\nFigures directory: {out}')


if __name__ == '__main__':
    main()
