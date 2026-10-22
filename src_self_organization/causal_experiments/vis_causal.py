"""Visualize causal parent-selection experiments: two figures, mean ± 1 SD.

Run from project root:
python -m src_self_organization.causal_experiments.vis_causal \
  --root data/neurogenesis/selection_sanity_v3 --dataset fashionmnist --n0 20
"""
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path('data/neurogenesis/selection_sanity_v3'))
parser.add_argument('--dataset', default='fashionmnist')
parser.add_argument('--n0', type=int, default=20)
parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
parser.add_argument('--out', type=Path, default=None)
args = parser.parse_args()

# Same palette as vis_n0.py: black, steelblue, orange, red.
CONDITIONS = ['high_flux', 'reverse_flux', 'random_all', 'random_all_matched']
COLORS = dict(zip(CONDITIONS, ['black', 'steelblue', 'orange', 'red']))
LABELS = {'high_flux': 'High flux', 'reverse_flux': 'Reverse flux',
          'random_all': 'Random all', 'random_all_matched': 'Random all (matched)'}
root = args.root.expanduser()
out = args.out.expanduser() if args.out else root / 'plots'
out.mkdir(parents=True, exist_ok=True)


def numeric(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    try:
        a = np.asarray(value, dtype=float).reshape(-1)
        return a if a.size else None
    except (ValueError, TypeError):
        return None


def aggregate(runs):
    # Align by epoch rather than truncating across gaps or unequal lengths.
    common = sorted(set.intersection(*(set(x) for x in runs)))
    if not common:
        return None
    values = np.array([[run[e] for e in common] for run in runs], dtype=float)
    return (np.array(common), values.mean(axis=0),
            values.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(len(common)))


results = {}
for condition in CONDITIONS:
    acc_runs, neuron_runs = [], []
    for seed in args.seeds:
        run_dir = (root / args.dataset / 'selection' / f'value_{condition}' /
                   f'neurons_{args.n0}' / f'seed_{seed}')
        path = run_dir / 'learning_curves.pt'
        if not path.is_file():
            print(f'[MISSING] {path}')
            continue
        try:
            curve = torch.load(path, map_location='cpu', weights_only=False)
            epochs = numeric(curve.get('epoch'))
            accuracy = numeric(curve.get('test_accuracy'))
            neurons = numeric(curve.get('num_neurons'))
            if epochs is None or accuracy is None:
                print(f'[WARN] Missing epoch/accuracy: {path}')
                continue
            acc_runs.append(dict(zip(epochs[:len(accuracy)].astype(int), accuracy)))
            if neurons is not None:
                neuron_runs.append(dict(zip(epochs[:len(neurons)].astype(int), neurons)))
            else:
                print(f'[WARN] Missing neuron history: {path}')
        except Exception as exc:
            print(f'[WARN] Cannot read {path}: {exc}')
    results[condition] = {
        'accuracy': aggregate(acc_runs) if acc_runs else None,
        'neurons': aggregate(neuron_runs) if neuron_runs else None,
    }
    print(f'{condition}: accuracy={len(acc_runs)} seeds; neurons={len(neuron_runs)} seeds')

for metric, ylabel, filename in [
    ('accuracy', 'Test accuracy (%)', 'fashionmnist_causal_test_accuracy'),
    ('neurons', 'Number of neurons', 'fashionmnist_causal_neurons'),
]:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    plotted = False
    for condition in CONDITIONS:
        result = results[condition][metric]
        if result is None:
            continue
        epochs, mean, std = result
        ax.plot(epochs, mean, linewidth=1.3, color=COLORS[condition], label=LABELS[condition])
        ax.fill_between(epochs, mean - std, mean + std,
                        color=COLORS[condition], alpha=0.065, linewidth=0)
        plotted = True
    if not plotted:
        print(f'[WARN] No data available for {metric}; skipping figure')
        plt.close(fig)
        continue
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.20, linestyle='--')
    ax.legend(frameon=False)
    fig.tight_layout()
    base = out / filename.replace('fashionmnist', args.dataset)
    fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {base}.png and {base}.pdf')
