"""Plot FlowGenesis connectivity (beta) ablation at fixed N0 across seeds.

Expected directory layout (from train_neurogenesis_ablation.py):
  ROOT/fashionmnist/beta/value_0.1/neurons_20/seed_0/learning_curves.pt

Also accepts ROOT already pointing at fashionmnist/, beta/, or value_* parent.
Reads partial learning_curves.pt files saved after each epoch.
"""
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', '--root', dest='folder', type=Path,
                   default=Path('data/neurogenesis/beta_ablation'))
    p.add_argument('--dataset', default='fashionmnist')
    p.add_argument('--n0', type=int, default=20)
    p.add_argument('--betas', type=float, nargs='+', default=[0.1, 0.2, 0.4, 0.8])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--out', type=Path, default=None)
    return p.parse_args()


def vector(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    try:
        a = np.asarray(x, dtype=float).reshape(-1)
    except (ValueError, TypeError):
        return None
    return a if a.size else None


def locate(root, dataset, beta, n0, seed):
    value = f'value_{beta:g}'
    suffix = Path(value) / f'neurons_{n0}' / f'seed_{seed}' / 'learning_curves.pt'
    bases = [root / dataset / 'beta', root / 'beta', root]
    for base in bases:
        candidate = base / suffix
        if candidate.is_file():
            return candidate
    return None


def load_run(path):
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError('expected dictionary')
        epochs = vector(payload.get('epoch'))
        accuracy = vector(payload.get('test_accuracy'))
        neurons = None
        for key in ('num_neurons', 'neurons', 'neuron_history', 'population'):
            neurons = vector(payload.get(key))
            if neurons is not None:
                break
        if epochs is None or accuracy is None:
            raise ValueError('missing epoch or test_accuracy')
        n = min(len(epochs), len(accuracy))
        if neurons is not None:
            n = min(n, len(neurons))
        epochs, accuracy = epochs[:n], accuracy[:n]
        neurons = neurons[:n] if neurons is not None else None
        valid = np.isfinite(epochs) & np.isfinite(accuracy)
        if neurons is not None:
            valid &= np.isfinite(neurons)
        return epochs[valid], accuracy[valid], (neurons[valid] if neurons is not None else None)
    except Exception as e:
        print(f'[WARN] Cannot load {path}: {e}')
        return None


def aggregate(runs, metric):
    # Align by actual epoch number. Missing and shorter seeds contribute only
    # to epochs they contain; no artificial truncation to the shortest run.
    by_epoch = {}
    for epochs, acc, neurons in runs:
        values = acc if metric == 'accuracy' else neurons
        if values is None:
            continue
        for epoch, value in zip(epochs, values):
            by_epoch.setdefault(int(epoch), []).append(float(value))
    if not by_epoch:
        return None
    xs = np.array(sorted(by_epoch))
    means = np.array([np.mean(by_epoch[e]) for e in xs])
    stds = np.array([np.std(by_epoch[e], ddof=1) if len(by_epoch[e]) > 1 else 0.
                     for e in xs])
    counts = np.array([len(by_epoch[e]) for e in xs])
    return xs, means, stds, counts


def plot(results, betas, metric, ylabel, out, n0):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    colors = plt.get_cmap('viridis')(np.linspace(0.12, 0.86, len(betas)))
    found = False
    for beta, color in zip(betas, colors):
        runs = results.get(beta, [])
        summary = aggregate(runs, metric)
        if summary is None:
            continue
        found = True
        x, mean, sd, counts = summary
        ax.plot(x, mean, lw=1.6, color=color,
                label=rf'$\beta={beta:g}$ (seeds={len(runs)})')
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.14,
                        linewidth=0)
    if not found:
        print(f'[WARN] No {metric} histories found; skipping figure')
        plt.close(fig)
        return
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(rf'Connectivity ablation ($N_0={n0}$)')
    ax.grid(alpha=0.2, linestyle='--')
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        target = out / f'connectivity_n0_{n0}_{metric}.{ext}'
        fig.savefig(target, dpi=300, bbox_inches='tight')
        print(f'Saved {target}')
    plt.close(fig)


def main():
    args = parse_args()
    root = args.folder.expanduser()
    out = args.out.expanduser() if args.out else root / 'plots'
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for beta in args.betas:
        runs = []
        for seed in args.seeds:
            path = locate(root, args.dataset, beta, args.n0, seed)
            if path is None:
                print(f'[MISSING] beta={beta:g}, N0={args.n0}, seed={seed}')
                continue
            run = load_run(path)
            if run is not None:
                runs.append(run)
                print(f'[OK] beta={beta:g}, seed={seed}, epochs={len(run[0])}: {path}')
        results[beta] = runs
    if not any(results.values()):
        raise RuntimeError(f'No learning_curves.pt found under {root}; '
                           'expected <dataset>/beta/value_<beta>/neurons_<N0>/seed_<seed>/')
    plot(results, args.betas, 'accuracy', 'Test accuracy (%)', out, args.n0)
    plot(results, args.betas, 'neurons', 'Number of neurons', out, args.n0)


if __name__ == '__main__':
    main()
