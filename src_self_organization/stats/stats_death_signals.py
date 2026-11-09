#!/usr/bin/env python3
"""Summarize FlowGenesis death-selection runs across seeds.

Reads summary.pt (preferred) and learning_curves.pt from the training script.
Produces per-seed and mean ± sample-SD CSV files, plus comparisons to original.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch

REGIMES = ('original', 'unrestricted_random', 'shuffle_activity', 'shuffle_flux')
METRICS = ('final_accuracy', 'final_neurons', 'best_accuracy', 'best_neurons',
           'final_total_born', 'final_total_pruned', 'best_epoch')


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def locate(root, dataset, regime, n0, seed):
    tail = Path(f'value_{regime}') / f'neurons_{n0}' / f'seed_{seed}'
    for base in (root / dataset / 'death_selection', root / 'death_selection', root):
        candidate = base / tail
        if (candidate / 'summary.pt').exists() or (candidate / 'learning_curves.pt').exists():
            return candidate
    return None


def extract(folder):
    summary_path = folder / 'summary.pt'
    curves_path = folder / 'learning_curves.pt'
    result = {}
    if summary_path.exists():
        s = load(summary_path)
        for key in METRICS:
            if s.get(key) is not None:
                result[key] = float(s[key])
    if curves_path.exists():
        c = load(curves_path)
        accuracy = np.asarray(c.get('test_accuracy', []), dtype=float)
        neurons = np.asarray(c.get('num_neurons', []), dtype=float)
        if accuracy.size:
            result.setdefault('final_accuracy', float(accuracy[-1]))
            result.setdefault('best_accuracy', float(np.max(accuracy)))
            result.setdefault('best_epoch', float(np.argmax(accuracy) + 1))
        if neurons.size:
            # These are post-structural-check populations, unlike summary.pt
            # final_neurons, which corresponds to the evaluated architecture.
            result['post_update_final_neurons'] = float(neurons[-1])
    return result


def write_csv(path, rows, columns):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', '--root', type=Path, default=Path('results/death_selection'))
    p.add_argument('--dataset', default='fashionmnist')
    p.add_argument('--n0', type=int, default=20)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--regimes', nargs='+', choices=REGIMES, default=list(REGIMES))
    p.add_argument('--out', type=Path, default=None)
    args = p.parse_args()
    out = args.out or args.folder / 'stats'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for regime in args.regimes:
        for seed in args.seeds:
            folder = locate(args.folder, args.dataset, regime, args.n0, seed)
            if folder is None:
                print(f'[MISSING] {regime} seed={seed}')
                continue
            values = extract(folder)
            if not values:
                print(f'[EMPTY] {folder}')
                continue
            rows.append(dict(regime=regime, seed=seed, **values))
    if not rows:
        raise SystemExit('No usable runs found')
    metrics = list(METRICS) + ['post_update_final_neurons']
    write_csv(out / 'death_signals_per_seed.csv', rows, ['regime', 'seed'] + metrics)
    summaries = []
    for regime in args.regimes:
        group = [r for r in rows if r['regime'] == regime]
        if not group:
            continue
        record = {'regime': regime, 'n_runs': len(group)}
        for metric in metrics:
            values = np.array([r[metric] for r in group if metric in r], dtype=float)
            record[f'{metric}_n'] = len(values)
            record[f'{metric}_mean'] = float(values.mean()) if len(values) else ''
            record[f'{metric}_sd'] = float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else '')
        summaries.append(record)
    cols = ['regime', 'n_runs'] + [f'{m}_{suffix}' for m in metrics for suffix in ('n', 'mean', 'sd')]
    write_csv(out / 'death_signals_summary.csv', summaries, cols)
    original = next((r for r in summaries if r['regime'] == 'original'), None)
    if original:
        comparisons = []
        for r in summaries:
            if r['regime'] == 'original':
                continue
            record = {'regime': r['regime'], 'reference': 'original'}
            for metric in ('final_accuracy', 'final_neurons', 'best_accuracy', 'post_update_final_neurons'):
                a, b = r[f'{metric}_mean'], original[f'{metric}_mean']
                record[f'{metric}_delta'] = a - b if a != '' and b != '' else ''
            comparisons.append(record)
        write_csv(out / 'death_signals_vs_original.csv', comparisons,
                  ['regime', 'reference'] + [f'{m}_delta' for m in ('final_accuracy', 'final_neurons', 'best_accuracy', 'post_update_final_neurons')])
    print(f'Wrote results to {out}')
    for r in summaries:
        def fmt(m):
            mean, sd = r[f'{m}_mean'], r[f'{m}_sd']
            return 'NA' if mean == '' else f'{mean:.2f} ± {sd:.2f}'
        print(f"{r['regime']:22s} n={r['n_runs']}  final accuracy={fmt('final_accuracy')}%  final neurons={fmt('final_neurons')}  best accuracy={fmt('best_accuracy')}%")
    print('Note: final_neurons requires summary.pt; post_update_final_neurons is from learning_curves.pt.')

if __name__ == '__main__':
    main()
