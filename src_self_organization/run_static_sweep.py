#!/usr/bin/env python3
"""Run fixed-width baselines with the existing FlowGenesis ablation trainer.

Example:
 python run_static_sweep.py --dataset cifar10 --seeds 1 2 3 4 5
 python run_static_sweep.py --dataset fashionmnist --seeds 1 2 3 4 5

Run from the project root so `src_self_organization` is importable.
"""
import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

MODULE = 'src_self_organization.train_neurogenesis_ablation'
WIDTHS = [20, 40, 60, 80, 120, 160, 200]


def load_summary(path):
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
        return data if isinstance(data, dict) and data.get('final_accuracy') is not None else None
    except (OSError, RuntimeError, ValueError, EOFError):
        return None


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def summarize(rows, widths):
    result = []
    for width in widths:
        selected = [r for r in rows if int(r['width']) == width]
        if not selected:
            continue
        final = np.array([float(r['final_accuracy']) for r in selected])
        best = np.array([float(r['best_accuracy']) for r in selected])
        result.append({
            'width': width,
            'n_seeds': len(selected),
            'final_mean': float(final.mean()),
            'final_std': float(final.std(ddof=1)) if len(final) > 1 else float('nan'),
            'best_mean': float(best.mean()),
            'best_std': float(best.std(ddof=1)) if len(best) > 1 else float('nan'),
            'final_min': float(final.min()),
            'final_max': float(final.max()),
        })
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=['fashionmnist', 'cifar10', 'mnist'])
    p.add_argument('--widths', nargs='+', type=int, default=WIDTHS)
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--output-dir', default='data/neurogenesis/static_sweep')
    p.add_argument('--module', default=MODULE, help='Existing trainer module; change if located elsewhere')
    p.add_argument('--rerun', action='store_true', help='Rerun even when a valid summary exists')
    p.add_argument('--keep-going', action='store_true', help='Continue after a failed run')
    args = p.parse_args()

    if any(n <= 0 for n in args.widths):
        p.error('Widths must be positive integers')
    root = Path(args.output_dir)
    records = []
    raw_fields = ['dataset', 'width', 'seed', 'final_accuracy', 'best_accuracy',
                  'final_loss', 'best_epoch', 'final_neurons', 'final_epoch', 'summary_path']
    agg_fields = ['width', 'n_seeds', 'final_mean', 'final_std', 'best_mean',
                  'best_std', 'final_min', 'final_max']

    def collect():
        records.clear()
        for width in args.widths:
            for seed in args.seeds:
                folder = root / args.dataset / 'regime' / 'value_static' / f'neurons_{width}' / f'seed_{seed}'
                summary_file = folder / 'summary.pt'
                data = load_summary(summary_file)
                if data is None:
                    continue
                if int(data.get('final_neurons', -1)) != width:
                    raise RuntimeError(f'Not static! {summary_file}: final_neurons={data.get("final_neurons")}')
                records.append({
                    'dataset': args.dataset, 'width': width, 'seed': seed,
                    'final_accuracy': float(data['final_accuracy']),
                    'best_accuracy': float(data['best_accuracy']),
                    'final_loss': float(data['final_loss']),
                    'best_epoch': int(data['best_epoch']),
                    'final_neurons': int(data['final_neurons']),
                    'final_epoch': int(data['final_epoch']),
                    'summary_path': str(summary_file),
                })
        write_csv(root / f'{args.dataset}_static_runs.csv', records, raw_fields)
        aggregated = summarize(records, args.widths)
        write_csv(root / f'{args.dataset}_static_summary.csv', aggregated, agg_fields)
        print('\nCurrent results (FINAL test accuracy):', flush=True)
        for r in aggregated:
            print(f"  N={r['width']:3d} | {r['final_mean']:.2f} ± {r['final_std']:.2f}% | seeds={r['n_seeds']}", flush=True)

    collect()
    for width in args.widths:
        for seed in args.seeds:
            folder = root / args.dataset / 'regime' / 'value_static' / f'neurons_{width}' / f'seed_{seed}'
            if not args.rerun and load_summary(folder / 'summary.pt') is not None:
                print(f'[SKIP] N={width} seed={seed} (completed)', flush=True)
                continue
            cmd = [sys.executable, '-u', '-m', args.module,
                   '--dataset', args.dataset, '--hidden_neurons', str(width),
                   '--seed', str(seed), '--ablation', 'regime', '--value', 'static',
                   '--output_dir', str(root)]
            folder.mkdir(parents=True, exist_ok=True)
            log_path = folder / 'run.log'
            print(f'[START] N={width} seed={seed} -> {log_path}', flush=True)
            with log_path.open('w') as log:
                proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                      env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            if proc.returncode != 0:
                print(f'[FAILED] N={width} seed={seed}, exit={proc.returncode}; see {log_path}', flush=True)
                if not args.keep_going:
                    collect()
                    sys.exit(proc.returncode)
            else:
                print(f'[DONE] N={width} seed={seed}', flush=True)
            collect()
    print(f'\nSaved: {root / (args.dataset + "_static_runs.csv")}', flush=True)
    print(f'Saved: {root / (args.dataset + "_static_summary.csv")}', flush=True)


if __name__ == '__main__':
    main()
