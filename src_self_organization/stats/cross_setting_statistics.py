#!/usr/bin/env python3
"""Cross-HYPERPARAMETER CKA and prediction agreement for FlowGenesis.

Run separately on one ablation root containing value_*/neurons_*/seed_*/final_model.pt.
Uses the existing comparative_statistics.py for checkpoint parsing and forward pass.
Only compares DIFFERENT value_* settings at the SAME N0, matching seeds by default.
Outputs pair-level, seed-level, setting-level and setting-pair-level CSVs.
"""
from __future__ import annotations
import argparse
import importlib.util
import itertools
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


def load_comparative(path):
    spec = importlib.util.spec_from_file_location('flowgenesis_comparative', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setting_of(run):
    matches = re.findall(r'(?:^|/)value_([^/]+)(?:/|$)', run['condition'])
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one value_* directory: {run['id']}")
    return matches[0]


def setting_sort(value):
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--comparative-script', type=Path,
                   default=Path('src_self_organization/stats/comparative_statistics.py'))
    p.add_argument('--dataset', default='fashionmnist')
    p.add_argument('--checkpoint', default='final_model.pt')
    p.add_argument('--probe-size', type=int, default=1000)
    p.add_argument('--probe-seed', type=int, default=20260829)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--data-root', default='./data')
    p.add_argument('--relaxation-steps', type=int, default=10)
    p.add_argument('--gamma', type=float, default=0.2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--all-seed-pairs', action='store_true',
                   help='Compare all seeds across settings; default is matched seeds only')
    args = p.parse_args()
    root = args.root.resolve()
    c = load_comparative(args.comparative_script.resolve())
    runs = c.load_runs(root, args.checkpoint)
    for run in runs:
        run['setting'] = setting_of(run)
    groups = defaultdict(list)
    for run in runs:
        groups[run['n0']].append(run)
    if not any(len({r['setting'] for r in g}) > 1 for g in groups.values()):
        raise RuntimeError('No two distinct value_* settings share an N0.')

    dataset = c.get_test_dataset(args.dataset, args.data_root)
    # Use the same probe indices as the existing comparative script when possible.
    probe_file = root / 'comparative_probe_indices.npy'
    if probe_file.exists():
        existing = np.load(probe_file)
        if len(existing) != min(args.probe_size, len(dataset)):
            probe_file = root / 'cross_setting_probe_indices.npy'
    indices = c.choose_probe_indices(len(dataset), args.probe_size,
                                      args.probe_seed, probe_file)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    output = {}
    for i, run in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] value={run['setting']} N0={run['n0']} seed={run['seed']}", flush=True)
        output[run['id']] = c.run_checkpoint_on_probe(
            run['tensors'], loader, args.device, args.relaxation_steps, args.gamma)
    # Avoid loading a large cache from unrelated earlier runs.
    pairs = []
    for n0, g in sorted(groups.items()):
        for a, b in itertools.combinations(g, 2):
            if a['setting'] == b['setting']:
                continue
            if not args.all_seed_pairs and a['seed'] != b['seed']:
                continue
            if setting_sort(a['setting']) > setting_sort(b['setting']):
                a, b = b, a
            oa, ob = output[a['id']], output[b['id']]
            pairs.append(dict(n0=n0, setting_a=a['setting'], setting_b=b['setting'],
                              seed_a=a['seed'], seed_b=b['seed'],
                              model_a=a['id'], model_b=b['id'],
                              cka=c.linear_cka(oa['hidden'], ob['hidden']),
                              agreement=100.0 * c.prediction_agreement(
                                  oa['predictions'], ob['predictions'])))
    if not pairs:
        raise RuntimeError('No cross-setting model pairs found; check seed overlap.')
    pair_df = pd.DataFrame(pairs)
    pair_df.to_csv(root / 'cross_setting_pairs.csv', index=False)
    pair_df.groupby(['n0','setting_a','setting_b'], as_index=False).agg(
        cka_mean=('cka','mean'), cka_sd=('cka','std'),
        agreement_mean=('agreement','mean'), agreement_sd=('agreement','std'),
        n_pairs=('cka','count')).to_csv(root / 'cross_setting_pair_summary.csv', index=False)

    # Each source model contributes one average over other settings. For matched
    # seeds this is one comparison per other setting; avoids pseudoreplicating
    # pair scores in the row-level standard deviation.
    model_rows = []
    for n0, g in sorted(groups.items()):
        for run in g:
            relevant = pair_df[(pair_df['model_a'] == run['id']) |
                               (pair_df['model_b'] == run['id'])]
            if relevant.empty:
                continue
            # Average within each OTHER setting first, giving equal setting weight.
            tmp = relevant.copy()
            tmp['other_setting'] = np.where(tmp['model_a'] == run['id'],
                                            tmp['setting_b'], tmp['setting_a'])
            means = tmp.groupby('other_setting')[['cka','agreement']].mean()
            model_rows.append(dict(n0=n0, setting=run['setting'], seed=run['seed'],
                                   model=run['id'], n_other_settings=len(means),
                                   cka=float(means['cka'].mean()),
                                   agreement=float(means['agreement'].mean())))
    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(root / 'cross_setting_model_level.csv', index=False)
    summary = model_df.groupby(['n0','setting'], as_index=False).agg(
        cka_mean=('cka','mean'), cka_sd=('cka','std'),
        agreement_mean=('agreement','mean'), agreement_sd=('agreement','std'),
        n_seeds=('seed','nunique'), n_other_settings=('n_other_settings','min'))
    summary = summary.sort_values(['n0','setting'], key=lambda s: s.map(setting_sort) if s.name == 'setting' else s)
    summary.to_csv(root / 'cross_setting_summary.csv', index=False)
    print('\nCross-setting summary (agreement in %):')
    print(summary.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'\nSaved four CSVs under {root}')
    print('IMPORTANT: metrics are cross-SETTING, not cross-seed; SD is across source models/seeds.')


if __name__ == '__main__':
    main()
