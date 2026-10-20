#!/usr/bin/env python3
"""Plot static vs FlowGenesis final-test accuracy, grouped by initial width.

Run from project root:
 python src_self_organization/vis_flowgenesis_pareto.py --dataset cifar10

Reads completed summary.pt first; optionally uses the last saved learning-curves
entry for in-progress runs (marked separately). Does not train anything.
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_pt(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except (OSError, EOFError, RuntimeError, ValueError, KeyError, ImportError) as exc:
        print(f'[WARN] Cannot read {path}: {exc}')
        return None


def scalar(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def folder_num(name, prefix):
    m = re.fullmatch(rf'{re.escape(prefix)}_(\d+)', name)
    return int(m.group(1)) if m else None



def active_parameters(seed_dir, expected_width):
    """Count forward-active parameters from the final checkpoint's actual mask.

    Returns None if the checkpoint is missing or represents a different
    (post-evaluation) architecture from the reported final test accuracy.
    """
    path = seed_dir / 'final_model.pt'
    if not path.is_file():
        return None
    payload = load_pt(path)
    if not isinstance(payload, dict):
        return None
    state = payload.get('state_dict', payload)
    if not isinstance(state, dict):
        return None

    # Find the recurrent connectivity mask, preserving any module prefix.
    masks = [(k, v) for k, v in state.items()
             if isinstance(v, torch.Tensor) and k.split('.')[-1] == 'mask'
             and v.ndim == 2 and v.shape[0] == v.shape[1]]
    if len(masks) != 1:
        print(f'[WARN] Cannot identify unique recurrent mask: {path}')
        return None
    mask_key, mask = masks[0]
    n = mask.shape[0]
    if n != expected_width:
        print(f'[WARN] Checkpoint width {n} != evaluated width {expected_width}: {path}. '
              'Likely a post-final structural update; active parameters unavailable.')
        return None
    prefix = mask_key[:-len('mask')]
    weight_key = prefix + 'W'
    if weight_key not in state or state[weight_key].shape != mask.shape:
        print(f'[WARN] Missing corresponding recurrent W: {path}')
        return None

    # Architecture from model_neurogenesis_new.py:
    # input_layer (bias=False), recurrent W*mask, recurrent bias,
    # output_layer (weight + bias).
    expected = [prefix + 'input_layer.weight', prefix + 'bias',
                prefix + 'output_layer.weight', prefix + 'output_layer.bias']
    if any(k not in state or not isinstance(state[k], torch.Tensor) for k in expected):
        print(f'[WARN] Unexpected architecture; cannot count safely: {path}')
        return None
    recurrent = int(torch.count_nonzero(mask).item())
    other = sum(int(state[k].numel()) for k in expected)
    return recurrent + other


def attach_active_counts(records):
    for r in records:
        r['active_parameters'] = active_parameters(Path(r['path']).parent, r['final'])
    available = sum(r['active_parameters'] is not None for r in records)
    print(f'[INFO] Active parameter counts: {available}/{len(records)} available')


def format_params(value):
    return 'N/A' if value is None else f'{value:,.0f}'

def collect(root, dataset, kind, include_partial):
    records = []
    if not root.exists():
        print(f'[WARN] Missing {kind} directory: {root}')
        return records
    for summary_path in sorted(root.rglob('summary.pt')):
        seed_dir = summary_path.parent
        seed = folder_num(seed_dir.name, 'seed')
        width_dir = seed_dir.parent
        initial = folder_num(width_dir.name, 'neurons')
        if seed is None or initial is None:
            continue
        data = load_pt(summary_path)
        if not isinstance(data, dict) or data.get('final_accuracy') is None:
            continue
        try:
            width = int(data.get('final_neurons', initial))
            acc = scalar(data['final_accuracy'])
        except (TypeError, ValueError):
            continue
        if kind == 'static' and width != initial:
            print(f'[WARN] Static width changed; skipping {summary_path}')
            continue
        records.append(dict(kind=kind, initial=initial, final=width, seed=seed,
                            accuracy=acc, partial=False, epoch=data.get('final_epoch'),
                            path=str(summary_path)))
    if include_partial:
        completed = {(r['initial'], r['seed']) for r in records}
        for curve_path in sorted(root.rglob('learning_curves.pt')):
            seed_dir = curve_path.parent
            seed = folder_num(seed_dir.name, 'seed')
            initial = folder_num(seed_dir.parent.name, 'neurons')
            if seed is None or initial is None or (initial, seed) in completed:
                continue
            data = load_pt(curve_path)
            if not isinstance(data, dict):
                continue
            accs = data.get('test_accuracy', [])
            neurons = data.get('num_neurons', [])
            if not accs:
                continue
            try:
                width = initial if kind == 'static' else int(neurons[-1])
                acc = scalar(accs[-1])
            except (IndexError, ValueError, TypeError):
                continue
            records.append(dict(kind=kind, initial=initial, final=width, seed=seed,
                                accuracy=acc, partial=True, epoch=len(accs),
                                path=str(curve_path)))
    print(f'[INFO] {kind}: {len(records)} runs ({sum(r["partial"] for r in records)} partial)')
    return records


def group_stats(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    output = []
    for value, group in sorted(groups.items()):
        acc = np.array([r['accuracy'] for r in group], dtype=float)
        final = np.array([r['final'] for r in group], dtype=float)
        params = np.array([r['active_parameters'] for r in group if r.get('active_parameters') is not None], dtype=float)
        output.append(dict(active_parameters=float(params.mean()) if len(params) == len(group) else None,
                           active_parameter_std=float(params.std(ddof=1)) if len(params)>1 and len(params)==len(group) else None,
                           group=value, x=float(final.mean()), x_std=float(final.std(ddof=1)) if len(final)>1 else 0.,
                           y=float(acc.mean()), y_std=float(acc.std(ddof=1)) if len(acc)>1 else 0.,
                           n=len(group), partial=sum(r['partial'] for r in group)))
    return output


def pareto_indices(points):
    # Maximize accuracy, minimize final width; ties retain best accuracy at each width.
    selected = []
    best = -np.inf
    for i in sorted(range(len(points)), key=lambda i: (points[i]['x'], -points[i]['y'])):
        p = points[i]
        if p['y'] > best + 1e-10:
            selected.append(i)
            best = p['y']
    return selected


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(rows, output, dataset, show_seeds=False):
    static = [r for r in rows if r['kind']=='static']
    flow = [r for r in rows if r['kind']=='flowgenesis']
    sg = group_stats(static, 'initial')
    fg = group_stats(flow, 'initial')
    if not sg and not fg:
        raise SystemExit('No data found. Check --static-root and --flow-root paths.')
    fig, ax = plt.subplots(figsize=(10, 6.5))
    if show_seeds:
        for group, color, marker in [(static, 'tab:blue','o'),(flow,'tab:orange','s')]:
            if group:
                ax.scatter([r['final'] for r in group], [r['accuracy'] for r in group],
                           c=color, marker=marker, alpha=.18, s=25, zorder=1)
    if sg:
        ax.errorbar([p['x'] for p in sg], [p['y'] for p in sg],
                    yerr=[p['y_std'] for p in sg], fmt='o-', color='tab:blue',
                    capsize=3, linewidth=1.5, label='Static (final accuracy)', zorder=3)
        idx = pareto_indices(sg)
        ax.step([sg[i]['x'] for i in idx], [sg[i]['y'] for i in idx],
                where='post', linestyle='--', color='tab:blue', alpha=.45,
                label='Static Pareto frontier', zorder=2)
    if fg:
        ax.errorbar([p['x'] for p in fg], [p['y'] for p in fg],
                    xerr=[p['x_std'] for p in fg], yerr=[p['y_std'] for p in fg],
                    fmt='s', color='tab:orange', capsize=3, markersize=7,
                    label='FlowGenesis (grouped by initial N₀)', zorder=4)
        for p in fg:
            ax.annotate(f'N₀={p["group"]} (n={p["n"]})', (p['x'],p['y']),
                        xytext=(6,7), textcoords='offset points', fontsize=8,
                        color='tab:orange')
    # Group labels: measured active parameters (not allocated dense weights).
    for group, color, offset in [(sg, 'tab:blue', (-14, -23)),
                                 (fg, 'tab:orange', (7, -22))]:
        for p in group:
            ax.annotate(f'P={format_params(p["active_parameters"])}',
                        (p['x'], p['y']), xytext=offset,
                        textcoords='offset points', fontsize=7.5, color=color)
    ax.set(xlabel='Final number of neurons', ylabel='Final test accuracy (%)',
           title=f'{dataset.upper()}: Static vs FlowGenesis — accuracy / final width')
    ax.grid(alpha=.2)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(output/'accuracy_vs_final_neurons.png', dpi=200)
    fig.savefig(output/'accuracy_vs_final_neurons.pdf')
    plt.close(fig)

    # True accuracy–active-parameter Pareto view.
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for groups, color, marker, label in [
        (sg, 'tab:blue', 'o', 'Static'),
        (fg, 'tab:orange', 's', 'FlowGenesis'),
    ]:
        valid = [p for p in groups if p['active_parameters'] is not None]
        if not valid:
            continue
        ax.errorbar([p['active_parameters'] for p in valid],
                    [p['y'] for p in valid],
                    yerr=[p['y_std'] for p in valid],
                    fmt=marker, color=color, capsize=3, label=label)
        for p in valid:
            ax.annotate(f'N₀={p["group"]} (P={format_params(p["active_parameters"])})',
                        (p['active_parameters'], p['y']), xytext=(6, 6),
                        textcoords='offset points', fontsize=8, color=color)
    ax.set(xlabel='Active forward-pass parameters',
           ylabel='Final test accuracy (%)',
           title=f'{dataset.upper()}: accuracy / active parameters')
    ax.grid(alpha=.2)
    if ax.has_data():
        ax.legend()
    fig.tight_layout()
    fig.savefig(output/'accuracy_vs_active_parameters.png', dpi=200)
    fig.savefig(output/'accuracy_vs_active_parameters.pdf')
    plt.close(fig)

    # Separate view: where each FlowGenesis run started and ended.
    if fg:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.errorbar([p['group'] for p in fg], [p['y'] for p in fg],
                    yerr=[p['y_std'] for p in fg], fmt='s-', color='tab:orange',
                    capsize=3, label='FlowGenesis final test accuracy')
        for p in fg:
            ax.annotate(f'final N={p["x"]:.1f}', (p['group'],p['y']),
                        xytext=(4,8), textcoords='offset points', fontsize=8)
        ax.set(xlabel='FlowGenesis initial neuron count (N₀)',
               ylabel='Final test accuracy (%)',
               title=f'{dataset.upper()}: FlowGenesis across initial widths')
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output/'flowgenesis_by_initial_width.png', dpi=200)
        plt.close(fig)
    return sg, fg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='cifar10')
    p.add_argument('--flow-root', default=None)
    p.add_argument('--static-root', default=None)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--include-partial', action='store_true',
                   help='Include in-progress runs; do not use for final paper figures')
    p.add_argument('--show-seeds', action='store_true')
    args = p.parse_args()
    flow_root = Path(args.flow_root or f'data/neurogenesis/{args.dataset}/so/n0_ablation/{args.dataset}')
    static_root = Path(args.static_root or f'data/neurogenesis/static_sweep/{args.dataset}/regime/value_static')
    out = Path(args.output_dir or f'data/neurogenesis/pareto/{args.dataset}')
    out.mkdir(parents=True, exist_ok=True)
    static = collect(static_root, args.dataset, 'static', args.include_partial)
    flow = collect(flow_root, args.dataset, 'flowgenesis', args.include_partial)
    all_rows = static+flow
    attach_active_counts(all_rows)
    write_csv(out/'individual_runs.csv', all_rows)
    sg, fg = plot(all_rows, out, args.dataset, args.show_seeds)
    stats = [dict(kind=kind, initial_width=p['group'], mean_final_width=p['x'],
                  std_final_width=p['x_std'], mean_accuracy=p['y'], std_accuracy=p['y_std'],
                  seeds=p['n'], partial_runs=p['partial'])
             for kind, group in [('static',sg),('flowgenesis',fg)] for p in group]
    write_csv(out/'grouped_results.csv', stats)
    for row in stats:
        print(f"{row['kind']:11s} N0={row['initial_width']:3d} "
              f"finalN={row['mean_final_width']:.1f} "
              f"acc={row['mean_accuracy']:.2f} ± {row['std_accuracy']:.2f}% "
              f"seeds={row['seeds']} partial={row['partial_runs']}")
    print(f'\nPlots and CSVs saved in {out}')
    if args.include_partial:
        print('[NOTE] In-progress runs are included and must not be interpreted as final accuracy.')
    print('[NOTE] Width frontier is width-based; the second plot uses active parameters.')
    print('[NOTE] N/A means a matching final-evaluation checkpoint was unavailable.')


if __name__ == '__main__':
    main()
