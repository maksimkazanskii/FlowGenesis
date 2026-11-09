#!/usr/bin/env python3
"""Plot Progressive FlowGenesis vs PNN from partial or completed runs.

Usage:
  python vis_progressive.py --input-folder data/progressive_cifar100
  python vis_progressive.py --input-folder ~/Downloads/progressive_results --output-folder ./figures

Looks recursively for pnn/seed_*/results.csv|json and flowgenesis/seed_*/results.csv|json.
Produces exactly two figures, each with two panels (PDF and PNG).
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

METHODS = ('pnn', 'flowgenesis')
COLORS = {'pnn': '#377eb8', 'flowgenesis': '#e68629'}
LABELS = {'pnn': 'PNN', 'flowgenesis': 'Progressive FlowGenesis'}
METRICS = ('average_accuracy', 'active_parameters', 'module_neurons')
EXTRA_METRICS = ('current_task_accuracy', 'new_parameters', 'births', 'deaths')


def discover(root):
    candidates = list(root.rglob('results.csv')) + list(root.rglob('results.json'))
    by_run = defaultdict(dict)
    for path in candidates:
        match = re.search(r'seed_(\d+)$', path.parent.name)
        method = path.parent.parent.name.lower()
        if not match or method not in METHODS:
            continue
        key = (method, int(match.group(1)))
        # Prefer JSON because it contains the complete per-task accuracy vectors.
        by_run[key][path.suffix] = path
    return {key: paths.get('.json', paths.get('.csv')) for key, paths in by_run.items()}


def load_run(path):
    if path.suffix == '.json':
        rows = json.loads(path.read_text(encoding='utf-8'))
    else:
        with path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    cleaned = {}
    for row in rows:
        try:
            task = int(row['task'])
            values = {metric: float(row[metric]) for metric in METRICS}
            for metric in EXTRA_METRICS:
                try:
                    values[metric] = float(row[metric])
                except (KeyError, ValueError, TypeError):
                    values[metric] = float('nan')
            if task > 0 and all(np.isfinite(v) for v in values.values()):
                cleaned[task] = values
        except (ValueError, TypeError, KeyError):
            continue
    return cleaned


def summarize(runs, method, metric):
    task_values = defaultdict(list)
    for (m, seed), records in runs.items():
        if m != method:
            continue
        for task, row in records.items():
            task_values[task].append(row[metric])
    tasks = np.array(sorted(task_values), dtype=int)
    means = np.array([np.mean(task_values[t]) for t in tasks])
    stds = np.array([np.std(task_values[t], ddof=1) if len(task_values[t]) > 1 else 0.0 for t in tasks])
    counts = np.array([len(task_values[t]) for t in tasks])
    return tasks, means, stds, counts


def draw(ax, runs, metric, ylabel, scale=1.0):
    for method in METHODS:
        tasks, mean, std, count = summarize(runs, method, metric)
        if not len(tasks):
            continue
        color = COLORS[method]
        ax.plot(tasks, mean / scale, '-o', lw=1.8, markersize=3,
                label=LABELS[method], color=color)
        # SD across available seeds at each task; one seed -> no band.
        lower = mean - std
        upper = mean + std

        if metric == 'module_neurons':
            lower = np.maximum(lower, 0)

        ax.fill_between(tasks, lower / scale, upper / scale,
                        color=color, alpha=0.16, linewidth=0)
    ax.set_xlabel('Task number')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    ax.set_xlim(left=1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


def save_figure(fig, output, basename):
    for ext in ('pdf', 'png'):
        path = output / f'{basename}.{ext}'
        fig.savefig(path, dpi=220, bbox_inches='tight')
        print('Saved:', path)
    plt.close(fig)


def _mean_sd(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return float('nan'), float('nan'), 0
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else float('nan'), len(arr)


def _fmt(mean, sd, n, digits=2):
    if n == 0:
        return 'NA'
    if n == 1:
        return f'{mean:.{digits}f} (n=1)'
    return f'{mean:.{digits}f} ± {sd:.{digits}f} (n={n})'


def _paired_test(runs, task, metric):
    """FlowGenesis minus PNN, pairing only seeds with both completed task rows."""
    seeds = sorted({seed for method, seed in runs if method == 'pnn'} &
                   {seed for method, seed in runs if method == 'flowgenesis'})
    diffs = []
    for seed in seeds:
        a = runs[('flowgenesis', seed)].get(task)
        b = runs[('pnn', seed)].get(task)
        if a is not None and b is not None:
            delta = a.get(metric, float('nan')) - b.get(metric, float('nan'))
            if np.isfinite(delta):
                diffs.append(delta)
    mean, sd, n = _mean_sd(diffs)
    ci_low = ci_high = p_value = float('nan')
    if n >= 2 and scipy_stats is not None:
        if sd == 0:
            ci_low = ci_high = mean
            p_value = 1.0 if mean == 0 else 0.0
        else:
            se = sd / np.sqrt(n)
            tcrit = scipy_stats.t.ppf(0.975, df=n-1)
            ci_low, ci_high = mean-tcrit*se, mean+tcrit*se
            p_value = float(scipy_stats.ttest_1samp(diffs, 0).pvalue)
    return dict(paired_n=n, delta_mean=mean, delta_sd=sd,
                ci95_low=ci_low, ci95_high=ci_high, p_value=p_value)


def _value_at_task(records, task, metric):
    if task not in records:
        return float('nan')
    if metric == 'cumulative_neurons':
        if any(t not in records for t in range(1, task+1)):
            return float('nan')
        return float(sum(records[t]['module_neurons'] for t in range(1, task+1)))
    if metric == 'mean_module_neurons':
        if any(t not in records for t in range(1, task+1)):
            return float('nan')
        return float(np.mean([records[t]['module_neurons'] for t in range(1, task+1)]))
    if metric == 'mean_accuracy_over_tasks':
        if any(t not in records for t in range(1, task+1)):
            return float('nan')
        return float(np.mean([records[t]['average_accuracy'] for t in range(1, task+1)]))
    return records[task].get(metric, float('nan'))


def write_tables(runs, output):
    """Write main-paper ready descriptive and paired statistics, including partial runs."""
    max_task = max(max(records) for records in runs.values())
    # Every observed task is included; final-task summaries are marked by task number.
    metrics = [
        ('average_accuracy', 'Average accuracy (%)'),
        ('current_task_accuracy', 'Current task accuracy (%)'),
        ('active_parameters', 'Cumulative active parameters'),
        ('module_neurons', 'New module neurons'),
        ('cumulative_neurons', 'Cumulative neurons'),
        ('mean_module_neurons', 'Mean neurons per module'),
        ('mean_accuracy_over_tasks', 'Mean accuracy across task checkpoints (%)'),
    ]
    descriptions = []
    comparisons = []
    for task in range(1, max_task+1):
        for metric, label in metrics:
            vals = {}
            for method in METHODS:
                observations = [_value_at_task(rec, task, metric)
                                for (m, _), rec in runs.items() if m == method]
                vals[method] = _mean_sd(observations)
                mu, sd, n = vals[method]
                descriptions.append(dict(task=task, metric=metric, method=method,
                                         n=n, mean=mu, sd=sd))
            # Pairing for derived metrics requires same-seed per-run derived values.
            paired_runs = {}
            for key, rec in runs.items():
                val = _value_at_task(rec, task, metric)
                paired_runs[key] = {task: {metric: val}} if np.isfinite(val) else {}
            pair = _paired_test(paired_runs, task, metric)
            comparisons.append(dict(task=task, metric=metric, **pair))

    def save_csv(name, rows):
        with (output/name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    save_csv('table_descriptive_by_task.csv', descriptions)
    save_csv('table_paired_by_task.csv', comparisons)

    # For publication: use pre-specified endpoints 20/50 when present;
    # also show latest common task (exploratory, useful for partial runs).
    complete_sets = [set(rec) for rec in runs.values()]
    common = set.intersection(*complete_sets) if complete_sets else set()
    endpoints = sorted({t for t in (20, 50) if t <= max_task} |
                       ({max(common)} if common else set()))
    lines = [
        '# Progressive FlowGenesis vs PNN: table statistics', '',
        'Values are mean ± sample SD across available seeds; n is shown.',
        'Paired differences are FlowGenesis minus PNN on seeds with both results.',
        '95% confidence intervals and two-sided paired t-test p-values require SciPy and at least two pairs.',
        'Tests are unadjusted and exploratory; pre-specify primary endpoints and correct for multiple testing if making confirmatory claims.',
        'Partial runs may have different seed counts at each task.', '',
    ]
    for task in endpoints:
        lines += [f'## Task {task}', '',
                  '| Metric | PNN | FlowGenesis | Paired Δ [95% CI] | p |',
                  '|---|---:|---:|---:|---:|']
        for metric, label in metrics:
            stats_by_method = {m: next(r for r in descriptions if r['task']==task and r['metric']==metric and r['method']==m)
                               for m in METHODS}
            pair = next(r for r in comparisons if r['task']==task and r['metric']==metric)
            a = stats_by_method['pnn']; b = stats_by_method['flowgenesis']
            pa = _fmt(a['mean'], a['sd'], a['n']); pb = _fmt(b['mean'], b['sd'], b['n'])
            if pair['paired_n']:
                delta = f"{pair['delta_mean']:.2f} (n={pair['paired_n']})"
                if np.isfinite(pair['ci95_low']):
                    delta += f" [{pair['ci95_low']:.2f}, {pair['ci95_high']:.2f}]"
            else:
                delta = 'NA'
            pstr = f"{pair['p_value']:.4g}" if np.isfinite(pair['p_value']) else 'NA'
            lines.append(f'| {label} | {pa} | {pb} | {delta} | {pstr} |')
        lines.append('')
    lines += [
        '### Notes', '',
        '- Average accuracy is the average test accuracy over all tasks seen at that checkpoint (task-incremental evaluation).',
        '- Cumulative active parameters include the shared encoder and all allocated task modules; they are not FLOPs.',
        '- Cumulative neurons sum the neurons allocated to task modules through the specified task.',
        '- Mean accuracy across task checkpoints is a trajectory summary, not final accuracy.',
        '- Different methods can have different numbers of completed seeds; paired tests use only matched seeds.',
    ]
    (output/'table_main.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Saved:', output/'table_main.md')
    print('Saved:', output/'table_descriptive_by_task.csv')
    print('Saved:', output/'table_paired_by_task.csv')
    if scipy_stats is None:
        print('SciPy unavailable: p-values and confidence intervals omitted (pip install scipy).')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-folder', '--folder', required=True, type=Path,
                        help='Root directory containing pnn/seed_*/results.json and flowgenesis/seed_*/results.json (recursive search).')
    parser.add_argument('--output-folder', type=Path, default=None,
                        help='Where to write figures (default: <input-folder>/figures).')
    args = parser.parse_args()
    root = args.input_folder.expanduser().resolve()
    if not root.is_dir():
        parser.error(f'Input folder does not exist: {root}')
    output = (args.output_folder.expanduser().resolve() if args.output_folder else root / 'figures')
    found = discover(root)
    if not found:
        parser.error(f'No pnn/seed_*/results.csv|json or flowgenesis/seed_*/results.csv|json found beneath {root}')

    runs = {}
    for (method, seed), path in sorted(found.items()):
        rows = load_run(path)
        if rows:
            runs[(method, seed)] = rows
            print(f'{method:12s} seed={seed}: {len(rows)} completed tasks, last={max(rows)} | {path}')
    if not runs:
        parser.error('Result files were found but contained no valid task rows.')
    output.mkdir(parents=True, exist_ok=True)
    write_tables(runs, output)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 120})

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    draw(axs[0], runs, 'average_accuracy', 'Average test accuracy (%)')
    draw(axs[1], runs, 'active_parameters', 'Cumulative active parameters (millions)', scale=1e6)
    axs[0].set_title('(a) Accuracy across tasks')
    axs[1].set_title('(b) Cumulative parameter cost')
    axs[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(w_pad=2.5)
    save_figure(fig, output, 'figure1_performance_capacity')

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
    draw(axs[0], runs, 'module_neurons', 'Neurons in new task module')
    # Cumulative neurons must be summed within each individual seed, not summed across seeds.
    cumulative = {}
    for key, records in runs.items():
        total = 0.0
        cumulative[key] = {}
        for task in sorted(records):
            # Missing tasks invalidate the cumulative trajectory from that point.
            if task != len(cumulative[key]) + 1:
                break
            total += records[task]['module_neurons']
            cumulative[key][task] = {'module_neurons': total}
    draw(axs[1], cumulative, 'module_neurons', 'Cumulative neurons across modules')
    axs[0].set_title('(a) Neurons allocated per task')
    axs[1].set_title('(b) Total allocated neurons')
    axs[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(w_pad=2.5)
    save_figure(fig, output, 'figure2_architectural_dynamics')
    print('\nBands: mean ± sample SD across available seeds per task (no band with one seed).')
    print('Partial runs supported: task counts can differ across seeds.')


if __name__ == '__main__':
    main()
