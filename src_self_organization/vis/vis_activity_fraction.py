"""Plot FashionMNIST death_activity_fraction ablation, matching vis_n0 style.

Two separate plots (PNG and PDF): test accuracy and neuron count vs epoch,
mean +/- sample SD across seeds. Supports partial runs.

Usage:
 python src_self_organization/vis_activity_fraction.py
 python src_self_organization/vis_activity_fraction.py --root data/neurogenesis/activity_fraction_ablations
"""
from pathlib import Path
import argparse
import re
import numpy as np
import torch
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--root', '--folder', dest='root', type=Path,
               default=Path('data/neurogenesis/activity_fraction_ablations'))
p.add_argument('--out', type=Path, default=None)
p.add_argument('--dataset', default='fashionmnist')
p.add_argument('--n0', type=int, default=20)
a = p.parse_args()
root = a.root.expanduser()
out = a.out.expanduser() if a.out else root / 'plots'
out.mkdir(parents=True, exist_ok=True)
fractions = [0.05, 0.10, 0.20, 0.40]
colors = {0.05:'black', 0.10:'steelblue', 0.20:'orange', 0.40:'red'}

def numeric(v):
    if v is None: return None
    if torch.is_tensor(v): v = v.detach().cpu().numpy()
    try: x = np.asarray(v, dtype=float).reshape(-1)
    except (TypeError, ValueError): return None
    return x if x.size else None

def neurons_from(run, curves):
    for key in ('num_neurons','neurons','neuron_history','population'):
        if isinstance(curves, dict) and key in curves:
            v = numeric(curves[key])
            if v is not None: return v
    stats_file = run / 'statistics.pt'
    if not stats_file.exists(): return None
    stats = torch.load(stats_file, map_location='cpu', weights_only=False)
    if not isinstance(stats, dict): return None
    for key in ('neurons','neuron_history','num_neurons_history','population',
                'population_history','alive_history'):
        v = numeric(stats.get(key))
        if v is not None: return v
    for key in ('history','epoch_history','development_history'):
        rows = stats.get(key)
        if isinstance(rows,(tuple,list)) and rows and isinstance(rows[0],dict):
            for nk in ('neurons','num_neurons','population','alive'):
                if nk in rows[0]:
                    return numeric([row[nk] for row in rows if nk in row])
    return None

def find_runs(fraction):
    base = root / a.dataset / 'activity_fraction'
    if not base.exists():
        return []
    found = []
    # Discover rather than assume exact value-folder formatting (0.1 vs 0.10).
    for f in base.rglob('learning_curves.pt'):
        if f.parent.name.startswith('seed_') and f.parent.parent.name == f'neurons_{a.n0}':
            parent = f.parent.parent.parent
            match = re.search(r'(?:value[_=]|fraction[_=])([0-9.]+)', parent.name)
            if match and abs(float(match.group(1))-fraction)<1e-9:
                found.append(f.parent)
    return sorted(set(found), key=lambda d: str(d))

def aggregate(runs, field):
    pairs = []
    for run in runs:
        try: curves = torch.load(run/'learning_curves.pt', map_location='cpu', weights_only=False)
        except Exception as e:
            print(f'[WARN] Cannot read {run}: {e}')
            continue
        x = numeric(curves.get('epoch'))
        y = numeric(curves.get('test_accuracy')) if field=='accuracy' else neurons_from(run, curves)
        if x is None or y is None:
            print(f'[WARN] Missing {field} trajectory: {run}')
            continue
        n = min(len(x),len(y))
        pairs.append((x[:n],y[:n]))
    if not pairs: return None
    # Align on epochs shared by all runs; avoids silently mixing different epochs.
    common = sorted(set(pairs[0][0]).intersection(*(set(x) for x,_ in pairs[1:])))
    if not common: return None
    vals = np.stack([np.array([dict(zip(x,y))[e] for e in common]) for x,y in pairs])
    return np.asarray(common), vals.mean(axis=0), vals.std(axis=0,ddof=1) if len(vals)>1 else np.zeros(len(common)), len(vals)

runs_by_fraction = {f:find_runs(f) for f in fractions}
if not any(runs_by_fraction.values()):
    raise SystemExit(f'No learning_curves.pt files found in {root / a.dataset / "activity_fraction"}. '
                     'Expected value_<fraction>/neurons_<N0>/seed_<seed>/learning_curves.pt')
for f, runs in runs_by_fraction.items():
    print(f'death_activity_fraction={f:g}: {len(runs)} saved run(s)')

for field, ylabel, stem in [
    ('accuracy','Test accuracy (%)','fashionmnist_activity_fraction_test_accuracy'),
    ('neurons','Number of neurons','fashionmnist_activity_fraction_neurons'),
]:
    fig, ax = plt.subplots(figsize=(7.2,4.8))
    n_plotted = 0
    for f in fractions:
        data = aggregate(runs_by_fraction[f],field)
        if data is None: continue
        x,mean,sd,n = data
        ax.plot(x,mean,lw=1.3,color=colors[f],label=rf'$f_{{\mathrm{{activity}}}}={f:g}$')
        ax.fill_between(x,mean-sd,mean+sd,color=colors[f],alpha=0.065,linewidth=0)
        print(f'  {field}, fraction={f:g}: {n} seed(s), {len(x)} epochs')
        n_plotted += 1
    if n_plotted==0:
        plt.close(fig)
        print(f'[WARN] No {field} trajectories found; plot skipped')
        continue
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.20,linestyle='--')
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ('png','pdf'):
        fig.savefig(out/f'{stem}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out / stem}.png and .pdf')
