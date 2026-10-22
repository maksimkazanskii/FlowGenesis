#!/usr/bin/env python3
"""Analyze FlowGenesis selection ablations from summary.pt and learning_curves.pt.

Usage:
 python stats_flowgenesis_causal.py --root data/neurogenesis/ablations --dataset fashionmnist
 python stats_flowgenesis_causal.py --root data/neurogenesis/ablations --dataset fashionmnist --n0 20 --late-fraction 0.2
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

POLICIES = ('high_flux', 'random', 'reverse_flux')
METRICS = ('final_accuracy', 'best_accuracy', 'final_neurons', 'best_neurons',
           'final_total_born', 'final_total_pruned', 'late_neurons_mean',
           'late_neurons_sd', 'late_accuracy_mean', 'late_accuracy_sd',
           'late_net_drift', 'late_turnover', 'path_length', 'last_change_fraction')

def load_pt(path):
    # Only load experiment files that you generated and trust.
    return torch.load(path, map_location='cpu', weights_only=False)

def number(v):
    try:
        if v is None: return float('nan')
        return float(v)
    except (ValueError, TypeError): return float('nan')

def summarize(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a): return (float('nan'), float('nan'), 0)
    return float(a.mean()), float(a.std(ddof=1)) if len(a)>1 else float('nan'), len(a)

def fmt(x):
    m,s,n = summarize(x)
    return f'{m:.3f} ± {s:.3f} (n={n})' if n>1 else f'{m:.3f} (n={n})' if n else 'NA'

def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/neurogenesis/ablations'))
    p.add_argument('--dataset', default='fashionmnist')
    p.add_argument('--n0', type=int, nargs='*', help='Optional initial neuron sizes')
    p.add_argument('--late-fraction', type=float, default=0.2, help='Fraction of final epochs used for late-window statistics')
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--no-plots', action='store_true')
    args=p.parse_args()
    if not 0<args.late_fraction<=1: p.error('--late-fraction must be in (0,1]')
    base=args.root/args.dataset/'selection'
    out=args.out or args.root/args.dataset/'selection_analysis'
    runs=[]; curves={}
    for policy in POLICIES:
        for summary_path in sorted((base/f'value_{policy}').glob('neurons_*/seed_*/summary.pt')):
            folder=summary_path.parent
            try:
                n0=int(folder.parent.name.split('_')[-1]); seed=int(folder.name.split('_')[-1])
                if args.n0 and n0 not in args.n0: continue
                summary=load_pt(summary_path)
                curve_path=folder/'learning_curves.pt'
                curve=load_pt(curve_path) if curve_path.exists() else {}
                r={'dataset':args.dataset,'policy':policy,'n0':n0,'seed':seed}
                for metric in METRICS[:6]: r[metric]=number(summary.get(metric))
                n=np.asarray(curve.get('num_neurons', []),dtype=float)
                acc=np.asarray(curve.get('test_accuracy', []),dtype=float)
                k=max(1,int(np.ceil(len(n)*args.late_fraction))) if len(n) else 0
                tail=n[-k:] if k else np.array([])
                r['late_neurons_mean']=float(tail.mean()) if len(tail) else float('nan')
                r['late_neurons_sd']=float(tail.std(ddof=1)) if len(tail)>1 else float('nan')
                ak=max(1,int(np.ceil(len(acc)*args.late_fraction))) if len(acc) else 0
                atail=acc[-ak:] if ak else np.array([])
                r['late_accuracy_mean']=float(atail.mean()) if len(atail) else float('nan')
                r['late_accuracy_sd']=float(atail.std(ddof=1)) if len(atail)>1 else float('nan')
                r['path_length']=float(np.abs(np.diff(n)).sum()) if len(n)>1 else float('nan')
                events=curve.get('structure_events',summary.get('structure_events',[])) or []
                if events:
                    max_examples=number(summary.get('final_examples_seen'))
                    if not np.isfinite(max_examples):
                        max_examples=max(number(e.get('examples_seen')) for e in events)
                    cutoff=max_examples*(1-args.late_fraction)
                    late=[e for e in events if number(e.get('examples_seen'))>cutoff]
                    births=sum(number(e.get('births',0)) for e in late)
                    deaths=sum(number(e.get('deaths',0)) for e in late)
                    r['late_net_drift']=(births-deaths)/len(late) if late else float('nan')
                    r['late_turnover']=(births+deaths)/len(late) if late else float('nan')
                    changed=[number(e.get('examples_seen')) for e in events if number(e.get('births',0))+number(e.get('deaths',0))>0]
                    r['last_change_fraction']=max(changed)/max_examples if changed and max_examples>0 else (0.0 if max_examples>0 else float('nan'))
                else:
                    r['late_net_drift']=r['late_turnover']=r['last_change_fraction']=float('nan')
                runs.append(r); curves[(n0,policy,seed)]=curve
            except Exception as e:
                print(f'WARNING: skipped {folder}: {e}')
    if not runs:
        raise SystemExit(f'No summary.pt files found in {base}/value_*/neurons_*/seed_*/')
    save_csv(out/'per_run.csv',runs,['dataset','n0','policy','seed',*METRICS])
    grouped=defaultdict(list)
    for r in runs: grouped[(r['n0'],r['policy'])].append(r)
    agg=[]
    for (n0,policy),items in sorted(grouped.items()):
        row={'n0':n0,'policy':policy,'seeds':','.join(str(r['seed']) for r in items),'n':len(items)}
        for metric in METRICS:
            m,s,count=summarize([r[metric] for r in items])
            row[metric+'_mean']=m;row[metric+'_sd']=s;row[metric+'_n']=count
        agg.append(row)
    aggfields=['n0','policy','seeds','n']+[f'{metric}_{suffix}' for metric in METRICS for suffix in ('mean','sd','n')]
    save_csv(out/'aggregate.csv',agg,aggfields)
    paired=[]
    for n0 in sorted({r['n0'] for r in runs}):
        for other in ('random','reverse_flux'):
            ref={r['seed']:r for r in grouped.get((n0,'high_flux'),[])}
            cmp={r['seed']:r for r in grouped.get((n0,other),[])}
            shared=sorted(ref.keys() & cmp.keys())
            for metric in METRICS:
                diffs=[cmp[s][metric]-ref[s][metric] for s in shared]
                mean,sd,count=summarize(diffs)
                # Paired 95% CI requires scipy; use t distribution if available.
                ci_low=ci_high=float('nan')
                if count>1:
                    try:
                        from scipy.stats import t
                        half=t.ppf(.975,count-1)*sd/np.sqrt(count)
                        ci_low,ci_high=mean-half,mean+half
                    except ImportError: pass
                paired.append({'n0':n0,'comparison':f'{other} - high_flux','metric':metric,
                               'n_pairs':count,'seeds':','.join(map(str,shared)),
                               'mean_difference':mean,'sd_difference':sd,
                               'ci95_low':ci_low,'ci95_high':ci_high})
    save_csv(out/'paired_comparisons.csv',paired,list(paired[0].keys()) if paired else ['n0','comparison','metric','n_pairs','seeds','mean_difference','sd_difference','ci95_low','ci95_high'])
    trajectory=[]
    for (n0,policy,seed),c in curves.items():
        for idx,(neurons,acc) in enumerate(zip(c.get('num_neurons',[]),c.get('test_accuracy',[])),1):
            trajectory.append({'n0':n0,'policy':policy,'seed':seed,'epoch':idx,
                               'neurons':number(neurons),'test_accuracy':number(acc)})
    save_csv(out/'trajectories.csv',trajectory,['n0','policy','seed','epoch','neurons','test_accuracy'])
    lines=[f'FlowGenesis selection ablation: {args.dataset}',f'Late window: final {args.late_fraction:.0%} of training','',
           'IMPORTANT: similar neuron counts alone do not prove identical full-network attractors.',
           'Paired differences are (random/reverse_flux) minus high_flux, matched by seed.',
           'Confidence intervals are descriptive; multiple comparisons are not adjusted.','']
    for n0 in sorted({r['n0'] for r in runs}):
        lines.append(f'=== N0={n0} ===')
        for policy in POLICIES:
            group=grouped.get((n0,policy),[])
            if not group: lines.append(f'{policy}: missing');continue
            lines.append(f'{policy} | seeds={sorted(r["seed"] for r in group)}')
            for metric in ('final_accuracy','final_neurons','late_neurons_mean','late_neurons_sd','late_net_drift','late_turnover','final_total_born','final_total_pruned'):
                lines.append(f'  {metric}: {fmt([r[metric] for r in group])}')
        lines.append('')
    out.mkdir(parents=True,exist_ok=True)
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt
            for n0 in sorted({r['n0'] for r in runs}):
                for metric,filename,ylabel in [('neurons','neuron_trajectories','Neurons'),('test_accuracy','accuracy_trajectories','Test accuracy (%)')]:
                    fig,ax=plt.subplots(figsize=(8,5))
                    for policy in POLICIES:
                        relevant=[(seed,c) for (size,pol,seed),c in curves.items() if size==n0 and pol==policy and len(c.get('num_neurons',[]))]
                        if not relevant:continue
                        max_len=max(len(c.get('num_neurons',[])) for _,c in relevant)
                        arr=np.full((len(relevant),max_len),np.nan)
                        for j,(_,c) in enumerate(relevant):
                            vals=c.get('num_neurons',[]) if metric=='neurons' else c.get('test_accuracy',[])
                            arr[j,:len(vals)]=vals
                        count=np.sum(np.isfinite(arr),axis=0)
                        avg=np.divide(np.nansum(arr,axis=0),count,out=np.full(max_len,np.nan),where=count>0)
                        sd=np.array([np.std(arr[np.isfinite(arr[:,i]),i],ddof=1) if count[i]>1 else np.nan for i in range(max_len)])
                        x=np.arange(1,max_len+1)
                        line,=ax.plot(x,avg,label=f'{policy} (n={len(relevant)})')
                        ax.fill_between(x,avg-sd,avg+sd,alpha=.15,color=line.get_color())
                    ax.set(xlabel='Epoch',ylabel=ylabel,title=f'{args.dataset} N0={n0} — {ylabel}')
                    ax.legend();fig.tight_layout();fig.savefig(out/f'{filename}_n0_{n0}.png',dpi=180);plt.close(fig)
        except ImportError:
            print('matplotlib unavailable: CSV and report generated without plots')
    print(f'\nSaved results to: {out.resolve()}')

if __name__=='__main__':main()
