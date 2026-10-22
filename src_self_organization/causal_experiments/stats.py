#!/usr/bin/env python3
"""Aggregate FlowGenesis causal parent-selection experiments from summary.pt files.

Example:
 python stats_causal.py --root data/neurogenesis/ablations --dataset fashionmnist --n0 20 --seeds 1 2 3 4 5
"""
import argparse
import csv
from pathlib import Path
import statistics as st
import torch

METRICS = [
    ("best_accuracy", "Best accuracy (%)"),
    ("final_accuracy", "Final accuracy (%)"),
    ("best_neurons", "Neurons at best"),
    ("final_neurons", "Final evaluated neurons"),
    ("final_total_born", "Final births"),
    ("final_total_pruned", "Final prunings"),
    ("best_epoch", "Best epoch"),
]
DEFAULT_SELECTIONS = ["high_flux", "reverse_flux", "random_all", "random_all_matched"]

def mean_sd(xs):
    if not xs:
        return "n/a"
    return f"{st.mean(xs):.3f} ± {st.stdev(xs):.3f}" if len(xs)>1 else f"{xs[0]:.3f} (n=1)"

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('data/neurogenesis/ablations'))
    p.add_argument('--dataset', default='fashionmnist')
    p.add_argument('--n0', type=int, default=20)
    p.add_argument('--seeds', nargs='+', type=int, default=[1,2,3,4,5])
    p.add_argument('--selections', nargs='+', default=DEFAULT_SELECTIONS)
    p.add_argument('--out', type=Path, default=None, help='CSV output (default: root/causal_stats_<dataset>_n0_<n0>.csv)')
    args=p.parse_args()

    rows=[]
    for selection in args.selections:
        for seed in args.seeds:
            path=(args.root/args.dataset/'selection'/f'value_{selection}'/
                  f'neurons_{args.n0}'/f'seed_{seed}'/'summary.pt')
            if not path.is_file():
                print(f'MISSING {path}')
                continue
            data=torch.load(path, map_location='cpu', weights_only=False)
            if (data.get('seed')!=seed or data.get('initial_neurons')!=args.n0 or
                    data.get('ablation')!='selection' or data.get('ablation_value')!=selection):
                raise ValueError(f'Metadata mismatch in {path}')
            row={'selection':selection, 'seed':seed, 'path':str(path)}
            for key,_ in METRICS:
                row[key]=float(data[key]) if data.get(key) is not None else None
            events=data.get('structure_events',[])
            row['checks']=len(events)
            row['event_births']=sum(int(e['births']) for e in events)
            row['event_deaths']=sum(int(e['deaths']) for e in events)
            row['changed_checks']=sum(bool(e['births'] or e['deaths']) for e in events)
            rows.append(row)

    if not rows:
        raise SystemExit('No summary.pt files found. Check --root and directory structure.')

    out=args.out or args.root/f'causal_stats_{args.dataset}_n0_{args.n0}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved per-seed CSV: {out}\n')

    for selection in args.selections:
        group=[r for r in rows if r['selection']==selection]
        print(f'=== {selection} (n={len(group)}; seeds={[r["seed"] for r in group]}) ===')
        for key,label in METRICS:
            values=[r[key] for r in group if r[key] is not None]
            print(f'{label:27s}: {mean_sd(values)}')
        for key,label in [('event_births','Event births'),('event_deaths','Event deaths'),('changed_checks','Changed checks')]:
            print(f'{label:27s}: {mean_sd([r[key] for r in group])}')
        print()

    # Paired comparisons: use only seeds present in both variants.
    baseline={r['seed']:r for r in rows if r['selection']=='high_flux'}
    if baseline:
        print('=== Paired differences vs high_flux (alternative − high_flux) ===')
        for selection in args.selections:
            if selection=='high_flux':
                continue
            alternatives={r['seed']:r for r in rows if r['selection']==selection}
            common=sorted(set(baseline)&set(alternatives))
            if not common:
                print(f'{selection}: no paired seeds')
                continue
            print(f'{selection} (paired seeds={common}):')
            for key,label in [('final_accuracy','Final accuracy (pp)'),('best_accuracy','Best accuracy (pp)'),('final_neurons','Final neurons'),('event_births','Total event births')]:
                diffs=[alternatives[s][key]-baseline[s][key] for s in common
                       if alternatives[s][key] is not None and baseline[s][key] is not None]
                print(f'  {label:25s}: {mean_sd(diffs)}')
            if selection=='random_all_matched':
                mismatches=[]
                for seed in common:
                    h=torch.load(baseline[seed]['path'],map_location='cpu',weights_only=False)['structure_events']
                    a=torch.load(alternatives[seed]['path'],map_location='cpu',weights_only=False)['structure_events']
                    if len(h)!=len(a) or any(
                            x['examples_seen']!=y['examples_seen'] or x['births']!=y['births']
                            for x,y in zip(h,a)
                    ):
                        mismatches.append(seed)
                print('  Matched birth schedule:', 'PASS' if not mismatches else f'FAIL for seeds {mismatches}')
            print()

if __name__=='__main__':
    main()
