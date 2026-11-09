#!/usr/bin/env python3
"""Aggregate existing FlowGenesis regime statistics into a 12-column LaTeX table.

Usage:
 python src_self_organization/stats/aggregate_regime.py \
   data/neurogenesis/regime_n_sweep/fashionmnist/regime

Requires stats.py run for each value_*, independent_statistics.py on the
regime root, and comparative_statistics.py separately on each value_*.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REGIMES = [('full','Full (birth and death)'), ('no_birth','No birth'),
           ('no_death','No death'), ('static','Static')]
N0 = [20, 40, 80, 160]

def column(df, names, required=True):
    lookup = {str(c).lower():c for c in df.columns}
    for n in names:
        if n.lower() in lookup:
            return lookup[n.lower()]
    if required:
        raise KeyError(f'Expected one of {names}; found {list(df.columns)}')
    return None

def number(row, key):
    if key is None:
        return np.nan
    return pd.to_numeric(row[key], errors='coerce')

def final_accuracy(run):
    p = run/'learning_curves.pt'
    if p.exists():
        d = torch.load(p, map_location='cpu', weights_only=False)
        for key in ('test_accuracy','test_acc','test_accuracy_history'):
            v = d.get(key)
            if v is not None and len(v):
                return float(v[-1])
    p = run/'final_model.pt'
    if p.exists():
        d = torch.load(p, map_location='cpu', weights_only=False)
        for key in ('final_test_acc','final_acc','test_acc'):
            if key in d:
                return float(d[key])
    return np.nan

def fmt(x, decimals=1):
    v = pd.to_numeric(pd.Series(x), errors='coerce').dropna().to_numpy(float)
    if not len(v):
        return '--'
    mean = v.mean()
    sd = v.std(ddof=1) if len(v)>1 else 0.0
    return rf'${mean:.{decimals}f}\pm{sd:.{decimals}f}$'

def load(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Run the corresponding existing stats script first.')
    return pd.read_csv(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--allow-incomplete', action='store_true')
    args=p.parse_args()
    root=args.root.resolve()
    structural=load(root/'independent_metrics_runs.csv')
    s_condition=column(structural,['condition'])
    s_n0=column(structural,['initial_neurons','initial_n','n0'])
    s_seed=column(structural,['seed'])
    s_edges=column(structural,['edge_count','final_edges','edges'])
    s_density=column(structural,['connectivity_density','density'])
    s_rho=column(structural,['spectral_radius','rho'])
    results=[]
    for regime, label in REGIMES:
        base=root/f'value_{regime}'
        train=load(base/'neurogenesis_summary.csv')
        comp=load(base/'comparative_metrics_model_level.csv')
        t_n0=column(train,['initial_n','initial_neurons','n0'])
        t_seed=column(train,['seed'])
        t_best_n=column(train,['best_n'])
        t_final_n=column(train,['final_n','final_neurons'])
        t_best_acc=column(train,['best_acc','best_accuracy'])
        t_final_acc=column(train,['final_acc','final_accuracy','final_test_acc'],False)
        t_births=column(train,['total_born','births'])
        t_pruned=column(train,['total_pruned','pruned','deaths'])
        c_n0=column(comp,['source_n0','initial_neurons','initial_n','n0'])
        c_seed=column(comp,['source_seed','seed'])
        c_cka=column(comp,['cross_n0_cka'])
        c_agree=column(comp,['cross_n0_prediction_agreement','cross_n0_agreement'])
        for n0 in N0:
            t=train[train[t_n0]==n0]
            s=structural[(structural[s_n0]==n0) & (structural[s_condition].astype(str).str.split('/').str[-1]==f'value_{regime}')]
            c=comp[comp[c_n0]==n0]
            t=t.drop_duplicates(subset=t_seed)
            s=s.drop_duplicates(subset=s_seed)
            c=c.drop_duplicates(subset=c_seed)
            expected=set(range(5))
            counts={'train':set(t[t_seed].astype(int)), 'structure':set(s[s_seed].astype(int)), 'similarity':set(c[c_seed].astype(int))}
            for kind, seeds in counts.items():
                if seeds != expected:
                    msg=f'{regime}, N0={n0}: {kind} seeds={sorted(seeds)}, expected={sorted(expected)}'
                    if not args.allow_incomplete:
                        raise RuntimeError(msg+'; use --allow-incomplete only if intentional')
                    print('WARNING:',msg)
            acc=[]
            for _,r in t.iterrows():
                value=number(r,t_final_acc)
                if not np.isfinite(value):
                    value=final_accuracy(base/f'neurons_{n0}'/f'seed_{int(r[t_seed])}')
                acc.append(value)
            if not args.allow_incomplete and not np.isfinite(acc).all():
                raise RuntimeError(f'Missing final accuracies for {regime} N0={n0}')
            row={'regime':regime,'N0':n0,'n_seeds':len(t),
                 'best_n':fmt(t[t_best_n]),'final_n':fmt(t[t_final_n]),
                 'best_acc':fmt(t[t_best_acc],2),'final_acc':fmt(acc,2),
                 'births':fmt(t[t_births]),'pruned':fmt(t[t_pruned]),
                 'edges':fmt(s[s_edges]),'density':fmt(s[s_density],3),
                 'rho':fmt(s[s_rho],3),'cka':fmt(c[c_cka],3),
                 'agreement':fmt(c[c_agree]*100,2)}
            results.append(row)
            print(f'{regime:9} N0={n0:3}: Nf={row["final_n"]}, CKA={row["cka"]}')
    df=pd.DataFrame(results)
    out=args.output or root/'regime_ablation_table.tex'
    lines=[r'\begin{table*}[t]',r'\centering',
           r'\caption{Structural-regime ablation on Fashion-MNIST (five seeds per setting). Values are mean $\pm$ standard deviation. Cross-$N_0$ similarities compare each trained model against models initialized with different widths within the same regime.}',
           r'\label{tab:regime_ablation}',r'\scriptsize',r'\setlength{\tabcolsep}{2pt}',r'\renewcommand{\arraystretch}{1.12}',r'\resizebox{\textwidth}{!}{%',
           r'\begin{tabular}{@{}l cc cc cc ccc cc@{}}',r'\toprule',
           r'& \multicolumn{2}{c}{Network size} & \multicolumn{2}{c}{Performance} & \multicolumn{2}{c}{Structural dynamics} & \multicolumn{3}{c}{Final structure} & \multicolumn{2}{c}{Cross-$N_0$ similarity} \\',
           r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}\cmidrule(lr){11-12}',
           r'$N_0$ & $N_{\mathrm{best}}$ & $N_f$ & Best acc. & Final acc. & Births & Pruned & $E_f$ & Density & $\rho$ & CKA & Agr. \\',r'\midrule']
    for regime,label in REGIMES:
        lines += [rf'\multicolumn{{12}}{{c}}{{\textbf{{{label}}}}} \\',r'\midrule']
        for r in results:
            if r['regime']!=regime:continue
            fields=[str(r['N0'])]+[r[k] for k in ['best_n','final_n','best_acc','final_acc','births','pruned','edges','density','rho','cka','agreement']]
            lines.append(' & '.join(fields)+r' \\')
        lines.append(r'\midrule' if regime!='static' else r'\bottomrule')
    lines += [r'\end{tabular}%',r'}',r'\vspace{3pt}',r'\begin{minipage}{\textwidth}',r'\footnotesize',
              r'\textit{Notes:} $N_{\mathrm{best}}$ and $N_f$ are neuron counts at best accuracy and at training end. $E_f$, density, and $\rho$ are final recurrent edges, connectivity density, and spectral radius. Accuracies and agreement are percentages. CKA and agreement are cross-$N_0$ model-level averages, with standard deviations computed across the five source models at each $N_0$; pairwise comparisons are not treated as independent replicates.',
              r'\end{minipage}',r'\end{table*}']
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text('\n'.join(lines)+'\n')
    df.to_csv(out.with_name('regime_ablation_aggregated.csv'),index=False)
    print('Saved:',out)
    print('Saved:',out.with_name('regime_ablation_aggregated.csv'))

if __name__=='__main__':main()
