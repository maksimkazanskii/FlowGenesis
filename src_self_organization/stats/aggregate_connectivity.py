#!/usr/bin/env python3
"""
Generate a LaTeX table for the FlowGenesis connectivity ablation.

Expected structure:

    beta_ablations/
        value_3/
            neurons_50/seed_*/
            neurogenesis_summary.csv
        value_6/
        value_12/
        value_24/
        independent_metrics_runs.csv
        comparative_metrics_pairs.csv

Similarity metrics are cross-seed comparisons within each connectivity condition.

Usage:
    python src_self_organization/stats/aggregated_beta.py \
        data/neurogenesis/beta_ablations
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import f as f_dist


def mean_sd(values):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(x) == 0:
        return np.nan, np.nan
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def fmt(mean, sd, decimals=1, scale=1.0):
    if not np.isfinite(mean):
        return "--"
    return rf"${mean*scale:.{decimals}f} \pm {sd*scale:.{decimals}f}$"


def find_column(df, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if required:
        raise KeyError(
            f"Could not find any of {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def beta_from_condition(value):
    m = re.search(r"(?:^|/)value_([-+]?(?:\d+(?:\.\d*)?|\.\d+))(?:/|$)", str(value))
    return float(m.group(1)) if m else None


def discover_betas(root):
    values = []
    for path in root.glob("value_*"):
        if path.is_dir():
            m = re.fullmatch(r"value_([-+]?(?:\d+(?:\.\d*)?|\.\d+))", path.name)
            if m:
                values.append(float(m.group(1)))
    values = sorted(set(values))
    if not values:
        raise RuntimeError(f"No value_* connectivity directories found under {root}")
    return values


def parse_final_accuracy(run_dir):
    curves_path = run_dir / "learning_curves.pt"
    if curves_path.exists():
        curves = torch.load(curves_path, map_location="cpu", weights_only=False)
        for key in ("test_accuracy", "test_acc", "test_accuracy_history"):
            values = curves.get(key)
            if values is not None and len(values):
                value = values[-1]
                if torch.is_tensor(value):
                    value = value.detach().cpu().item()
                return float(value)

    final_path = run_dir / "final_model.pt"
    if final_path.exists():
        checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
        for key in ("final_test_acc", "final_acc", "test_acc"):
            if key in checkpoint:
                return float(checkpoint[key])
    return np.nan


def welch_anova(groups):
    clean = []
    for group in groups:
        x = pd.to_numeric(pd.Series(group), errors="coerce").dropna().to_numpy(float)
        if len(x) >= 2 and np.var(x, ddof=1) > 0:
            clean.append(x)

    k = len(clean)
    if k < 2:
        return np.nan, np.nan, np.nan, np.nan

    n = np.array([len(x) for x in clean], dtype=float)
    means = np.array([np.mean(x) for x in clean], dtype=float)
    var = np.array([np.var(x, ddof=1) for x in clean], dtype=float)
    w = n / var
    wsum = w.sum()
    ybar = np.sum(w * means) / wsum
    term = np.sum(((1.0 - w / wsum) ** 2) / (n - 1.0))

    F = (np.sum(w * (means - ybar) ** 2) / (k - 1.0)) / (
            1.0 + 2.0 * (k - 2.0) * term / (k**2 - 1.0)
    )
    df1 = k - 1.0
    df2 = (k**2 - 1.0) / (3.0 * term)
    return float(F), float(df1), float(df2), float(f_dist.sf(F, df1, df2))


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m - rank) * p[idx])
        running = max(running, adjusted)
        out[idx] = running
    return out


def fmt_p(p):
    if not np.isfinite(p):
        return "--"
    if p < 0.001:
        return r"$<0.001$"
    return f"${p:.3f}$"


def load_training_summaries(root, betas):
    frames = []
    for beta in betas:
        condition_dir = root / f"value_{beta:g}"
        summary_path = condition_dir / "neurogenesis_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Missing {summary_path}. Run stats.py for every connectivity first."
            )
        frame = pd.read_csv(summary_path)
        frame["beta"] = beta
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="Connectivity ablation experiment directory")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output .tex file (default: ROOT/beta_ablation_table.tex)",
    )
    parser.add_argument("--n0", type=int, default=20)
    args = parser.parse_args()

    root = args.folder.resolve()
    output = args.output.resolve() if args.output else root / "connectivity_ablation_table.tex"

    betas = discover_betas(root)
    print("Connectivity values:", ", ".join(map(str, betas)))

    train = load_training_summaries(root, betas)
    train = train[pd.to_numeric(train["initial_n"], errors="coerce") == args.n0].copy()
    if train.empty:
        raise RuntimeError(f"No training rows for N0={args.n0}")

    independent_path = root / "independent_metrics_runs.csv"
    if not independent_path.exists():
        raise FileNotFoundError(
            f"Missing {independent_path}. Run independent_statistics.py first."
        )
    structural = pd.read_csv(independent_path)
    condition_col = find_column(structural, ["condition"])
    structural["beta"] = structural[condition_col].apply(beta_from_condition)
    structural = structural[structural["beta"].notna()].copy()
    structural = structural[pd.to_numeric(structural["initial_neurons"], errors="coerce") == args.n0].copy()
    structural["beta"] = structural["beta"].astype(float)

    comparative_path = root / "comparative_metrics_pairs.csv"
    if not comparative_path.exists():
        raise FileNotFoundError(
            f"Missing {comparative_path}. Run comparative_statistics.py first."
        )
    comp = pd.read_csv(comparative_path)
    comp_condition_col = find_column(comp, ["condition"])
    comp["beta"] = comp[comp_condition_col].apply(beta_from_condition)
    comp = comp[comp["beta"].notna()].copy()
    comp = comp[(pd.to_numeric(comp["n0_a"], errors="coerce") == args.n0) & (pd.to_numeric(comp["n0_b"], errors="coerce") == args.n0)].copy()
    comp["beta"] = comp["beta"].astype(float)

    train_seed = find_column(train, ["seed"])
    best_n_col = find_column(train, ["best_n"])
    final_n_col = find_column(train, ["final_n", "final_neurons"])
    best_acc_col = find_column(train, ["best_acc", "best_accuracy"])
    births_col = find_column(train, ["total_born", "births"])
    pruned_col = find_column(train, ["total_pruned", "pruned", "deaths"])

    train["final_acc"] = np.nan
    for idx, row in train.iterrows():
        beta = float(row["beta"])
        seed = int(row[train_seed])
        condition_dir = root / f"value_{beta:g}"

        # Do not rely on initial_n from neurogenesis_summary.csv because
        # older summaries may contain NaN there. Locate the actual run
        # directory from the experiment hierarchy instead.
        candidates = sorted(condition_dir.glob(f"neurons_{args.n0}/seed_{seed}"))

        if len(candidates) == 0:
            print(
                f"WARNING: could not find run directory for "
                f"beta={beta}, seed={seed}"
            )
            run_dir = condition_dir / "neurons_50" / f"seed_{seed}"
        elif len(candidates) > 1:
            raise RuntimeError(
                f"Multiple neurons_* directories found for "
                f"beta={beta}, seed={seed}: {candidates}"
            )
        else:
            run_dir = candidates[0]

        train.at[idx, "final_acc"] = parse_final_accuracy(run_dir)

    struct_seed = find_column(structural, ["seed"])
    final_struct_n_col = find_column(structural, ["final_neurons", "final_n"])
    edges_col = find_column(structural, ["edge_count", "final_edges", "edges"])
    density_col = find_column(structural, ["connectivity_density", "density"])
    rho_col = find_column(structural, ["spectral_radius", "rho"])
    structural = structural.drop_duplicates(
        subset=["beta", struct_seed], keep="last"
    )

    cka_col = find_column(comp, ["cka"])
    agreement_col = find_column(comp, ["prediction_agreement", "agreement"])
    hungarian_col = find_column(
        comp,
        ["hungarian_weight_similarity", "hungarian_similarity", "weight_similarity"],
    )

    pair_type_col = find_column(comp, ["pair_type"], required=False)
    if pair_type_col is not None:
        within_mask = comp[pair_type_col].astype(str).str.lower().eq("within_n0")
        if within_mask.any():
            comp = comp[within_mask].copy()

    rows = []
    for beta in betas:
        t = train[train["beta"] == beta].copy()
        s = structural[structural["beta"] == beta].copy()
        c = comp[comp["beta"] == beta].copy()

        best_n = mean_sd(t[best_n_col])
        final_n = mean_sd(t[final_n_col])
        best_acc = mean_sd(t[best_acc_col])
        final_acc = mean_sd(t["final_acc"])
        births = mean_sd(t[births_col])
        pruned = mean_sd(t[pruned_col])
        edges = mean_sd(s[edges_col])
        density = mean_sd(s[density_col])
        rho = mean_sd(s[rho_col])
        hungarian = mean_sd(c[hungarian_col])
        cka = mean_sd(c[cka_col])
        agreement = mean_sd(c[agreement_col])

        best_acc_scale = 100.0 if np.isfinite(best_acc[0]) and best_acc[0] <= 1 else 1.0
        final_acc_scale = 100.0 if np.isfinite(final_acc[0]) and final_acc[0] <= 1 else 1.0
        agreement_scale = 100.0 if np.isfinite(agreement[0]) and agreement[0] <= 1 else 1.0

        rows.append([
            str(beta),
            fmt(*best_n, decimals=1),
            fmt(*final_n, decimals=1),
            fmt(*best_acc, decimals=2, scale=best_acc_scale),
            fmt(*final_acc, decimals=2, scale=final_acc_scale),
            fmt(*births, decimals=1),
            fmt(*pruned, decimals=1),
            fmt(*edges, decimals=1),
            fmt(*density, decimals=3),
            fmt(*rho, decimals=3),
            fmt(*hungarian, decimals=3),
            fmt(*cka, decimals=3),
            fmt(*agreement, decimals=1, scale=agreement_scale),
        ])

    test_specs = [
        (r"$N_f$", structural, final_struct_n_col),
        (r"$E_f$", structural, edges_col),
        ("density", structural, density_col),
        (r"$\rho$", structural, rho_col),
    ]

    structural_tests = []
    for label, frame, value_col in test_specs:
        groups = [
            frame[frame["beta"] == beta][value_col]
            for beta in betas
        ]
        F, df1, df2, p = welch_anova(groups)
        structural_tests.append(
            {"label": label, "F": F, "df1": df1, "df2": df2, "p": p}
        )

    adjusted = holm_adjust([x["p"] for x in structural_tests])
    for result, p_adj in zip(structural_tests, adjusted):
        result["p_holm"] = p_adj

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{",
        r"Effect of recurrent connectivity probability $\beta$ on the structural and functional",
        r"properties of FlowGenesis on FashionMNIST.",
        r"$N_{\mathrm{best}}$ is the number of neurons at the epoch of maximum test",
        r"accuracy, and $N_f$ and $E_f$ are the final numbers of neurons and connections, respectively.",
        r"Best and final accuracy denote the maximum and last-epoch test accuracy.",
        r"Births and Pruned are the cumulative numbers of neurons added and removed during training.",
        r"Density is the final connection density, and $\rho$ is the spectral radius",
        r"of the final recurrent weight matrix.",
        r"Weight sim.\ is the Hungarian-aligned cosine similarity between neuron weight profiles,",
        r"CKA measures similarity between learned representations, and Agreement is the fraction",
        r"of examples receiving identical predicted labels across independently trained networks.",
        r"Similarity metrics report cross-seed comparisons within each connectivity condition.",
        r"Training and structural metrics are mean $\pm$ SD across seeds; similarity metrics summarize overlapping seed pairs.",
        r"}",
        r"\label{tab:connectivity_ablation}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{c cc cc cc ccc ccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Network size}",
        r"& \multicolumn{2}{c}{Performance}",
        r"& \multicolumn{2}{c}{Structural dynamics}",
        r"& \multicolumn{3}{c}{Final structure}",
        r"& \multicolumn{3}{c}{Cross-seed similarity} \\",
        r"\cmidrule(lr){2-3}",
        r"\cmidrule(lr){4-5}",
        r"\cmidrule(lr){6-7}",
        r"\cmidrule(lr){8-10}",
        r"\cmidrule(lr){11-13}",
        r"$\beta$",
        r"& $N_{\mathrm{best}}$",
        r"& $N_f$",
        r"& Best acc. (\%)",
        r"& Final acc. (\%)",
        r"& Births",
        r"& Pruned",
        r"& $E_f$",
        r"& Density",
        r"& $\rho$",
        r"& Weight sim.",
        r"& CKA",
        r"& Agreement (\%) \\",
        r"\midrule",
    ]

    for row in rows:
        lines.append(" & ".join(row) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\linewidth}",
        r"\footnotesize",
        (
                r"\textit{Dependence of final structure on $\beta$:} "
                r"separate Welch one-way ANOVAs across $\beta$ for $N_f$, $E_f$, density, and $\rho$, "
                r"with Holm correction across the four tests. Adjusted $p$-values: "
                + ", ".join(
            f"{result['label']} {fmt_p(result['p_holm'])}"
            for result in structural_tests
        )
                + r"."
        ),
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved LaTeX table: {output}")

    missing_final = int(
        pd.to_numeric(train["final_acc"], errors="coerce").isna().sum()
    )
    if missing_final:
        print(
            f"WARNING: final accuracy could not be parsed for {missing_final} run(s)."
        )

    print("\nRuns per connectivity:")
    for beta in betas:
        n_train = int((train["beta"] == beta).sum())
        n_struct = int((structural["beta"] == beta).sum())
        n_pairs = int((comp["beta"] == beta).sum())
        print(
            f"  beta={beta:>3}: training={n_train}, "
            f"structural={n_struct}, similarity pairs={n_pairs}"
        )

    print("\nStructural connectivity tests:")
    for result in structural_tests:
        print(
            f"  {result['label']:>8s}: F={result['F']:.4g}, "
            f"raw p={result['p']:.4g}, Holm p={result['p_holm']:.4g}"
        )


if __name__ == "__main__":
    main()

