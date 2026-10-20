#!/usr/bin/env python3
"""
Generate a LaTeX table for the FlowGenesis death-flux-fraction ablation.

Expected structure:

    death_fraction_ablations/fashionmnist/death_fraction/
        value_0.05/neurons_20/seed_*/
        value_0.1/neurons_20/seed_*/
        value_0.2/neurons_20/seed_*/
        value_0.4/neurons_20/seed_*/
        independent_metrics_runs.csv

Before running this script:
    1) Run stats.py separately for every value_* directory.
    2) Run independent_statistics.py once at the death_fraction root.

Usage:
    python -m src_self_organization.stats.aggregate_death \
        data/neurogenesis/death_fraction_ablations/fashionmnist/death_fraction

The script intentionally does NOT use comparative/similarity statistics.
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
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return float(np.mean(x)), sd


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


def death_from_condition(value):
    m = re.search(
        r"(?:^|/)value_([-+]?(?:\d+(?:\.\d*)?|\.\d+))(?:/|$)",
        str(value),
    )
    return float(m.group(1)) if m else None


def discover_death_values(root):
    values = []
    for path in root.glob("value_*"):
        if not path.is_dir():
            continue
        try:
            values.append(float(path.name[len("value_"):]))
        except ValueError:
            pass

    values = sorted(set(values))
    if not values:
        raise RuntimeError(f"No numeric value_* death-fraction directories found under {root}")
    return values


def matching_value_dir(root, value):
    candidates = []
    for path in root.glob("value_*"):
        if not path.is_dir():
            continue
        try:
            if np.isclose(float(path.name[len("value_"):]), value):
                candidates.append(path)
        except ValueError:
            pass

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one directory for death_fraction={value:g}, found: {candidates}"
        )
    return candidates[0]


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


def load_training_summaries(root, death_values):
    frames = []
    for death_fraction in death_values:
        condition_dir = matching_value_dir(root, death_fraction)
        summary_path = condition_dir / "neurogenesis_summary.csv"

        if not summary_path.exists():
            raise FileNotFoundError(
                f"Missing {summary_path}. Run stats.py for every death-fraction value first."
            )

        frame = pd.read_csv(summary_path)
        frame["death_fraction"] = death_fraction
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folder", type=Path, help="Death-flux-fraction ablation experiment directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .tex file (default: ROOT/death_fraction_ablation_table.tex)",
    )
    args = parser.parse_args()

    root = args.folder.resolve()
    output = (
        args.output.resolve()
        if args.output
        else root / "death_fraction_ablation_table.tex"
    )

    death_values = discover_death_values(root)
    print("Death-flux fractions:", ", ".join(f"{x:g}" for x in death_values))

    train = load_training_summaries(root, death_values)

    independent_path = root / "independent_metrics_runs.csv"
    if not independent_path.exists():
        raise FileNotFoundError(
            f"Missing {independent_path}. Run independent_statistics.py first."
        )

    structural = pd.read_csv(independent_path)
    condition_col = find_column(structural, ["condition"])
    structural["death_fraction"] = structural[condition_col].apply(death_from_condition)
    structural = structural[structural["death_fraction"].notna()].copy()

    train_seed = find_column(train, ["seed"])
    best_n_col = find_column(train, ["best_n"])
    final_n_col = find_column(train, ["final_n", "final_neurons"])
    best_acc_col = find_column(train, ["best_acc", "best_accuracy"])
    births_col = find_column(train, ["total_born", "births"])
    pruned_col = find_column(train, ["total_pruned", "pruned", "deaths"])

    # Recover last-epoch accuracy for every seed.
    train["final_acc"] = np.nan
    for idx, row in train.iterrows():
        death_fraction = float(row["death_fraction"])
        seed = int(row[train_seed])
        condition_dir = matching_value_dir(root, death_fraction)

        candidates = sorted(condition_dir.glob(f"neurons_*/seed_{seed}"))
        if len(candidates) == 0:
            print(
                f"WARNING: could not find run directory for "
                f"death_fraction={death_fraction:g}, seed={seed}"
            )
            continue
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple neurons_* directories found for "
                f"death_fraction={death_fraction:g}, seed={seed}: {candidates}"
            )

        train.at[idx, "final_acc"] = parse_final_accuracy(candidates[0])

    struct_seed = find_column(structural, ["seed"])
    edges_col = find_column(structural, ["edge_count", "final_edges", "edges"])
    density_col = find_column(structural, ["connectivity_density", "density"])
    rho_col = find_column(structural, ["spectral_radius", "rho"])

    structural = structural.drop_duplicates(
        subset=["death_fraction", struct_seed], keep="last"
    )

    # ------------------------------------------------------------
    # Aggregate table rows
    # ------------------------------------------------------------
    rows = []
    for death_fraction in death_values:
        t = train[np.isclose(train["death_fraction"], death_fraction)].copy()
        s = structural[np.isclose(structural["death_fraction"], death_fraction)].copy()

        best_n = mean_sd(t[best_n_col])
        final_n = mean_sd(t[final_n_col])
        best_acc = mean_sd(t[best_acc_col])
        final_acc = mean_sd(t["final_acc"])
        births = mean_sd(t[births_col])
        pruned = mean_sd(t[pruned_col])
        edges = mean_sd(s[edges_col])
        density = mean_sd(s[density_col])
        rho = mean_sd(s[rho_col])

        best_acc_scale = (
            100.0 if np.isfinite(best_acc[0]) and best_acc[0] <= 1 else 1.0
        )
        final_acc_scale = (
            100.0 if np.isfinite(final_acc[0]) and final_acc[0] <= 1 else 1.0
        )

        label = f"{death_fraction:g}"
        if np.isclose(death_fraction, 0.10):
            label += " (default)"

        rows.append([
            label,
            fmt(*best_n, decimals=1),
            fmt(*final_n, decimals=1),
            fmt(*best_acc, decimals=2, scale=best_acc_scale),
            fmt(*final_acc, decimals=2, scale=final_acc_scale),
            fmt(*births, decimals=1),
            fmt(*pruned, decimals=1),
            fmt(*edges, decimals=1),
            fmt(*density, decimals=3),
            fmt(*rho, decimals=3),
        ])

    # ------------------------------------------------------------
    # Inferential statistics: performance only
    # ------------------------------------------------------------
    performance_specs = [
        ("Best accuracy", best_acc_col),
        ("Final accuracy", "final_acc"),
    ]

    performance_tests = []
    for label, value_col in performance_specs:
        groups = [
            train[np.isclose(train["death_fraction"], death_fraction)][value_col]
            for death_fraction in death_values
        ]
        F, df1, df2, p = welch_anova(groups)
        performance_tests.append(
            {"label": label, "F": F, "df1": df1, "df2": df2, "p": p}
        )

    adjusted = holm_adjust([x["p"] for x in performance_tests])
    for result, p_adj in zip(performance_tests, adjusted):
        result["p_holm"] = p_adj

    # ------------------------------------------------------------
    # LaTeX
    # ------------------------------------------------------------
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{",
        r"Effect of the neurogenesis death-flux fraction $f_{\mathrm{death}}$ on the structural",
        r"and functional properties of FlowGenesis on FashionMNIST.",
        r"$N_{\mathrm{best}}$ is the number of neurons at the epoch of maximum test accuracy,",
        r"and $N_f$ and $E_f$ are the final numbers of neurons and connections, respectively.",
        r"Best and final accuracy denote the maximum and last-epoch test accuracy.",
        r"Births and Pruned are the cumulative numbers of neurons added and removed during training.",
        r"Density is the final connection density, and $\rho$ is the spectral radius",
        r"of the final effective recurrent weight matrix.",
        r"Values are mean $\pm$ standard deviation across five seeds.",
        r"}",
        r"\label{tab:death_fraction_ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{c cc cc cc ccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Network size}",
        r"& \multicolumn{2}{c}{Performance}",
        r"& \multicolumn{2}{c}{Structural dynamics}",
        r"& \multicolumn{3}{c}{Final structure} \\",
        r"\cmidrule(lr){2-3}",
        r"\cmidrule(lr){4-5}",
        r"\cmidrule(lr){6-7}",
        r"\cmidrule(lr){8-10}",
        r"$f_{\mathrm{death}}$",
        r"& $N_{\mathrm{best}}$",
        r"& $N_f$",
        r"& Best acc. (\%)",
        r"& Final acc. (\%)",
        r"& Births",
        r"& Pruned",
        r"& $E_f$",
        r"& Density",
        r"& $\rho$ \\",
        r"\midrule",
    ]

    for row in rows:
        lines.append(" & ".join(row) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\textwidth}",
        r"\footnotesize",
        (
                r"\textit{Effect of death-flux fraction on performance:} "
                r"separate Welch one-way ANOVAs across death-flux-fraction conditions "
                r"for best and final test accuracy, with Holm correction across the two tests. "
                r"Adjusted $p$-values: "
                + ", ".join(
            f"{result['label']} {fmt_p(result['p_holm'])}"
            for result in performance_tests
        )
                + r"."
        ),
        r"\end{minipage}",
        r"\end{table*}",
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
            f"WARNING: final accuracy could not be parsed for "
            f"{missing_final} run(s)."
        )

    print("\nRuns per death-flux fraction:")
    for death_fraction in death_values:
        n_train = int(np.isclose(train["death_fraction"], death_fraction).sum())
        n_struct = int(np.isclose(structural["death_fraction"], death_fraction).sum())
        print(
            f"  death_fraction={death_fraction:g}: "
            f"training={n_train}, structural={n_struct}"
        )

    print("\nPerformance tests:")
    for result in performance_tests:
        print(
            f"  {result['label']:>14s}: "
            f"F={result['F']:.4g}, "
            f"df=({result['df1']:.3g}, {result['df2']:.3g}), "
            f"raw p={result['p']:.4g}, "
            f"Holm p={result['p_holm']:.4g}"
        )


if __name__ == "__main__":
    main()
