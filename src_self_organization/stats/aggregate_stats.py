#!/usr/bin/env python3
"""
Generate a LaTeX table for the FlowGenesis N0 ablation.

Reads from the experiment directory:
    neurogenesis_summary.csv
    independent_metrics_runs.csv
    comparative_metrics_model_level.csv

Writes only:
    n0_ablation_table.tex

Usage:
    python src_self_organization/stats/aggregate_stats.py \
        data/neurogenesis/fashionmnist
"""

import argparse
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



def welch_anova(groups):
    """Welch one-way ANOVA; returns F, df1, df2, p."""
    clean = []
    for g in groups:
        x = pd.to_numeric(pd.Series(g), errors="coerce").dropna().to_numpy(float)
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


def parse_final_accuracy(root, initial_n, seed):
    """Parse last test accuracy from learning_curves.pt, with checkpoint fallback."""
    run_dir = root / f"neurons_{int(initial_n)}" / f"seed_{int(seed)}"
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
        ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
        for key in ("final_test_acc", "final_acc", "test_acc"):
            if key in ckpt:
                return float(ckpt[key])
    return np.nan

def fmt(mean, sd, decimals=1, scale=1.0):
    if not np.isfinite(mean):
        return "--"
    mean *= scale
    sd *= scale
    return f"${mean:.{decimals}f} \\pm {sd:.{decimals}f}$"


def find_column(df, candidates, required=True):
    """Return the first matching column, allowing a few naming variants."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="N0 ablation experiment directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .tex file (default: ROOT/n0_ablation_table.tex)",
    )
    args = parser.parse_args()

    root = args.folder.resolve()
    output = args.output.resolve() if args.output else root / "n0_ablation_table.tex"

    summary_path = root / "neurogenesis_summary.csv"
    independent_path = root / "independent_metrics_runs.csv"
    comparative_path = root / "comparative_metrics_model_level.csv"

    for path in (summary_path, independent_path, comparative_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Run the corresponding statistics script first."
            )

    train = pd.read_csv(summary_path)
    structural = pd.read_csv(independent_path)
    comp = pd.read_csv(comparative_path)

    # Populate final accuracy directly from each run if stats.py did not save it.
    final_candidates = [
        "final_acc", "final_accuracy", "last_acc", "last_test_acc",
        "final_test_acc", "test_acc_final",
    ]
    final_existing = next((c for c in final_candidates if c in train.columns), None)
    if final_existing is None:
        train["final_acc"] = np.nan
        final_existing = "final_acc"

    for idx, row in train.iterrows():
        if pd.isna(row[final_existing]):
            train.at[idx, final_existing] = parse_final_accuracy(
                root, row["initial_n"], row["seed"]
            )

    # ------------------------------------------------------------------
    # Training-summary columns
    # ------------------------------------------------------------------
    train_n0 = find_column(train, ["initial_n", "initial_neurons", "n0"])
    train_seed = find_column(train, ["seed"])
    best_n_col = find_column(train, ["best_n"])
    final_n_col = find_column(train, ["final_n", "final_neurons"])
    best_acc_col = find_column(train, ["best_acc", "best_accuracy"])
    births_col = find_column(train, ["total_born", "births"])
    pruned_col = find_column(train, ["total_pruned", "pruned", "deaths"])

    # Final accuracy is not present in the stats.py output shown previously in
    # the terminal. Support it automatically if stats.py has since added it.
    final_acc_col = find_column(
        train,
        ["final_acc", "final_accuracy", "last_acc", "last_test_acc",
         "final_test_acc", "test_acc_final"],
        required=False,
    )

    # ------------------------------------------------------------------
    # Independent structural metrics
    # ------------------------------------------------------------------
    struct_n0 = find_column(structural, ["initial_neurons", "initial_n", "n0"])
    struct_seed = find_column(structural, ["seed"])
    density_col = find_column(structural, ["connectivity_density", "density"])
    rho_col = find_column(structural, ["spectral_radius", "rho"])
    edges_col = find_column(structural, ["edge_count", "final_edges", "edges"])

    # Remove accidental duplicate (N0, seed) entries, e.g. a stray root model.
    structural = structural.drop_duplicates(
        subset=[struct_n0, struct_seed], keep="last"
    )

    # ------------------------------------------------------------------
    # Comparative functional metrics
    # ------------------------------------------------------------------
    comp_n0 = find_column(
        comp,
        ["source_n0", "initial_neurons", "initial_n", "n0"]
    )

    # Each model contributes one within-N0 and one cross-N0 score.
    # This avoids treating overlapping model pairs as independent replicates.
    similarity_columns = {}
    metric_aliases = {
        "cka": ["cka"],
        "agreement": ["prediction_agreement", "agreement"],
        "weight": ["hungarian_weight_similarity", "hungarian_similarity", "weight_similarity"],
    }
    for metric, aliases in metric_aliases.items():
        for comparison, prefixes in {
            "within": ["within_n0", "same_n0", "intra_n0"],
            "cross": ["cross_n0", "between_n0", "inter_n0"],
        }.items():
            candidates = [f"{prefix}_{alias}" for prefix in prefixes for alias in aliases]
            candidates += [f"mean_{candidate}" for candidate in candidates]
            similarity_columns[(comparison, metric)] = find_column(
                comp, candidates, required=False
            )

    required = [(comparison, metric) for comparison in ("within", "cross")
                for metric in ("cka", "agreement", "weight")]
    missing = [f"{comparison}/{metric}" for comparison, metric in required
               if similarity_columns[(comparison, metric)] is None]
    if missing:
        raise KeyError(
            "Missing model-level similarity columns for: " + ", ".join(missing)
            + f". Available columns in {comparative_path.name}: {list(comp.columns)}. "
              "Regenerate comparative_metrics_model_level.csv using the comparative "
              "statistics script, or check its within/cross column names."
        )

    comp_seed = find_column(comp, ["source_seed", "seed"], required=False)
    if comp_seed is not None:
        comp = comp.drop_duplicates(subset=[comp_n0, comp_seed], keep="last")

    n0_values = sorted(
        pd.to_numeric(train[train_n0], errors="coerce").dropna().astype(int).unique()
    )

    rows = []

    for n0 in n0_values:
        t = train[pd.to_numeric(train[train_n0], errors="coerce") == n0].copy()
        s = structural[
            pd.to_numeric(structural[struct_n0], errors="coerce") == n0
            ].copy()
        c = comp[pd.to_numeric(comp[comp_n0], errors="coerce") == n0].copy()

        best_n = mean_sd(t[best_n_col])
        final_n = mean_sd(t[final_n_col])
        best_acc = mean_sd(t[best_acc_col])
        births = mean_sd(t[births_col])
        pruned = mean_sd(t[pruned_col])

        if final_acc_col is not None:
            final_acc = mean_sd(t[final_acc_col])
        else:
            final_acc = (np.nan, np.nan)

        density = mean_sd(s[density_col])
        rho = mean_sd(s[rho_col])
        edges = mean_sd(s[edges_col])
        similarities = {}
        for comparison in ("within", "cross"):
            for metric in ("weight", "cka", "agreement"):
                similarities[(comparison, metric)] = mean_sd(
                    c[similarity_columns[(comparison, metric)]]
                )

        # Agreement may be stored as a fraction or percentage.
        agreement_scales = {}
        for comparison in ("within", "cross"):
            avg = similarities[(comparison, "agreement")][0]
            agreement_scales[comparison] = (
                100.0 if np.isfinite(avg) and avg <= 1.0 else 1.0
            )

        # Accuracy from stats.py is normally already percentage units.
        best_acc_scale = 100.0 if np.isfinite(best_acc[0]) and best_acc[0] <= 1 else 1.0
        final_acc_scale = (
            100.0 if np.isfinite(final_acc[0]) and final_acc[0] <= 1 else 1.0
        )

        rows.append([
            str(n0),
            fmt(*best_n, decimals=1),
            fmt(*final_n, decimals=1),
            fmt(*best_acc, decimals=2, scale=best_acc_scale),
            fmt(*final_acc, decimals=2, scale=final_acc_scale),
            fmt(*births, decimals=1),
            fmt(*pruned, decimals=1),
            fmt(*edges, decimals=1),
            fmt(*density, decimals=3),
            fmt(*rho, decimals=3),
            *[
                fmt(*similarities[(comparison, metric)],
                    decimals=1 if metric == "agreement" else 3,
                    scale=agreement_scales[comparison] if metric == "agreement" else 1.0)
                for comparison in ("within", "cross")
                for metric in ("weight", "cka", "agreement")
            ],
        ])

    # Separate tests for each descriptor used in the structural claim.
    # H0: the descriptor has the same mean across N0 groups.
    # Holm correction controls the family of four structural tests.
    test_specs = [
        (r"$N_f$", structural, struct_n0,
         find_column(structural, ["final_neurons", "final_n"])),
        (r"$E_f$", structural, struct_n0, edges_col),
        ("density", structural, struct_n0, density_col),
        (r"$\rho$", structural, struct_n0, rho_col),
    ]
    structural_tests = []
    for label, frame, n0_col, value_col in test_specs:
        groups = [
            frame[pd.to_numeric(frame[n0_col], errors="coerce") == n0][value_col]
            for n0 in n0_values
        ]
        F, df1, df2, p = welch_anova(groups)
        structural_tests.append(
            {"label": label, "F": F, "df1": df1, "df2": df2, "p": p}
        )

    adjusted = holm_adjust([x["p"] for x in structural_tests])
    for result, p_adj in zip(structural_tests, adjusted):
        result["p_holm"] = p_adj

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Effect of initial network size $N_0$ on the structural and functional properties of FlowGenesis on FashionMNIST. Values are mean $\pm$ standard deviation across seeds; similarity statistics are aggregated at the model level.}",
        r"\label{tab:n0_ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{c cc cc cc ccc ccc ccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Network size}",
        r"& \multicolumn{2}{c}{Performance}",
        r"& \multicolumn{2}{c}{Structural dynamics}",
        r"& \multicolumn{3}{c}{Final structure}",
        r"& \multicolumn{3}{c}{Within-$N_0$ similarity} & \multicolumn{3}{c}{Cross-$N_0$ similarity} \\",
        r"\cmidrule(lr){2-3}",
        r"\cmidrule(lr){4-5}",
        r"\cmidrule(lr){6-7}",
        r"\cmidrule(lr){8-10}",
        r"\cmidrule(lr){11-13}",
        r"\cmidrule(lr){14-16}",
        r"$N_0$",
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
        r"& Agreement (\%)",
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
        r"\begin{minipage}{\textwidth}",
        r"\footnotesize",
        r"\textit{Dependence of final structure on $N_0$:} separate Welch "
        r"one-way ANOVAs for $N_f$, $E_f$, density, and $\rho$, with Holm "
        r"correction across the four tests. Adjusted $p$-values: "
        + ", ".join(f"{x['label']} {fmt_p(x['p_holm'])}" for x in structural_tests)
        + r".",
        r"\end{minipage}",
        r"\end{table*}",
        "",
        ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved LaTeX table: {output}")

    missing_final = int(
        pd.to_numeric(train[final_acc_col], errors="coerce").isna().sum()
    )
    if missing_final:
        print(f"WARNING: final accuracy could not be parsed for {missing_final} run(s).")

    print("\nStructural N0 tests:")
    for result in structural_tests:
        print(
            f"  {result['label']:>8s}: F={result['F']:.4g}, "
            f"raw p={result['p']:.4g}, Holm p={result['p_holm']:.4g}"
        )


if __name__ == "__main__":
    main()
