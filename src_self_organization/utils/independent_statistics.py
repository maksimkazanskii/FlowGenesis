"""Compute independent architecture metrics for FlowGenesis experiment runs.

Metrics per run:
  1. final neuron count N
  2. recurrent connectivity density
  3. spectral radius of W_eff = W * mask

The script recursively scans an experiment root for final_model.pt files.
It is intended for directory layouts such as:

  data/so/warmup_ablation/fashionmnist/
      warmup_0/neurons_10/seed_1/final_model.pt
      ...

Outputs:
  independent_metrics_runs.csv       one row per run
  independent_metrics_summary.csv    mean/std across seeds per condition
  independent_metrics_seed_level.csv seed variability within each N0
  independent_metrics_n0_tests.csv   Welch ANOVA across N0 + Holm correction within each condition

Example:
  python -m src_self_organization.utils.analyze_independent_architecture_metrics \
      data/so/warmup_ablation/fashionmnist
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import f as f_distribution


def _find_tensor(state_dict, preferred_names, ndim=None):
    """Find a tensor by exact/suffix key, with conservative fallback."""
    for name in preferred_names:
        if name in state_dict:
            x = state_dict[name]
            if torch.is_tensor(x) and (ndim is None or x.ndim == ndim):
                return name, x

    for key, x in state_dict.items():
        if not torch.is_tensor(x):
            continue
        if ndim is not None and x.ndim != ndim:
            continue
        if any(key.endswith(name) for name in preferred_names):
            return key, x

    return None, None


def extract_recurrent_weight_and_mask(state_dict):
    """Extract recurrent W and mask from a NeuralContainer state_dict."""
    w_key, W = _find_tensor(
        state_dict,
        preferred_names=("W", "recurrent_weight", "weight_recurrent"),
        ndim=2,
    )
    m_key, mask = _find_tensor(
        state_dict,
        preferred_names=("mask", "recurrent_mask", "connectivity_mask"),
        ndim=2,
    )

    # Fallback: identify a square matrix matching the mask shape.
    if mask is not None and W is None:
        for key, x in state_dict.items():
            if (
                    torch.is_tensor(x)
                    and x.ndim == 2
                    and x.shape == mask.shape
                    and x.shape[0] == x.shape[1]
                    and "mask" not in key.lower()
            ):
                w_key, W = key, x
                break

    if W is None or mask is None:
        keys = ", ".join(state_dict.keys())
        raise KeyError(
            "Could not identify recurrent weight and mask in state_dict. "
            f"Available keys: {keys}"
        )

    if W.shape != mask.shape or W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(
            f"Invalid recurrent tensors: {w_key}={tuple(W.shape)}, "
            f"{m_key}={tuple(mask.shape)}"
        )

    return w_key, W, m_key, mask


def parse_path_metadata(model_path: Path, root: Path):
    """Read condition / N0 / seed from directory names without assuming ablation type."""
    rel_parts = model_path.relative_to(root).parts[:-1]

    seed = None
    initial_neurons = None
    condition_parts = []

    for part in rel_parts:
        m = re.fullmatch(r"seed_(-?\d+)", part)
        if m:
            seed = int(m.group(1))
            continue

        m = re.fullmatch(r"neurons_(\d+)", part)
        if m:
            initial_neurons = int(m.group(1))
            continue

        condition_parts.append(part)

    condition = "/".join(condition_parts) if condition_parts else "."
    return condition, initial_neurons, seed


def compute_metrics(model_path: Path, root: Path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    _, W, _, mask = extract_recurrent_weight_and_mask(state_dict)

    W = W.detach().cpu().double()
    mask = mask.detach().cpu().double()

    n = int(W.shape[0])

    # Density over all N^2 recurrent positions, exactly matching the stored mask.
    # This avoids silently assuming that self-connections are forbidden.
    connected = mask != 0
    density = float(connected.double().mean().item())
    edge_count = int(connected.sum().item())

    W_eff = W * mask
    eigvals = torch.linalg.eigvals(W_eff)
    spectral_radius = float(torch.abs(eigvals).max().item()) if n else float("nan")

    condition, n0_path, seed_path = parse_path_metadata(model_path, root)

    seed = checkpoint.get("seed", seed_path)
    n0 = checkpoint.get("initial_hidden_neurons", n0_path)

    return {
        "condition": condition,
        "initial_neurons": n0,
        "seed": seed,
        "final_neurons": n,
        "edge_count": edge_count,
        "connectivity_density": density,
        "spectral_radius": spectral_radius,
        "model_path": str(model_path),
    }


def mean_std(values):
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan")
    if len(a) == 1:
        return float(a[0]), float("nan")
    return float(a.mean()), float(a.std(ddof=1))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def welch_anova(groups):
    """Welch's one-way ANOVA for groups with potentially unequal variances.

    Returns (F, df1, df2, p_value). Requires at least two groups with at
    least two observations each and non-zero sample variances.
    """
    groups = [np.asarray(g, dtype=float) for g in groups]
    k = len(groups)
    if k < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    n = np.asarray([len(g) for g in groups], dtype=float)
    means = np.asarray([np.mean(g) for g in groups], dtype=float)
    variances = np.asarray([np.var(g, ddof=1) for g in groups], dtype=float)

    if np.any(n < 2) or np.any(~np.isfinite(variances)) or np.any(variances <= 0):
        return float("nan"), float("nan"), float("nan"), float("nan")

    weights = n / variances
    w_sum = np.sum(weights)
    weighted_mean = np.sum(weights * means) / w_sum

    numerator = np.sum(weights * (means - weighted_mean) ** 2) / (k - 1)

    term = np.sum((1.0 / (n - 1.0)) * (1.0 - weights / w_sum) ** 2)
    correction = 1.0 + (2.0 * (k - 2.0) / (k**2 - 1.0)) * term
    F = numerator / correction

    df1 = float(k - 1)
    df2 = float((k**2 - 1.0) / (3.0 * term)) if term > 0 else float("inf")
    p = float(f_distribution.sf(F, df1, df2))

    return float(F), df1, df2, p


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values, preserving original order."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan, dtype=float)

    valid = np.where(np.isfinite(p))[0]
    if len(valid) == 0:
        return adjusted

    order = valid[np.argsort(p[valid])]
    m = len(order)

    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max

    return adjusted


def compute_two_level_tests(rows, alpha=0.05):
    """Two-level architecture analysis within each condition.

    Level 1 (seed level):
        For every condition x N0 cell, summarize replicate seeds using
        mean, sample SD, CV, and SEM. Seeds are replicates, not groups.

    Level 2 (N0 level):
        For every condition and metric, run Welch's one-way ANOVA across
        all available N0 groups. Welch's ANOVA tests equality of means while
        allowing unequal variances and unequal sample sizes across N0.

        H0: all N0 population means are equal.

    Different conditions (e.g. warm-up values) are NEVER compared.
    """
    metrics = ("final_neurons", "connectivity_density", "spectral_radius")
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    seed_rows = []
    n0_test_rows = []

    for condition, condition_rows in sorted(by_condition.items(), key=lambda kv: str(kv[0])):
        n0_values = sorted({
            r["initial_neurons"] for r in condition_rows
            if r["initial_neurons"] is not None
        })

        # Level 1: quantify seed-to-seed variability for each N0.
        for n0 in n0_values:
            cell = [r for r in condition_rows if r["initial_neurons"] == n0]
            seeds = sorted(r["seed"] for r in cell if r["seed"] is not None)

            for metric in metrics:
                values = np.asarray(
                    [float(r[metric]) for r in cell if np.isfinite(float(r[metric]))],
                    dtype=float,
                )
                n = len(values)
                mean = float(np.mean(values)) if n else float("nan")
                sd = float(np.std(values, ddof=1)) if n >= 2 else float("nan")
                sem = float(sd / np.sqrt(n)) if n >= 2 else float("nan")
                cv = (
                    float(sd / abs(mean))
                    if n >= 2 and np.isfinite(mean) and not np.isclose(mean, 0.0)
                    else float("nan")
                )

                seed_rows.append({
                    "condition": condition,
                    "initial_neurons": n0,
                    "metric": metric,
                    "n_seeds": n,
                    "seeds": ";".join(map(str, seeds)),
                    "mean": mean,
                    "sd": sd,
                    "sem": sem,
                    "cv": cv,
                })

        # Level 2: one omnibus N0 test per metric.
        for metric in metrics:
            groups = []
            used_n0 = []
            group_sizes = []

            for n0 in n0_values:
                vals = np.asarray([
                    float(r[metric])
                    for r in condition_rows
                    if r["initial_neurons"] == n0 and np.isfinite(float(r[metric]))
                ], dtype=float)
                if len(vals) >= 2:
                    groups.append(vals)
                    used_n0.append(n0)
                    group_sizes.append(len(vals))

            status = "ok"
            f_stat = p_value = eta_squared = omega_squared = float("nan")
            df1 = df2 = float("nan")

            if len(groups) < 2:
                status = "insufficient_n0_groups"
            else:
                f_stat, df1, df2, p_value = welch_anova(groups)
                if not np.isfinite(f_stat):
                    status = "welch_anova_undefined"

                all_values = np.concatenate(groups)
                grand_mean = float(np.mean(all_values))
                ss_between = float(sum(
                    len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in groups
                ))
                ss_within = float(sum(
                    np.sum((g - float(np.mean(g))) ** 2) for g in groups
                ))
                ss_total = ss_between + ss_within
                eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

                k = len(groups)
                n_total = len(all_values)
                df_between = k - 1
                df_within = n_total - k
                ms_within = ss_within / df_within if df_within > 0 else float("nan")
                denom = ss_total + ms_within if np.isfinite(ms_within) else float("nan")
                omega_squared = (
                    max(0.0, (ss_between - df_between * ms_within) / denom)
                    if np.isfinite(denom) and denom > 0 else float("nan")
                )

            n0_test_rows.append({
                "condition": condition,
                "metric": metric,
                "n0_levels": ";".join(map(str, used_n0)),
                "n_n0_levels": len(used_n0),
                "seed_counts": ";".join(map(str, group_sizes)),
                "welch_F": f_stat,
                "df1": df1,
                "df2": df2,
                "p_value": p_value,
                "significant_at_alpha": bool(p_value < alpha) if np.isfinite(p_value) else "",
                "eta_squared": eta_squared,
                "omega_squared": omega_squared,
                "alpha": alpha,
                "status": status,
            })

    # Multiple-testing correction: within each condition, the three
    # architectural metrics (N, density, spectral radius) form one family.
    # Different conditions/ablation values remain independent analyses.
    condition_indices = defaultdict(list)
    for idx, row in enumerate(n0_test_rows):
        condition_indices[row["condition"]].append(idx)

    for condition, indices in condition_indices.items():
        raw_p = [n0_test_rows[i]["p_value"] for i in indices]
        adjusted = holm_adjust(raw_p)
        for i, p_adj in zip(indices, adjusted):
            n0_test_rows[i]["p_value_holm"] = (
                float(p_adj) if np.isfinite(p_adj) else float("nan")
            )
            n0_test_rows[i]["significant_holm_at_alpha"] = (
                bool(p_adj < alpha) if np.isfinite(p_adj) else ""
            )

    return seed_rows, n0_test_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="Experiment root to scan recursively")
    parser.add_argument(
        "--model-name",
        default="final_model.pt",
        help="Checkpoint filename to analyze (default: final_model.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: experiment root)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for N0 Welch ANOVA tests (default: 0.05)",
    )
    args = parser.parse_args()

    root = args.folder.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else root

    model_paths = sorted(root.rglob(args.model_name))
    if not model_paths:
        raise FileNotFoundError(f"No {args.model_name} files found under {root}")

    rows = []
    for path in model_paths:
        try:
            row = compute_metrics(path, root)
            rows.append(row)
            print(
                f"{row['condition']:20s} | "
                f"N0={str(row['initial_neurons']):>4s} | "
                f"seed={str(row['seed']):>3s} | "
                f"N={row['final_neurons']:4d} | "
                f"density={row['connectivity_density']:.4f} | "
                f"rho={row['spectral_radius']:.4f}"
            )
        except Exception as exc:
            print(f"WARNING: skipping {path}: {exc}")

    if not rows:
        raise RuntimeError("No checkpoints could be analyzed.")

    run_fields = [
        "condition",
        "initial_neurons",
        "seed",
        "final_neurons",
        "edge_count",
        "connectivity_density",
        "spectral_radius",
        "model_path",
    ]
    write_csv(output_dir / "independent_metrics_runs.csv", rows, run_fields)

    # Aggregate only across seeds: condition + initial N0 define an experiment cell.
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["initial_neurons"])].append(row)

    summary_rows = []
    for (condition, n0), group in sorted(
            grouped.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))
    ):
        n_mean, n_sd = mean_std([x["final_neurons"] for x in group])
        d_mean, d_sd = mean_std([x["connectivity_density"] for x in group])
        r_mean, r_sd = mean_std([x["spectral_radius"] for x in group])

        summary_rows.append(
            {
                "condition": condition,
                "initial_neurons": n0,
                "n_seeds": len(group),
                "final_neurons_mean": n_mean,
                "final_neurons_sd": n_sd,
                "density_mean": d_mean,
                "density_sd": d_sd,
                "spectral_radius_mean": r_mean,
                "spectral_radius_sd": r_sd,
            }
        )

    summary_fields = [
        "condition",
        "initial_neurons",
        "n_seeds",
        "final_neurons_mean",
        "final_neurons_sd",
        "density_mean",
        "density_sd",
        "spectral_radius_mean",
        "spectral_radius_sd",
    ]
    write_csv(output_dir / "independent_metrics_summary.csv", summary_rows, summary_fields)

    # Two-level analysis within each condition:
    # Level 1: seed variability within each N0.
    # Level 2: Welch ANOVA across N0 levels using seeds as replicates.
    seed_test_rows, n0_test_rows = compute_two_level_tests(rows, alpha=args.alpha)

    seed_fields = [
        "condition",
        "initial_neurons",
        "metric",
        "n_seeds",
        "seeds",
        "mean",
        "sd",
        "sem",
        "cv",
    ]
    write_csv(
        output_dir / "independent_metrics_seed_level.csv",
        seed_test_rows,
        seed_fields,
        )

    n0_fields = [
        "condition",
        "metric",
        "n0_levels",
        "n_n0_levels",
        "seed_counts",
        "welch_F",
        "df1",
        "df2",
        "p_value",
        "significant_at_alpha",
        "p_value_holm",
        "significant_holm_at_alpha",
        "eta_squared",
        "omega_squared",
        "alpha",
        "status",
    ]
    write_csv(
        output_dir / "independent_metrics_n0_tests.csv",
        n0_test_rows,
        n0_fields,
        )

    print("\nSaved:")
    print(f"  {output_dir / 'independent_metrics_runs.csv'}")
    print(f"  {output_dir / 'independent_metrics_summary.csv'}")
    print(f"  {output_dir / 'independent_metrics_seed_level.csv'}")
    print(f"  {output_dir / 'independent_metrics_n0_tests.csv'}")


if __name__ == "__main__":
    main()

