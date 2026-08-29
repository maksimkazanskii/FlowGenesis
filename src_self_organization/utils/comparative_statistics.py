"""
Comparative statistics for FlowGenesis final architectures.

For every pair of trained models within each ablation condition, computes:
  1. Linear CKA of final hidden representations on a fixed probe set
  2. Hungarian-aligned neuron weight-profile cosine similarity
  3. Prediction agreement on the same probe set
  4. Matched-neuron fraction (diagnostic)

Pair types:
  - within_n0: different seeds, same initial N0
  - cross_n0:  different initial N0 values (all seed x seed pairs)

IMPORTANT:
Pairwise similarities are NOT independent observations because each trained
model appears in many pairs. Therefore this script:
  - saves all pairwise values for descriptive analysis;
  - summarizes all pairs by N0 x N0;
  - creates one independent "model-level" score per trained model by averaging
    its similarities to models in each target N0 group;
  - tests N0 dependence using those model-level scores, not the raw pairs.

For the N0 test, each source model is assigned the mean similarity to models
from OTHER N0 groups. Welch's one-way ANOVA compares these independent
per-model cross-N0 similarity scores across source N0. Holm correction is
applied across the three primary metrics within each condition.

The script reproduces the FlowGenesis forward pass directly from state_dict,
so it does not need to reconstruct a NeuralContainer object. It assumes the
saved model uses:
    input_drive = input_layer(x)
    W_eff = (W * mask) / degree
    h <- 0.8 h + 0.2 ReLU(input_drive + recurrent)
for --relaxation-steps iterations (default 10).

Supported image datasets:
    MNIST, FashionMNIST, CIFAR10, CIFAR100

Example:
    python -m src_self_organization.utils.comparative_statistics \
        data/so/warmup_ablation/fashionmnist

Outputs are written to the experiment root:
    comparative_probe_indices.npy
    comparative_model_outputs.pt
    comparative_metrics_pairs.csv
    comparative_metrics_n0_matrix.csv
    comparative_metrics_model_level.csv
    comparative_metrics_n0_tests.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import f as f_distribution
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor


PRIMARY_METRICS = (
    "cka",
    "hungarian_weight_similarity",
    "prediction_agreement",
)


# ---------------------------------------------------------------------------
# Generic checkpoint/path helpers
# ---------------------------------------------------------------------------

def _find_tensor(state_dict, preferred_names, ndim=None):
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


def extract_tensors(state_dict):
    _, W = _find_tensor(
        state_dict, ("W", "recurrent_weight", "weight_recurrent"), ndim=2
    )
    _, mask = _find_tensor(
        state_dict, ("mask", "recurrent_mask", "connectivity_mask"), ndim=2
    )
    _, input_weight = _find_tensor(
        state_dict, ("input_layer.weight",), ndim=2
    )
    _, output_weight = _find_tensor(
        state_dict, ("output_layer.weight",), ndim=2
    )
    _, output_bias = _find_tensor(
        state_dict, ("output_layer.bias",), ndim=1
    )
    _, recurrent_bias = _find_tensor(
        state_dict, ("bias",), ndim=1
    )

    missing = [
        name for name, value in (
            ("W", W),
            ("mask", mask),
            ("input_layer.weight", input_weight),
            ("output_layer.weight", output_weight),
            ("output_layer.bias", output_bias),
            ("bias", recurrent_bias),
        )
        if value is None
    ]
    if missing:
        raise KeyError(f"Missing required state_dict tensors: {missing}")

    n = W.shape[0]
    if W.shape != (n, n) or mask.shape != (n, n):
        raise ValueError("Recurrent W/mask shapes are inconsistent.")
    if input_weight.shape[0] != n:
        raise ValueError("Input layer width does not match recurrent width.")
    if output_weight.shape[1] != n:
        raise ValueError("Output layer width does not match recurrent width.")
    if recurrent_bias.numel() != n:
        raise ValueError("Recurrent bias width does not match recurrent width.")

    return {
        "W": W.detach().float().cpu(),
        "mask": mask.detach().float().cpu(),
        "input_weight": input_weight.detach().float().cpu(),
        "output_weight": output_weight.detach().float().cpu(),
        "output_bias": output_bias.detach().float().cpu(),
        "recurrent_bias": recurrent_bias.detach().float().cpu(),
    }


def parse_path_metadata(model_path: Path, root: Path):
    rel_parts = model_path.relative_to(root).parts[:-1]

    seed = None
    initial_neurons = None
    condition_parts = []

    for part in rel_parts:
        m = re.fullmatch(r"seed_(-?\d+)", part)
        if m:
            seed = int(m.group(1))
            continue

        m = re.fullmatch(r"neurons_(-?\d+)", part)
        if m:
            initial_neurons = int(m.group(1))
            continue

        condition_parts.append(part)

    condition = "/".join(condition_parts) if condition_parts else "default"

    if seed is None:
        raise ValueError(f"Could not parse seed from {model_path}")
    if initial_neurons is None:
        raise ValueError(f"Could not parse initial N0 from {model_path}")

    return condition, initial_neurons, seed


def load_runs(root: Path, checkpoint_name: str):
    paths = sorted(root.rglob(checkpoint_name))
    if not paths:
        raise FileNotFoundError(
            f"No {checkpoint_name!r} files found below {root}"
        )

    runs = []
    datasets_seen = set()

    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        tensors = extract_tensors(state_dict)

        condition, n0, seed = parse_path_metadata(path, root)
        dataset_name = str(ckpt.get("dataset", "")).strip()
        if dataset_name:
            datasets_seen.add(dataset_name.lower())

        runs.append({
            "id": str(path.relative_to(root)),
            "path": path,
            "condition": condition,
            "n0": n0,
            "seed": seed,
            "dataset": dataset_name,
            "n_final": int(tensors["W"].shape[0]),
            "tensors": tensors,
        })

    if len(datasets_seen) > 1:
        raise ValueError(
            f"Experiment root contains multiple datasets: {sorted(datasets_seen)}"
        )

    return runs


# ---------------------------------------------------------------------------
# Probe dataset
# ---------------------------------------------------------------------------

def canonical_dataset_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def get_test_dataset(dataset_name: str, data_root: str):
    key = canonical_dataset_name(dataset_name)

    if key == "mnist":
        cls = datasets.MNIST
    elif key == "fashionmnist":
        cls = datasets.FashionMNIST
    elif key == "cifar10":
        cls = datasets.CIFAR10
    elif key == "cifar100":
        cls = datasets.CIFAR100
    else:
        raise ValueError(
            f"Unsupported dataset {dataset_name!r}. "
            "Supported: MNIST, FashionMNIST, CIFAR10, CIFAR100."
        )

    return cls(
        data_root,
        train=False,
        download=True,
        transform=ToTensor(),
    )


def choose_probe_indices(n_total, n_probe, seed, path):
    n_probe = min(int(n_probe), int(n_total))

    if path.exists():
        indices = np.load(path)
        if len(indices) != n_probe:
            raise ValueError(
                f"{path} contains {len(indices)} indices, but --probe-size "
                f"requests {n_probe}. Delete the file to regenerate it."
            )
        if np.any(indices < 0) or np.any(indices >= n_total):
            raise ValueError(f"Invalid probe indices in {path}")
        return indices.astype(np.int64)

    rng = np.random.default_rng(seed)
    indices = np.sort(
        rng.choice(n_total, size=n_probe, replace=False)
    ).astype(np.int64)
    np.save(path, indices)
    return indices


# ---------------------------------------------------------------------------
# Functional FlowGenesis forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_checkpoint_on_probe(
        tensors,
        loader,
        device,
        relaxation_steps,
        gamma,
):
    W = tensors["W"].to(device)
    mask = tensors["mask"].to(device)
    input_weight = tensors["input_weight"].to(device)
    output_weight = tensors["output_weight"].to(device)
    output_bias = tensors["output_bias"].to(device)
    recurrent_bias = tensors["recurrent_bias"].to(device)

    degree = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    recurrent_weight = (W * mask) / degree

    hidden_all = []
    logits_all = []
    pred_all = []
    labels_all = []

    for images, labels in loader:
        x = images.to(device).view(images.size(0), -1)

        if x.shape[1] != input_weight.shape[1]:
            raise ValueError(
                f"Probe input dimension {x.shape[1]} does not match "
                f"checkpoint input dimension {input_weight.shape[1]}."
            )

        input_drive = F.linear(x, input_weight)
        h = torch.zeros_like(input_drive)

        for _ in range(relaxation_steps):
            recurrent = F.linear(h, recurrent_weight, recurrent_bias)
            candidate = F.relu(input_drive + recurrent)
            h = (1.0 - gamma) * h + gamma * candidate

        logits = F.linear(h, output_weight, output_bias)

        hidden_all.append(h.cpu())
        logits_all.append(logits.cpu())
        pred_all.append(logits.argmax(dim=1).cpu())
        labels_all.append(labels.cpu())

    return {
        "hidden": torch.cat(hidden_all, dim=0),
        "logits": torch.cat(logits_all, dim=0),
        "predictions": torch.cat(pred_all, dim=0),
        "labels": torch.cat(labels_all, dim=0),
    }


# ---------------------------------------------------------------------------
# Comparative metrics
# ---------------------------------------------------------------------------

def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps=1e-12) -> float:
    """Linear CKA; X and Y may have different feature dimensions."""
    X = X.double()
    Y = Y.double()
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    cross = X.T @ Y
    xx = X.T @ X
    yy = Y.T @ Y

    numerator = torch.sum(cross * cross)
    denominator = torch.linalg.norm(xx) * torch.linalg.norm(yy)

    if denominator <= eps:
        return float("nan")
    return float((numerator / denominator).item())


def row_cosine_matrix(A: np.ndarray, B: np.ndarray, eps=1e-12):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)

    A_unit = A / np.maximum(A_norm, eps)
    B_unit = B / np.maximum(B_norm, eps)
    return A_unit @ B_unit.T


def neuron_profiles(tensors):
    """
    Per-neuron parameter profile in common coordinates.

    We concatenate:
      - incoming input weights
      - output weights for that neuron
      - recurrent bias

    These dimensions are shared even when the two networks have unequal width.
    The recurrent NxN matrix itself cannot be directly concatenated before
    alignment because its coordinate system is neuron-permutation dependent.
    """
    inp = tensors["input_weight"].numpy()             # N x input_dim
    out = tensors["output_weight"].numpy().T          # N x output_dim
    bias = tensors["recurrent_bias"].numpy()[:, None] # N x 1
    return np.concatenate([inp, out, bias], axis=1)


def hungarian_weight_similarity(tensors_a, tensors_b):
    A = neuron_profiles(tensors_a)
    B = neuron_profiles(tensors_b)

    sim = row_cosine_matrix(A, B)
    row_ind, col_ind = linear_sum_assignment(-sim)
    matched = sim[row_ind, col_ind]

    n_a, n_b = A.shape[0], B.shape[0]
    matched_fraction = min(n_a, n_b) / max(n_a, n_b)

    return float(np.mean(matched)), float(matched_fraction)


def prediction_agreement(pred_a, pred_b):
    if pred_a.shape != pred_b.shape:
        raise ValueError("Prediction vectors have different shapes.")
    return float((pred_a == pred_b).float().mean().item())


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mean_sd_sem_cv(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"),) * 4

    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if len(x) >= 2 else float("nan")
    sem = sd / math.sqrt(len(x)) if len(x) >= 2 else float("nan")
    cv = sd / abs(mean) if len(x) >= 2 and mean != 0 else float("nan")
    return mean, sd, sem, cv


def welch_anova(groups):
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return (float("nan"),) * 4

    k = len(groups)
    n = np.asarray([len(g) for g in groups], dtype=float)
    means = np.asarray([np.mean(g) for g in groups], dtype=float)
    variances = np.asarray([np.var(g, ddof=1) for g in groups], dtype=float)

    # Degenerate but identical groups: no evidence of a difference.
    if np.all(variances == 0):
        if np.allclose(means, means[0]):
            return 0.0, float(k - 1), float("inf"), 1.0
        return float("nan"), float(k - 1), float("nan"), float("nan")

    # Welch weights require positive variances.
    if np.any(variances <= 0) or np.any(~np.isfinite(variances)):
        return (float("nan"),) * 4

    weights = n / variances
    w_sum = weights.sum()
    weighted_mean = np.sum(weights * means) / w_sum

    numerator = np.sum(weights * (means - weighted_mean) ** 2) / (k - 1)
    term = np.sum(
        (1.0 / (n - 1.0)) * (1.0 - weights / w_sum) ** 2
    )
    correction = 1.0 + (2.0 * (k - 2.0) / (k**2 - 1.0)) * term
    F_stat = numerator / correction

    df1 = float(k - 1)
    df2 = float((k**2 - 1.0) / (3.0 * term)) if term > 0 else float("inf")
    p = float(f_distribution.sf(F_stat, df1, df2))

    return float(F_stat), df1, df2, p


def holm_adjust(p_values):
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


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comparative statistics for FlowGenesis architectures."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--checkpoint", default="final_model.pt")
    parser.add_argument("--probe-size", type=int, default=1000)
    parser.add_argument("--probe-seed", type=int, default=20260829)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--relaxation-steps", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override dataset name if it is absent from checkpoints.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    runs = load_runs(root, args.checkpoint)

    dataset_names = {
        canonical_dataset_name(r["dataset"])
        for r in runs if r["dataset"]
    }
    if args.dataset is not None:
        dataset_name = args.dataset
    elif len(dataset_names) == 1:
        # Preserve a checkpoint spelling if possible.
        dataset_name = next(r["dataset"] for r in runs if r["dataset"])
    else:
        raise ValueError(
            "Could not infer one dataset from checkpoints. Pass --dataset."
        )

    print(f"Found {len(runs)} models")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {args.device}")

    dataset = get_test_dataset(dataset_name, args.data_root)
    probe_path = root / "comparative_probe_indices.npy"
    probe_indices = choose_probe_indices(
        len(dataset), args.probe_size, args.probe_seed, probe_path
    )
    probe = Subset(dataset, probe_indices.tolist())
    loader = DataLoader(
        probe,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Run every trained model only once on the fixed probe set.
    outputs = {}
    for i, run in enumerate(runs, 1):
        print(
            f"[{i:3d}/{len(runs)}] {run['condition']} "
            f"N0={run['n0']} seed={run['seed']} N={run['n_final']}"
        )
        outputs[run["id"]] = run_checkpoint_on_probe(
            run["tensors"],
            loader,
            args.device,
            args.relaxation_steps,
            args.gamma,
        )

    # Cache compact model outputs so pair metrics can be recomputed cheaply.
    torch.save(
        {
            "dataset": dataset_name,
            "probe_indices": torch.as_tensor(probe_indices),
            "relaxation_steps": args.relaxation_steps,
            "gamma": args.gamma,
            "outputs": outputs,
        },
        root / "comparative_model_outputs.pt",
        )

    # All unique model pairs, but only within the same ablation condition.
    pair_rows = []
    by_condition = defaultdict(list)
    for run in runs:
        by_condition[run["condition"]].append(run)

    for condition, condition_runs in sorted(by_condition.items()):
        print(f"\nPairwise comparisons: {condition}")

        for a, b in itertools.combinations(condition_runs, 2):
            oa = outputs[a["id"]]
            ob = outputs[b["id"]]

            cka = linear_cka(oa["hidden"], ob["hidden"])
            hung, matched_fraction = hungarian_weight_similarity(
                a["tensors"], b["tensors"]
            )
            agreement = prediction_agreement(
                oa["predictions"], ob["predictions"]
            )

            pair_rows.append({
                "condition": condition,
                "n0_a": a["n0"],
                "seed_a": a["seed"],
                "n_final_a": a["n_final"],
                "model_a": a["id"],
                "n0_b": b["n0"],
                "seed_b": b["seed"],
                "n_final_b": b["n_final"],
                "model_b": b["id"],
                "pair_type": (
                    "within_n0" if a["n0"] == b["n0"] else "cross_n0"
                ),
                "cka": cka,
                "hungarian_weight_similarity": hung,
                "matched_fraction": matched_fraction,
                "prediction_agreement": agreement,
            })

    pair_fields = [
        "condition",
        "n0_a", "seed_a", "n_final_a", "model_a",
        "n0_b", "seed_b", "n_final_b", "model_b",
        "pair_type",
        "cka",
        "hungarian_weight_similarity",
        "matched_fraction",
        "prediction_agreement",
    ]
    write_csv(
        root / "comparative_metrics_pairs.csv",
        pair_rows,
        pair_fields,
        )

    # N0 x N0 descriptive matrix. Raw pairs are used only descriptively.
    matrix_rows = []
    matrix_groups = defaultdict(lambda: defaultdict(list))

    for row in pair_rows:
        lo, hi = sorted((int(row["n0_a"]), int(row["n0_b"])))
        key = (row["condition"], lo, hi)
        for metric in PRIMARY_METRICS + ("matched_fraction",):
            matrix_groups[key][metric].append(float(row[metric]))

    for (condition, n0_a, n0_b), metric_map in sorted(matrix_groups.items()):
        out = {
            "condition": condition,
            "n0_a": n0_a,
            "n0_b": n0_b,
            "pair_type": "within_n0" if n0_a == n0_b else "cross_n0",
        }

        # Pair count is identical for all metrics unless a metric is NaN.
        first = np.asarray(metric_map[PRIMARY_METRICS[0]], dtype=float)
        out["n_pairs"] = int(np.isfinite(first).sum())

        for metric in PRIMARY_METRICS + ("matched_fraction",):
            mean, sd, sem, cv = mean_sd_sem_cv(metric_map[metric])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_sd"] = sd
            out[f"{metric}_sem"] = sem
            out[f"{metric}_cv"] = cv

        matrix_rows.append(out)

    matrix_fields = [
        "condition", "n0_a", "n0_b", "pair_type", "n_pairs"
    ]
    for metric in PRIMARY_METRICS + ("matched_fraction",):
        matrix_fields += [
            f"{metric}_mean",
            f"{metric}_sd",
            f"{metric}_sem",
            f"{metric}_cv",
        ]

    write_csv(
        root / "comparative_metrics_n0_matrix.csv",
        matrix_rows,
        matrix_fields,
        )

    # ------------------------------------------------------------------
    # Independent model-level units
    #
    # For each trained model:
    #   within_n0_* = average similarity to other seeds at same N0
    #   cross_n0_*  = average similarity to ALL models at other N0 values
    #
    # The cross_n0 score is the primary inferential unit. There is one
    # observation per trained model, avoiding pseudoreplication from treating
    # all 25 seed-pairs as independent.
    # ------------------------------------------------------------------
    adjacency = defaultdict(lambda: defaultdict(list))

    for row in pair_rows:
        id_a = row["model_a"]
        id_b = row["model_b"]
        same = row["pair_type"] == "within_n0"

        for metric in PRIMARY_METRICS + ("matched_fraction",):
            value = float(row[metric])
            bucket = "within" if same else "cross"
            adjacency[id_a][f"{bucket}:{metric}"].append(value)
            adjacency[id_b][f"{bucket}:{metric}"].append(value)

    model_level_rows = []
    for run in runs:
        out = {
            "condition": run["condition"],
            "n0": run["n0"],
            "seed": run["seed"],
            "n_final": run["n_final"],
            "model": run["id"],
        }

        for metric in PRIMARY_METRICS + ("matched_fraction",):
            within = adjacency[run["id"]].get(f"within:{metric}", [])
            cross = adjacency[run["id"]].get(f"cross:{metric}", [])

            out[f"within_n0_{metric}"] = (
                float(np.nanmean(within)) if len(within) else float("nan")
            )
            out[f"cross_n0_{metric}"] = (
                float(np.nanmean(cross)) if len(cross) else float("nan")
            )
            out[f"n_within_pairs_{metric}"] = len(within)
            out[f"n_cross_pairs_{metric}"] = len(cross)

        model_level_rows.append(out)

    model_fields = ["condition", "n0", "seed", "n_final", "model"]
    for metric in PRIMARY_METRICS + ("matched_fraction",):
        model_fields += [
            f"within_n0_{metric}",
            f"cross_n0_{metric}",
            f"n_within_pairs_{metric}",
            f"n_cross_pairs_{metric}",
        ]

    write_csv(
        root / "comparative_metrics_model_level.csv",
        model_level_rows,
        model_fields,
        )

    # Welch ANOVA across source N0 using independent model-level cross-N0
    # scores. Holm correct the three primary metrics within each condition.
    test_rows = []

    for condition in sorted(by_condition):
        condition_model_rows = [
            r for r in model_level_rows if r["condition"] == condition
        ]

        for metric in PRIMARY_METRICS:
            grouped = defaultdict(list)
            col = f"cross_n0_{metric}"

            for row in condition_model_rows:
                value = float(row[col])
                if np.isfinite(value):
                    grouped[int(row["n0"])].append(value)

            usable = {
                n0: vals for n0, vals in sorted(grouped.items())
                if len(vals) >= 2
            }

            if len(usable) >= 2:
                F_stat, df1, df2, p = welch_anova(list(usable.values()))
                status = "ok" if np.isfinite(p) else "welch_anova_undefined"
            else:
                F_stat = df1 = df2 = p = float("nan")
                status = "insufficient_n0_groups"

            test_rows.append({
                "condition": condition,
                "metric": metric,
                "score": "model_mean_similarity_to_other_n0",
                "n0_levels": ";".join(map(str, usable.keys())),
                "n_n0_levels": len(usable),
                "model_counts": ";".join(
                    f"{n0}:{len(vals)}" for n0, vals in usable.items()
                ),
                "welch_F": F_stat,
                "df1": df1,
                "df2": df2,
                "p_value": p,
                "significant_at_alpha": (
                    bool(p < args.alpha) if np.isfinite(p) else ""
                ),
                "alpha": args.alpha,
                "status": status,
            })

    # Holm family = the three primary comparative metrics within condition.
    condition_indices = defaultdict(list)
    for idx, row in enumerate(test_rows):
        condition_indices[row["condition"]].append(idx)

    for condition, indices in condition_indices.items():
        adjusted = holm_adjust(
            [test_rows[i]["p_value"] for i in indices]
        )
        for i, p_adj in zip(indices, adjusted):
            test_rows[i]["p_value_holm"] = (
                float(p_adj) if np.isfinite(p_adj) else float("nan")
            )
            test_rows[i]["significant_holm_at_alpha"] = (
                bool(p_adj < args.alpha) if np.isfinite(p_adj) else ""
            )

    test_fields = [
        "condition",
        "metric",
        "score",
        "n0_levels",
        "n_n0_levels",
        "model_counts",
        "welch_F",
        "df1",
        "df2",
        "p_value",
        "significant_at_alpha",
        "p_value_holm",
        "significant_holm_at_alpha",
        "alpha",
        "status",
    ]
    write_csv(
        root / "comparative_metrics_n0_tests.csv",
        test_rows,
        test_fields,
        )

    print("\nDone.")
    print(f"Probe indices : {probe_path}")
    print(f"Pair metrics  : {root / 'comparative_metrics_pairs.csv'}")
    print(f"N0 matrix     : {root / 'comparative_metrics_n0_matrix.csv'}")
    print(f"Model level   : {root / 'comparative_metrics_model_level.csv'}")
    print(f"N0 tests      : {root / 'comparative_metrics_n0_tests.csv'}")


if __name__ == "__main__":
    main()
