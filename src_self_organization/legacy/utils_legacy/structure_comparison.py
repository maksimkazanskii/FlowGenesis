"""
Functional comparison of two neurogenesis networks.

Each hidden neuron is represented by its response profile across
the same fixed probe dataset:

    R_i = [h_i(x_1), ..., h_i(x_M)]

Neuron-to-neuron functional similarity is measured using Pearson
correlation.

The script also performs one-to-one Hungarian matching between
neurons from the two networks.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

from .config import Config
from .model_neurogenesis import NeuralContainer


# ================================================================
# Load model
# ================================================================

def load_model(checkpoint_path, device):

    checkpoint_path = Path(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    cfg = Config()

    # The model must be constructed with the exact number of
    # neurons present when this checkpoint was saved.
    cfg.hidden_neurons = int(
        checkpoint["hidden_neurons"]
    )

    if "dataset" in checkpoint:
        cfg.dataset = checkpoint["dataset"]

    model = NeuralContainer(cfg).to(device)

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.eval()

    return model, checkpoint


# ================================================================
# Probe dataset
# ================================================================

def get_probe_loader(
        dataset_name,
        n_probe,
        batch_size,
):

    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        dataset_class = datasets.MNIST

    elif dataset_name == "fashionmnist":
        dataset_class = datasets.FashionMNIST

    else:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    dataset = dataset_class(
        "./data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    n_probe = min(
        n_probe,
        len(dataset),
    )

    # Deterministic:
    # both networks receive EXACTLY the same examples.
    subset = Subset(
        dataset,
        list(range(n_probe)),
    )

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
    )

    return loader


# ================================================================
# Response profiles
# ================================================================

@torch.no_grad()
def extract_response_profiles(
        model,
        loader,
        device,
):

    model.eval()

    all_hidden = []
    all_labels = []

    for images, labels in loader:

        images = images.to(device)

        _, hidden = model(
            images,
            return_hidden=True,
        )

        all_hidden.append(
            hidden.detach().cpu()
        )

        all_labels.append(
            labels.cpu()
        )

    # [samples, neurons]
    H = torch.cat(
        all_hidden,
        dim=0,
    )

    labels = torch.cat(
        all_labels,
        dim=0,
    )

    # Convert to:
    #
    # [neurons, samples]
    #
    # Every row is one neuron's functional fingerprint.
    profiles = H.T.contiguous()

    return profiles, labels


# ================================================================
# Pearson correlation
# ================================================================

def correlation_matrix(
        profiles_a,
        profiles_b,
):

    A = profiles_a.numpy().astype(
        np.float64
    )

    B = profiles_b.numpy().astype(
        np.float64
    )

    # ------------------------------------------------------------
    # Center every neuron's response profile
    # ------------------------------------------------------------

    A = A - A.mean(
        axis=1,
        keepdims=True,
    )

    B = B - B.mean(
        axis=1,
        keepdims=True,
    )

    # ------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------

    norm_a = np.linalg.norm(
        A,
        axis=1,
        keepdims=True,
    )

    norm_b = np.linalg.norm(
        B,
        axis=1,
        keepdims=True,
    )

    # Detect effectively constant neurons
    active_a = (
            norm_a[:, 0] > 1e-12
    )

    active_b = (
            norm_b[:, 0] > 1e-12
    )

    A = A / np.maximum(
        norm_a,
        1e-12,
    )

    B = B / np.maximum(
        norm_b,
        1e-12,
    )

    # ------------------------------------------------------------
    # Pearson correlation matrix
    #
    # [N_A, samples] @ [samples, N_B]
    #       -> [N_A, N_B]
    # ------------------------------------------------------------

    corr = A @ B.T

    # Constant neurons have no meaningful Pearson correlation.
    corr[
        ~active_a,
        :
    ] = np.nan

    corr[
        :,
        ~active_b
    ] = np.nan

    return corr, active_a, active_b


# ================================================================
# Class response profiles
# ================================================================

def class_profiles(
        profiles,
        labels,
):

    labels_np = labels.numpy()

    classes = np.unique(
        labels_np
    )

    result = []

    for c in classes:

        mask = labels_np == c

        class_mean = (
            profiles[:, mask]
            .mean(dim=1)
            .numpy()
        )

        result.append(
            class_mean
        )

    # [neurons, classes]
    return np.stack(
        result,
        axis=1,
    )


# ================================================================
# Hungarian matching
# ================================================================

def hungarian_matching(corr):

    # Work only with neurons having meaningful correlations.
    valid_a = np.where(
        ~np.all(np.isnan(corr), axis=1)
    )[0]

    valid_b = np.where(
        ~np.all(np.isnan(corr), axis=0)
    )[0]

    valid_corr = corr[
        np.ix_(
            valid_a,
            valid_b,
        )
    ]

    # NaN correlations should never be selected.
    similarity = np.nan_to_num(
        valid_corr,
        nan=-1.0,
    )

    # scipy solves a MINIMIZATION problem.
    #
    # maximizing correlation is equivalent to
    # minimizing negative correlation.
    row_ind, col_ind = linear_sum_assignment(
        -similarity
    )

    neurons_a = valid_a[
        row_ind
    ]

    neurons_b = valid_b[
        col_ind
    ]

    correlations = corr[
        neurons_a,
        neurons_b,
    ]

    return (
        neurons_a,
        neurons_b,
        correlations,
    )


# ================================================================
# Main
# ================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare functional response profiles "
            "of two neurogenesis networks."
        )
    )

    parser.add_argument(
        "model_a",
        type=str,
        help="Path to first best_model.pt",
    )

    parser.add_argument(
        "model_b",
        type=str,
        help="Path to second best_model.pt",
    )

    parser.add_argument(
        "--n-probe",
        type=int,
        default=1000,
        help="Number of common test examples.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="response_profile_comparison",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # ------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------

    model_a, checkpoint_a = load_model(
        args.model_a,
        device,
    )

    model_b, checkpoint_b = load_model(
        args.model_b,
        device,
    )

    dataset_a = checkpoint_a["dataset"]
    dataset_b = checkpoint_b["dataset"]

    if dataset_a.lower() != dataset_b.lower():

        raise ValueError(
            "The two models were trained on "
            "different datasets."
        )

    print()
    print(
        f"Network A: {checkpoint_a['hidden_neurons']} neurons"
    )

    print(
        f"Network B: {checkpoint_b['hidden_neurons']} neurons"
    )

    # ------------------------------------------------------------
    # Probe dataset
    # ------------------------------------------------------------

    loader = get_probe_loader(
        dataset_a,
        args.n_probe,
        args.batch_size,
    )

    # ------------------------------------------------------------
    # Extract functional response profiles
    # ------------------------------------------------------------

    profiles_a, labels_a = extract_response_profiles(
        model_a,
        loader,
        device,
    )

    profiles_b, labels_b = extract_response_profiles(
        model_b,
        loader,
        device,
    )

    if not torch.equal(
            labels_a,
            labels_b,
    ):
        raise RuntimeError(
            "Probe datasets differ."
        )

    print()
    print(
        "Response matrix A:",
        tuple(profiles_a.shape),
    )

    print(
        "Response matrix B:",
        tuple(profiles_b.shape),
    )

    # ------------------------------------------------------------
    # Full neuron-neuron correlation matrix
    # ------------------------------------------------------------

    corr, active_a, active_b = correlation_matrix(
        profiles_a,
        profiles_b,
    )

    # ------------------------------------------------------------
    # Class tuning
    # ------------------------------------------------------------

    class_a = class_profiles(
        profiles_a,
        labels_a,
    )

    class_b = class_profiles(
        profiles_b,
        labels_b,
    )

    # ------------------------------------------------------------
    # Hungarian matching
    # ------------------------------------------------------------

    neuron_a, neuron_b, matched_corr = (
        hungarian_matching(corr)
    )

    # ------------------------------------------------------------
    # Results directory
    # ------------------------------------------------------------

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Save complete numerical data
    # ------------------------------------------------------------

    torch.save(
        {
            "model_a": str(args.model_a),
            "model_b": str(args.model_b),

            "profiles_a": profiles_a,
            "profiles_b": profiles_b,

            "labels": labels_a,

            "correlation_matrix":
                torch.from_numpy(corr),

            "class_profiles_a":
                torch.from_numpy(class_a),

            "class_profiles_b":
                torch.from_numpy(class_b),

            "matched_neuron_a":
                torch.from_numpy(neuron_a),

            "matched_neuron_b":
                torch.from_numpy(neuron_b),

            "matched_correlations":
                torch.from_numpy(matched_corr),
        },
        output_dir / "response_profiles.pt",
        )

    # ------------------------------------------------------------
    # Save correlation matrix as CSV
    # ------------------------------------------------------------

    corr_df = pd.DataFrame(
        corr,
        index=[
            f"A_{i}"
            for i in range(corr.shape[0])
        ],
        columns=[
            f"B_{j}"
            for j in range(corr.shape[1])
        ],
    )

    corr_df.to_csv(
        output_dir / "correlation_matrix.csv"
    )

    # ------------------------------------------------------------
    # Save Hungarian matches
    # ------------------------------------------------------------

    matches = pd.DataFrame(
        {
            "neuron_a": neuron_a,
            "neuron_b": neuron_b,
            "correlation": matched_corr,
        }
    )

    matches = matches.sort_values(
        "correlation",
        ascending=False,
    )

    matches.to_csv(
        output_dir / "matched_neurons.csv",
        index=False,
        )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print()
    print("========================================")
    print("Functional comparison")
    print("========================================")

    print(
        f"Active neurons A : "
        f"{active_a.sum()} / {len(active_a)}"
    )

    print(
        f"Active neurons B : "
        f"{active_b.sum()} / {len(active_b)}"
    )

    print(
        f"Matched neurons  : "
        f"{len(matched_corr)}"
    )

    print(
        f"Mean correlation : "
        f"{np.mean(matched_corr):.4f}"
    )

    print(
        f"Median correlation: "
        f"{np.median(matched_corr):.4f}"
    )

    print(
        f"Min correlation  : "
        f"{np.min(matched_corr):.4f}"
    )

    print(
        f"Max correlation  : "
        f"{np.max(matched_corr):.4f}"
    )

    print()
    print(
        f"Saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()