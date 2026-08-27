"""
Independent functional-architecture analysis of one neurogenesis network.

Each hidden neuron is represented by its response profile across a fixed
probe dataset:

    R_i = [h_i(x_1), ..., h_i(x_M)]

The functional architecture is the within-network Pearson-correlation
matrix S_ij = corr(R_i, R_j).

The script analyzes ONE checkpoint at a time. It does not perform
cross-network matching or comparison.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

from ..config import Config
from ..model_neurogenesis import NeuralContainer


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

    # Reconstruct the architecture exactly as saved.
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

    # Deterministic fixed probe set.
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

        images = images.view(images.size(0), -1)
        hidden = model.relax(images)

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

    # [neurons, samples]
    profiles = H.T.contiguous()

    return profiles, labels


# ================================================================
# Functional architecture matrix
# ================================================================

def functional_architecture(profiles):

    X = profiles.numpy().astype(
        np.float64
    )

    # Center each neuron's response profile.
    X = X - X.mean(
        axis=1,
        keepdims=True,
    )

    norms = np.linalg.norm(
        X,
        axis=1,
        keepdims=True,
    )

    active = (
            norms[:, 0] > 1e-12
    )

    X = X / np.maximum(
        norms,
        1e-12,
    )

    # S_ij = Pearson correlation between neuron i and neuron j.
    similarity = X @ X.T

    similarity[
        ~active,
        :
    ] = np.nan

    similarity[
        :,
        ~active
    ] = np.nan

    # Numerical roundoff can produce tiny excursions outside [-1, 1].
    similarity = np.clip(
        similarity,
        -1.0,
        1.0,
    )

    return similarity, active


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
    return (
        np.stack(result, axis=1),
        classes,
    )


# ================================================================
# Per-neuron properties
# ================================================================

def neuron_properties(
        profiles,
        labels,
        similarity,
        active,
):

    profiles_np = profiles.numpy()

    class_mean, classes = class_profiles(
        profiles,
        labels,
    )

    # Preferred class = class with largest mean activation.
    preferred_idx = np.argmax(
        class_mean,
        axis=1,
    )

    preferred_class = classes[
        preferred_idx
    ]

    mean_activation = profiles_np.mean(
        axis=1
    )

    activation_std = profiles_np.std(
        axis=1
    )

    max_class_response = class_mean[
        np.arange(class_mean.shape[0]),
        preferred_idx,
    ]

    # Difference between strongest and second-strongest class response.
    if class_mean.shape[1] >= 2:
        sorted_class = np.sort(
            class_mean,
            axis=1,
        )
        class_selectivity = (
                sorted_class[:, -1]
                - sorted_class[:, -2]
        )
    else:
        class_selectivity = np.zeros(
            class_mean.shape[0]
        )

    # Mean positive functional similarity to other active neurons.
    mean_positive_similarity = np.full(
        len(active),
        np.nan,
        dtype=np.float64,
    )

    max_similarity = np.full(
        len(active),
        np.nan,
        dtype=np.float64,
    )

    for i in range(len(active)):

        if not active[i]:
            continue

        row = similarity[i].copy()
        row[i] = np.nan

        finite = row[np.isfinite(row)]

        if len(finite) == 0:
            continue

        positive = finite[
            finite > 0
            ]

        if len(positive) > 0:
            mean_positive_similarity[i] = (
                positive.mean()
            )

        max_similarity[i] = finite.max()

    table = pd.DataFrame(
        {
            "neuron": np.arange(
                profiles.shape[0]
            ),
            "active": active,
            "mean_activation": mean_activation,
            "activation_std": activation_std,
            "preferred_class": preferred_class,
            "preferred_class_response":
                max_class_response,
            "class_selectivity":
                class_selectivity,
            "mean_positive_similarity":
                mean_positive_similarity,
            "max_similarity_to_other":
                max_similarity,
        }
    )

    return table, class_mean, classes


# ================================================================
# Global architecture properties
# ================================================================

def architecture_metrics(
        similarity,
        active,
):

    n_total = len(active)
    n_active = int(active.sum())

    if n_active < 2:
        return {
            "neurons": n_total,
            "active_neurons": n_active,
            "inactive_neurons": n_total - n_active,
            "mean_pairwise_correlation": np.nan,
            "mean_absolute_correlation": np.nan,
            "mean_positive_correlation": np.nan,
            "fraction_positive_pairs": np.nan,
            "fraction_negative_pairs": np.nan,
        }

    active_idx = np.where(
        active
    )[0]

    S = similarity[
        np.ix_(
            active_idx,
            active_idx,
        )
    ]

    ii, jj = np.triu_indices(
        n_active,
        k=1,
    )

    values = S[
        ii,
        jj,
    ]

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        mean_corr = np.nan
        mean_abs = np.nan
        mean_positive = np.nan
        fraction_positive = np.nan
        fraction_negative = np.nan
    else:
        mean_corr = float(
            values.mean()
        )

        mean_abs = float(
            np.abs(values).mean()
        )

        positive = values[
            values > 0
            ]

        mean_positive = (
            float(positive.mean())
            if len(positive) > 0
            else np.nan
        )

        fraction_positive = float(
            np.mean(values > 0)
        )

        fraction_negative = float(
            np.mean(values < 0)
        )

    return {
        "neurons": n_total,
        "active_neurons": n_active,
        "inactive_neurons": n_total - n_active,
        "mean_pairwise_correlation": mean_corr,
        "mean_absolute_correlation": mean_abs,
        "mean_positive_correlation": mean_positive,
        "fraction_positive_pairs": fraction_positive,
        "fraction_negative_pairs": fraction_negative,
    }


# ================================================================
# Visualizations
# ================================================================

def plot_architecture_matrix(
        similarity,
        output_path,
):

    fig, ax = plt.subplots(
        figsize=(7, 6),
    )

    im = ax.imshow(
        similarity,
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
    )

    ax.set_title(
        "Functional architecture"
    )

    ax.set_xlabel(
        "Neuron"
    )

    ax.set_ylabel(
        "Neuron"
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Response-profile correlation",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_class_profiles(
        class_mean,
        classes,
        output_path,
):

    fig, ax = plt.subplots(
        figsize=(8, 6),
    )

    im = ax.imshow(
        class_mean,
        aspect="auto",
    )

    ax.set_title(
        "Neuron class-response profiles"
    )

    ax.set_xlabel(
        "Class"
    )

    ax.set_ylabel(
        "Neuron"
    )

    ax.set_xticks(
        np.arange(len(classes))
    )

    ax.set_xticklabels(
        classes
    )

    fig.colorbar(
        im,
        ax=ax,
        label="Mean hidden activation",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ================================================================
# Main
# ================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze the functional architecture "
            "of one neurogenesis network."
        )
    )

    parser.add_argument(
        "model",
        type=str,
        help="Path to best_model.pt or final_model.pt",
    )

    parser.add_argument(
        "--n-probe",
        type=int,
        default=1000,
        help="Number of test examples used as probes.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="architecture_analysis",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
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

    print(
        f"Device: {device}"
    )

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    model, checkpoint = load_model(
        args.model,
        device,
    )

    dataset_name = checkpoint[
        "dataset"
    ]

    print(
        f"Dataset: {dataset_name}"
    )

    print(
        f"Neurons: {checkpoint['hidden_neurons']}"
    )

    # ------------------------------------------------------------
    # Probe responses
    # ------------------------------------------------------------

    loader = get_probe_loader(
        dataset_name,
        args.n_probe,
        args.batch_size,
    )

    profiles, labels = (
        extract_response_profiles(
            model,
            loader,
            device,
        )
    )

    print(
        "Response matrix:",
        tuple(profiles.shape),
    )

    # ------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------

    similarity, active = (
        functional_architecture(
            profiles
        )
    )

    neuron_table, class_mean, classes = (
        neuron_properties(
            profiles,
            labels,
            similarity,
            active,
        )
    )

    metrics = architecture_metrics(
        similarity,
        active,
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model": str(args.model),
            "dataset": dataset_name,
            "profiles": profiles,
            "labels": labels,
            "similarity_matrix":
                torch.from_numpy(similarity),
            "class_profiles":
                torch.from_numpy(class_mean),
            "classes":
                torch.from_numpy(classes),
            "active_neurons":
                torch.from_numpy(active),
            "metrics": metrics,
        },
        output_dir / "architecture.pt",
        )

    pd.DataFrame(
        similarity,
        index=[
            f"N_{i}"
            for i in range(
                similarity.shape[0]
            )
        ],
        columns=[
            f"N_{i}"
            for i in range(
                similarity.shape[1]
            )
        ],
    ).to_csv(
        output_dir
        / "similarity_matrix.csv"
    )

    pd.DataFrame(
        class_mean,
        index=[
            f"N_{i}"
            for i in range(
                class_mean.shape[0]
            )
        ],
        columns=[
            f"class_{c}"
            for c in classes
        ],
    ).to_csv(
        output_dir
        / "class_profiles.csv"
    )

    neuron_table.to_csv(
        output_dir
        / "neuron_properties.csv",
        index=False,
        )

    pd.DataFrame(
        [metrics]
    ).to_csv(
        output_dir
        / "architecture_metrics.csv",
        index=False,
        )

    if not args.no_plot:

        plot_architecture_matrix(
            similarity,
            output_dir
            / "architecture_matrix.png",
            )

        plot_class_profiles(
            class_mean,
            classes,
            output_dir
            / "class_profiles.png",
            )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print()
    print(
        "========================================"
    )
    print(
        "Functional architecture"
    )
    print(
        "========================================"
    )

    for key, value in metrics.items():

        if isinstance(
                value,
                (float, np.floating),
        ):
            print(
                f"{key:28s}: {value:.4f}"
            )
        else:
            print(
                f"{key:28s}: {value}"
            )

    print()
    print(
        f"Saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()