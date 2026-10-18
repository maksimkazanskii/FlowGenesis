"""
Representative FlowGenesis architecture.

LEFT:
    Input -> hidden neurons grouped by functional similarity -> Output

RIGHT:
    Functional correlation matrix using exactly the same neuron ordering.

Representative model = global CKA medoid.

Usage:
python -m src_self_organization.utils.visualize_architecture \
    data/so/warmup_ablation/fashionmnist
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

from ..config import Config
from ..model_neurogenesis import NeuralContainer


# ============================================================
# CKA MEDOID
# ============================================================

def find_cka_medoid(root):

    root = Path(root).resolve()

    csv_path = None

    for directory in [root, *root.parents]:

        candidate = directory / "comparative_metrics_pairs.csv"

        if candidate.exists():
            csv_path = candidate
            break

    if csv_path is None:
        raise FileNotFoundError(
            "comparative_metrics_pairs.csv not found."
        )

    df = pd.read_csv(csv_path)

    csv_root = csv_path.parent

    def resolve_model(value):

        p = Path(str(value))

        if p.is_absolute():
            return p.resolve()

        p1 = (csv_root / p).resolve()

        if p1.exists():
            return p1

        return p.resolve()

    df["_a"] = df["model_a"].map(resolve_model)
    df["_b"] = df["model_b"].map(resolve_model)

    def below_root(path):

        try:
            path.relative_to(root)
            return True

        except ValueError:
            return False

    models = set()

    for path in df["_a"]:

        if below_root(path):
            models.add(path)

    for path in df["_b"]:

        if below_root(path):
            models.add(path)

    scores = {
        model: []
        for model in models
    }

    for _, row in df.iterrows():

        a = row["_a"]
        b = row["_b"]

        if a not in models or b not in models:
            continue

        cka = float(row["cka"])

        if not np.isfinite(cka):
            continue

        scores[a].append(cka)
        scores[b].append(cka)

    ranking = []

    for model, values in scores.items():

        if values:

            ranking.append(
                (
                    model,
                    float(np.mean(values)),
                )
            )

    if not ranking:
        raise RuntimeError(
            "No CKA comparisons found."
        )

    ranking.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    model, score = ranking[0]

    print()
    print("Representative model")
    print("--------------------")
    print(model)
    print(f"Mean CKA: {score:.4f}")
    print()

    return model


# ============================================================
# MODEL
# ============================================================

def load_model(path, device):

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    cfg = Config()

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


# ============================================================
# PROBE DATA
# ============================================================

def get_probe_loader(
        dataset_name,
        n_probe,
        batch_size,
        seed,
):

    name = dataset_name.lower()

    if name == "fashionmnist":

        dataset_class = datasets.FashionMNIST

    elif name == "mnist":

        dataset_class = datasets.MNIST

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

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(dataset),
        size=n_probe,
        replace=False,
    )

    return DataLoader(
        Subset(
            dataset,
            indices.tolist(),
        ),
        batch_size=batch_size,
        shuffle=False,
    )


# ============================================================
# HIDDEN RESPONSES
# ============================================================

@torch.no_grad()
def get_hidden_responses(
        model,
        loader,
        device,
):

    responses = []

    for images, _ in loader:

        images = images.to(device)

        images = images.view(
            images.size(0),
            -1,
        )

        hidden = model.relax(images)

        responses.append(
            hidden.cpu()
        )

    return torch.cat(
        responses,
        dim=0,
    ).numpy()


# ============================================================
# FUNCTIONAL CORRELATION
# ============================================================

def functional_correlation(H):

    # samples x neurons -> neurons x samples

    X = H.T

    X = X - X.mean(
        axis=1,
        keepdims=True,
    )

    norms = np.linalg.norm(
        X,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    X = X / norms

    similarity = X @ X.T

    similarity = np.clip(
        similarity,
        -1,
        1,
    )

    np.fill_diagonal(
        similarity,
        1.0,
    )

    return similarity


# ============================================================
# FUNCTIONAL CLUSTERING
# ============================================================

def cluster_neurons(
        similarity,
        n_clusters,
):

    # Correlation -> distance
    #
    # identical: 0
    # uncorrelated: 1
    # anticorrelated: 2

    distance = 1.0 - similarity

    distance = np.clip(
        distance,
        0,
        2,
    )

    distance = (
                       distance
                       + distance.T
               ) / 2

    np.fill_diagonal(
        distance,
        0,
    )

    condensed = squareform(
        distance,
        checks=False,
    )

    Z = linkage(
        condensed,
        method="average",
    )

    order = leaves_list(Z)

    cluster_labels = fcluster(
        Z,
        t=n_clusters,
        criterion="maxclust",
    )

    return (
        order,
        cluster_labels,
    )


# ============================================================
# RECURRENT WEIGHTS
# ============================================================

def recurrent_weights(model):

    W = (
        model.W
        .detach()
        .cpu()
        .numpy()
    )

    mask = (
        model.mask
        .detach()
        .cpu()
        .numpy()
    )

    degree = np.maximum(
        mask.sum(
            axis=1,
            keepdims=True,
        ),
        1.0,
    )

    W_eff = (
                    W * mask
            ) / degree

    return (
        W_eff,
        mask,
    )


# ============================================================
# HIDDEN POSITIONS
# ============================================================

def hidden_positions(
        order,
        cluster_labels,
):

    """
    All hidden neurons lie between input and output.

    Their vertical order follows hierarchical functional
    clustering. Small gaps separate functional clusters.
    """

    positions = {}

    y = 0.0

    previous_cluster = None

    for neuron in order:

        cluster = cluster_labels[
            neuron
        ]

        if (
                previous_cluster is not None
                and cluster != previous_cluster
        ):
            y += 1.0

        positions[int(neuron)] = (
            0.0,
            -y,
        )

        y += 1.0

        previous_cluster = cluster

    return positions


# ============================================================
# SCALE
# ============================================================

def scale(
        values,
        minimum,
        maximum,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        return values

    if np.allclose(
            values.min(),
            values.max(),
    ):

        return np.full(
            len(values),
            (minimum + maximum) / 2,
            )

    x = (
                values - values.min()
        ) / (
                values.max()
                - values.min()
        )

    return (
            minimum
            + x
            * (maximum - minimum)
    )


# ============================================================
# PLOT
# ============================================================
def plot(
        model,
        H,
        similarity,
        order,
        cluster_labels,
        output,
        edge_quantile,
        dpi,
):
    """
    FlowGenesis representative architecture.

    Hidden neurons:
        color  = functional cluster
        size   = mean absolute activation
        number = true incoming recurrent degree

    Ordering:
        functional cluster -> decreasing activity

    Connections:
        input -> hidden  = aggregate trained input strength
        recurrent       = top-2 strongest outgoing edges per neuron
        hidden -> output = strongest trained output connections

    All connections are gray.
    Nodes are drawn above connections.
    """

    # ========================================================
    # WEIGHTS
    # ========================================================

    W_eff, mask = recurrent_weights(model)

    W_in = (
        model.input_layer.weight
        .detach()
        .cpu()
        .numpy()
    )

    W_out = (
        model.output_layer.weight
        .detach()
        .cpu()
        .numpy()
    )

    n_hidden = W_in.shape[0]
    n_input = W_in.shape[1]
    n_output = W_out.shape[0]

    # ========================================================
    # ACTIVITY
    # ========================================================

    activity = np.abs(H).mean(axis=0)

    # ========================================================
    # TRUE INCOMING RECURRENT DEGREE
    #
    # mask[target, source]
    # row sum = incoming degree
    # ========================================================

    incoming_degree = (
            mask != 0
    ).sum(axis=1).astype(int)

    # ========================================================
    # FUNCTIONAL CLUSTERS
    # ========================================================

    unique_clusters = []

    for neuron in order:
        c = int(cluster_labels[neuron])

        if c not in unique_clusters:
            unique_clusters.append(c)

    # ========================================================
    # SORT NEURONS WITHIN EACH CLUSTER BY ACTIVITY
    # ========================================================

    cluster_neurons = {}

    for c in unique_clusters:

        neurons = [
            int(n)
            for n in order
            if int(cluster_labels[n]) == c
        ]

        neurons.sort(
            key=lambda n: activity[n],
            reverse=True,
        )

        cluster_neurons[c] = neurons

    clustered_order = np.array(
        [
            neuron
            for c in unique_clusters
            for neuron in cluster_neurons[c]
        ],
        dtype=int,
    )

    n_clusters = len(unique_clusters)

    # ========================================================
    # CLUSTER COLORS
    # ========================================================

    cluster_colors = [
        "red",
        "steelblue",
        "orange",
        "gray",
        "dimgray",
        "darkslateblue",
        "firebrick",
        "slategray",
    ]

    cluster_color = {
        c: cluster_colors[i % len(cluster_colors)]
        for i, c in enumerate(unique_clusters)
    }

    # ========================================================
    # FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        gridspec_kw={
            "width_ratios": [1.35, 1.0]
        },
    )

    # ========================================================
    # LEFT PANEL
    # ========================================================

    ax = axes[0]

    input_x = -4.3

    hidden_x_min = -1.9
    hidden_x_max = 1.9

    output_x = 4.3

    # ========================================================
    # HIDDEN POSITIONS
    #
    # Cluster determines vertical position.
    # Activity determines left-to-right order.
    # ========================================================

    cluster_centers = np.linspace(
        1.10,
        -1.10,
        n_clusters,
    )

    hidden_positions = {}

    for cluster_index, c in enumerate(unique_clusters):

        neurons = cluster_neurons[c]

        cy = cluster_centers[cluster_index]

        m = len(neurons)

        if m == 1:

            hidden_positions[neurons[0]] = (
                0.0,
                cy,
            )

        else:

            xs = np.linspace(
                hidden_x_min,
                hidden_x_max,
                m,
            )

            # Very small vertical displacement prevents
            # perfectly overlapping horizontal lines.
            phase = np.linspace(
                -np.pi,
                np.pi,
                m,
            )

            ys = (
                    cy
                    + 0.035 * np.cos(phase)
            )

            for neuron, x, y in zip(
                    neurons,
                    xs,
                    ys,
            ):
                hidden_positions[neuron] = (
                    float(x),
                    float(y),
                )

    # ========================================================
    # INPUT BLOCK
    # ========================================================

    input_pos = (
        input_x,
        0.0,
    )

    ax.text(
        input_pos[0],
        input_pos[1],
        f"Input\n({n_input})",
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="black",
        ),
        zorder=20,
    )

    # ========================================================
    # OUTPUT POSITIONS
    # ========================================================

    output_y = np.linspace(
        0.90,
        -0.90,
        n_output,
    )

    output_positions = {
        j: (
            output_x,
            output_y[j],
        )
        for j in range(n_output)
    }

    # ========================================================
    # INPUT -> HIDDEN
    #
    # Each line represents aggregate input strength:
    #
    # ||W_in[i,:]||_2
    #
    # These are deliberately very faint.
    # ========================================================

    input_strength = np.linalg.norm(
        W_in,
        axis=1,
    )

    input_min = input_strength.min()
    input_max = input_strength.max()

    input_span = max(
        input_max - input_min,
        1e-12,
        )

    for neuron in range(n_hidden):

        if neuron not in hidden_positions:
            continue

        x2, y2 = hidden_positions[neuron]

        strength = (
                           input_strength[neuron]
                           - input_min
                   ) / input_span

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=input_pos,
            arrowprops=dict(
                arrowstyle="-|>",
                color="0.75",
                alpha=(
                        0.035
                        + 0.055 * strength
                ),
                linewidth=(
                        0.30
                        + 0.40 * strength
                ),
                mutation_scale=5,
                shrinkA=30,
                shrinkB=10,
            ),
            zorder=1,
        )

    # ========================================================
    # HIDDEN -> OUTPUT
    #
    # Keep strongest 10% of trained output weights.
    # ========================================================

    output_connections = []

    for out_idx in range(n_output):

        for neuron in range(n_hidden):

            magnitude = abs(
                float(
                    W_out[
                        out_idx,
                        neuron,
                    ]
                )
            )

            if magnitude <= 0:
                continue

            output_connections.append(
                (
                    neuron,
                    out_idx,
                    magnitude,
                )
            )

    if output_connections:

        output_magnitudes = np.array(
            [
                x[2]
                for x in output_connections
            ]
        )

        output_threshold = np.quantile(
            output_magnitudes,
            0.90,
        )

        output_connections = [
            x
            for x in output_connections
            if x[2] >= output_threshold
        ]

        retained_output = np.array(
            [
                x[2]
                for x in output_connections
            ]
        )

        o_min = retained_output.min()
        o_max = retained_output.max()

        o_span = max(
            o_max - o_min,
            1e-12,
            )

        for (
                neuron,
                out_idx,
                magnitude,
        ) in output_connections:

            if neuron not in hidden_positions:
                continue

            x1, y1 = hidden_positions[neuron]

            x2, y2 = output_positions[out_idx]

            strength = (
                               magnitude - o_min
                       ) / o_span

            linewidth = (
                    0.65
                    + 0.45 * strength
            )

            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="0.55",
                    alpha=0.30,
                    linewidth=linewidth,
                    mutation_scale=7,
                    shrinkA=10,
                    shrinkB=7,
                ),
                zorder=2,
            )

    # ========================================================
    # RECURRENT CONNECTIONS
    #
    # W_eff[target, source]
    #
    # For every source neuron:
    #     retain its top-2 strongest outgoing edges.
    # ========================================================

    top_k_per_neuron = 2

    recurrent_connections = []

    for source in range(n_hidden):

        outgoing = []

        for target in range(n_hidden):

            if source == target:
                continue

            if mask[target, source] == 0:
                continue

            magnitude = abs(
                float(
                    W_eff[
                        target,
                        source,
                    ]
                )
            )

            if magnitude <= 0:
                continue

            outgoing.append(
                (
                    source,
                    target,
                    magnitude,
                )
            )

        outgoing.sort(
            key=lambda x: x[2],
            reverse=True,
        )

        recurrent_connections.extend(
            outgoing[:top_k_per_neuron]
        )

    # ========================================================
    # DRAW RECURRENT CONNECTIONS
    # ========================================================

    if recurrent_connections:

        magnitudes = np.array(
            [
                x[2]
                for x in recurrent_connections
            ]
        )

        r_min = magnitudes.min()
        r_max = magnitudes.max()

        r_span = max(
            r_max - r_min,
            1e-12,
            )

        for (
                source,
                target,
                magnitude,
        ) in recurrent_connections:

            if (
                    source not in hidden_positions
                    or target not in hidden_positions
            ):
                continue

            x1, y1 = hidden_positions[source]
            x2, y2 = hidden_positions[target]

            strength = (
                               magnitude - r_min
                       ) / r_span

            # Deliberately narrow thickness range.
            linewidth = (
                    0.80
                    + 0.55 * strength
            )

            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="0.35",
                    alpha=0.40,
                    linewidth=linewidth,
                    mutation_scale=7,
                    shrinkA=11,
                    shrinkB=11,
                    connectionstyle="arc3,rad=0.035",
                ),
                zorder=3,
            )

    # ========================================================
    # OUTPUT NODES
    #
    # Draw after edges.
    # ========================================================

    ax.scatter(
        np.full(
            n_output,
            output_x,
        ),
        output_y,
        s=75,
        color="lightgray",
        edgecolors="black",
        linewidths=0.8,
        zorder=10,
    )

    ax.text(
        output_x,
        1.10,
        f"Output ({n_output})",
        ha="center",
        va="bottom",
        fontsize=10,
        zorder=11,
    )

    # ========================================================
    # HIDDEN NODE SIZE
    #
    # Size = activity
    #
    # Larger than previous version so degree labels
    # are clearly readable.
    # ========================================================

    activity_scaled = np.sqrt(
        np.maximum(
            activity,
            0,
        )
    )

    hidden_sizes = scale(
        activity_scaled,
        120,   # minimum
        380,   # maximum
    )

    # ========================================================
    # HIDDEN NODES
    #
    # Color  = functional cluster
    # Size   = activity
    # Number = true incoming recurrent degree
    #
    # IMPORTANT:
    # Nodes are drawn AFTER edges.
    # ========================================================

    for c in unique_clusters:

        neurons = cluster_neurons[c]

        xs = [
            hidden_positions[n][0]
            for n in neurons
        ]

        ys = [
            hidden_positions[n][1]
            for n in neurons
        ]

        sizes = [
            hidden_sizes[n]
            for n in neurons
        ]

        # --------------------------------------------
        # Draw neurons
        # --------------------------------------------

        ax.scatter(
            xs,
            ys,
            s=sizes,
            color=cluster_color[c],
            alpha=0.72,
            edgecolors="black",
            linewidths=0.9,
            zorder=10,
        )

        # --------------------------------------------
        # Incoming recurrent degree
        # --------------------------------------------

        for neuron, x, y in zip(
                neurons,
                xs,
                ys,
        ):

            ax.text(
                x,
                y,
                str(
                    incoming_degree[
                        neuron
                    ]
                ),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
                fontweight="normal",
                zorder=11,
            )

    # ========================================================
    # LEFT PANEL FORMATTING
    # ========================================================

    ax.set_xlim(
        -5.0,
        5.0,
    )

    ax.set_ylim(
        -1.55,
        1.55,
    )

    ax.set_title(
        "Trained architecture"
    )

    ax.set_axis_off()

    # ========================================================
    # RIGHT PANEL
    #
    # EXACT SAME ORDER:
    #
    # cluster -> decreasing activity
    # ========================================================

    ax = axes[1]

    reordered = similarity[
        np.ix_(
            clustered_order,
            clustered_order,
        )
    ]

    im = ax.imshow(
        reordered,
        vmin=-1,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
    )

    # ========================================================
    # CLUSTER BOUNDARIES
    # ========================================================

    cumulative = 0

    for c in unique_clusters[:-1]:

        cumulative += len(
            cluster_neurons[c]
        )

        boundary = cumulative - 0.5

        ax.axhline(
            boundary,
            color="0.7",
            linewidth=0.8,
            linestyle="--",
            alpha=0.8,
        )

        ax.axvline(
            boundary,
            color="0.7",
            linewidth=0.8,
            linestyle="--",
            alpha=0.8,
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
        fraction=0.046,
        pad=0.04,
        label="Response correlation",
    )

    # ========================================================
    # FINAL
    # ========================================================

    fig.suptitle(
        "FlowGenesis representative architecture"
    )

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "root",
        help="Experiment root, e.g. data/so/warmup_ablation/fashionmnist",
    )

    parser.add_argument(
        "--output",
        default="flowgenesis_architecture.png",
    )

    parser.add_argument(
        "--n-probe",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--probe-seed",
        type=int,
        default=20260829,
    )

    parser.add_argument(
        "--edge-quantile",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
    )

    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")

    elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")

    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Representative model = global CKA medoid
    # --------------------------------------------------------

    model_path = find_cka_medoid(
        args.root
    )

    model, checkpoint = load_model(
        model_path,
        device,
    )

    # --------------------------------------------------------
    # Fixed probe set
    # --------------------------------------------------------

    loader = get_probe_loader(
        checkpoint["dataset"],
        args.n_probe,
        args.batch_size,
        args.probe_seed,
    )

    H = get_hidden_responses(
        model,
        loader,
        device,
    )

    # --------------------------------------------------------
    # Functional architecture
    # --------------------------------------------------------

    similarity = functional_correlation(
        H
    )

    order, cluster_labels = cluster_neurons(
        similarity,
        args.clusters,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot(
        model,
        H,
        similarity,
        order,
        cluster_labels,
        output,
        args.edge_quantile,
        args.dpi,
    )

    print(f"Final neurons: {H.shape[1]}")
    print(f"Functional clusters: {args.clusters}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()