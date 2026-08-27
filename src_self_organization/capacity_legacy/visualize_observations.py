"""
visualize_observations.py

Load saved neural-container observation statistics and generate figures.

Expected statistics file keys:
    flux        : Tensor [epochs, neurons]
    capacity    : Tensor [epochs, neurons]
    activation  : Tensor [epochs, neurons]

Optional keys:
    train_loss
    train_acc
    test_loss
    test_acc
    dataset
    epochs
    hidden_neurons

Default output directory:
    data/so/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


import networkx as nx

def save_capacity_backbone(
        capacity,
        weights,
        output_path,
        top_fraction=0.20,
):
    """
    Draw the computational backbone.

    Nodes
    -----
    size  ~ capacity
    color ~ capacity

    Edges
    -----
    strongest |W| * Ci * Cj
    """

    final_capacity = capacity[-1]

    W = weights.copy()

    n = len(final_capacity)

    scores = []

    # --------------------------------------------------
    # Score edges
    # --------------------------------------------------

    for i in range(n):

        for j in range(i + 1, n):

            if abs(W[i, j]) < 1e-8:
                continue

            score = (
                    abs(W[i, j])
                    * final_capacity[i]
                    * final_capacity[j]
            )

            scores.append((score, i, j))

    scores.sort(reverse=True)

    keep = max(
        1,
        int(len(scores) * top_fraction),
    )

    # --------------------------------------------------
    # Build graph
    # --------------------------------------------------

    G = nx.Graph()

    for i in range(n):

        G.add_node(
            i,
            capacity=final_capacity[i],
        )

    for _, i, j in scores[:keep]:

        G.add_edge(
            i,
            j,
            weight=abs(W[i, j]),
        )

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------

    # --------------------------------------------------
    # Fixed layout from the FULL graph
    # --------------------------------------------------

    full_graph = nx.Graph()

    for i in range(n):
        full_graph.add_node(i)

    for i in range(n):
        for j in range(i + 1, n):

            if abs(W[i, j]) > 1e-8:

                full_graph.add_edge(
                    i,
                    j,
                )

    full_pos = nx.spring_layout(
            full_graph,
            seed=0,
            k=0.7,
            iterations=400,
    )

    # Use the same coordinates
    pos = {
            node: full_pos[node]
            for node in G.nodes()
    }

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    edge_widths = [
        0.5 + 8 * G[u][v]["weight"]
        for u, v in G.edges()
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        alpha=0.35,
        width=edge_widths,
        edge_color="gray",
    )

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=300 + 2500 * final_capacity,
        node_color=final_capacity,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.4,
    )

    # --------------------------------------------------
    # Label highest-capacity neurons
    # --------------------------------------------------

    top_nodes = np.argsort(final_capacity)[-10:]

    labels = {
        i: str(i)
        for i in top_nodes
    }

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=9,
        ax=ax,
    )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    ax.set_title(
        "Capacity Backbone"
    )

    ax.axis("off")

    cbar = fig.colorbar(
        nodes,
        ax=ax,
        shrink=0.8,
    )

    cbar.set_label(
        "Capacity"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)
def save_computational_core(
        capacity,
        weights,
        output_path,
        top_k=25,
        edge_percentile=75,
):
    """
    Computational core.

    Nodes:
        Top-K capacity neurons.

    Edges:
        Strongest recurrent connections
        between those neurons.
    """

    final_capacity = capacity[-1]

    W = np.abs(weights.copy())

    # --------------------------------------------------
    # Top neurons
    # --------------------------------------------------

    top_nodes = np.argsort(final_capacity)[-top_k:]

    # --------------------------------------------------
    # Determine edge threshold
    # --------------------------------------------------

    subW = W[np.ix_(top_nodes, top_nodes)]

    edge_values = subW[subW > 0]

    if len(edge_values) == 0:
        return

    threshold = np.percentile(
        edge_values,
        edge_percentile,
    )

    # --------------------------------------------------
    # Build graph
    # --------------------------------------------------

    G = nx.Graph()

    for node in top_nodes:

        G.add_node(
            int(node),
            capacity=float(final_capacity[node]),
        )

    for i in top_nodes:

        for j in top_nodes:

            if j <= i:
                continue

            if W[i, j] < threshold:
                continue

            G.add_edge(
                int(i),
                int(j),
                weight=float(W[i, j]),
            )

    # --------------------------------------------------
    # Largest connected component
    # --------------------------------------------------

    if nx.number_connected_components(G) > 1:

        largest = max(
            nx.connected_components(G),
            key=len,
        )

        G = G.subgraph(largest).copy()

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------

    pos = nx.spring_layout(
        G,
        seed=0,
        k=0.8,
        iterations=300,
    )

    capacities = np.array(
        [
            G.nodes[n]["capacity"]
            for n in G.nodes()
        ]
    )

    node_sizes = (
            300
            + 7000 * capacities**1.5
    )

    edge_widths = [
        1
        + 8 * G[u][v]["weight"]
        for u, v in G.edges()
    ]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 9)
    )

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        width=edge_widths,
        edge_color="gray",
        alpha=0.55,
    )

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=capacities,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.8,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_size=10,
        font_weight="bold",
    )

    ax.set_title(
        f"Computational Core ({len(G.nodes())} neurons)"
    )

    ax.axis("off")

    cbar = fig.colorbar(
        nodes,
        ax=ax,
        shrink=0.8,
    )

    cbar.set_label("Capacity")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Flux, Capacity, and activation observations."
    )
    parser.add_argument("--stats", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/so")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def to_numpy(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    if array.ndim != 2:
        raise ValueError(
            f"Expected '{name}' with shape [epochs, neurons], got {array.shape}."
        )

    return array.astype(np.float64, copy=False)


def load_statistics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Statistics file not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(data, dict):
        raise TypeError("Statistics file must contain a dictionary.")

    required = {"flux", "capacity", "activation"}
    missing = required.difference(data.keys())
    if missing:
        raise KeyError(f"Missing keys: {sorted(missing)}")

    return data


def save_heatmap(values, title, colorbar_label, output_path, dpi):
    figure, axis = plt.subplots(figsize=(11, 6))
    image = axis.imshow(
        values.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
    )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Neuron")
    epochs = values.shape[0]
    axis.set_xticks(np.arange(epochs))
    axis.set_xticklabels(np.arange(1, epochs + 1))
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_epoch_curve(values, title, ylabel, output_path, dpi):
    epochs = np.arange(1, values.shape[0] + 1)
    mean_values = values.mean(axis=1)
    std_values = values.std(axis=1)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, mean_values, marker="o", label="Mean")
    axis.fill_between(
        epochs,
        mean_values - std_values,
        mean_values + std_values,
        alpha=0.2,
        label="±1 neuron std",
    )
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_xticks(epochs)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_final_histogram(values, title, xlabel, output_path, dpi):
    final_values = values[-1]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(
        final_values,
        bins=min(20, max(5, len(final_values) // 5)),
        edgecolor="black",
    )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Number of neurons")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_top_capacity(capacity, top_k, output_path, dpi):
    final_capacity = capacity[-1]
    top_k = min(top_k, final_capacity.shape[0])
    indices = np.argsort(final_capacity)[-top_k:][::-1]
    values = final_capacity[indices]

    figure, axis = plt.subplots(figsize=(10, 6))
    positions = np.arange(top_k)
    axis.bar(positions, values)
    axis.set_title(f"Top {top_k} Neurons by Final Capacity")
    axis.set_xlabel("Neuron")
    axis.set_ylabel("Final capacity")
    axis.set_xticks(positions)
    axis.set_xticklabels(indices, rotation=45, ha="right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_top_capacity_trajectories(capacity, top_k, output_path, dpi):
    final_capacity = capacity[-1]
    top_k = min(top_k, final_capacity.shape[0])
    indices = np.argsort(final_capacity)[-top_k:][::-1]
    epochs = np.arange(1, capacity.shape[0] + 1)

    figure, axis = plt.subplots(figsize=(10, 6))
    for neuron_index in indices:
        axis.plot(
            epochs,
            capacity[:, neuron_index],
            label=f"Neuron {neuron_index}",
        )
    axis.set_title(f"Capacity Trajectories of Top {top_k} Neurons")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Capacity")
    axis.set_xticks(epochs)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_flux_capacity_scatter(flux, capacity, output_path, dpi):
    final_flux = flux[-1]
    final_capacity = capacity[-1]
    correlation = np.corrcoef(final_flux, final_capacity)[0, 1]

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(final_flux, final_capacity, alpha=0.75)
    axis.set_title(
        f"Final Flux vs Final Capacity\nPearson r = {correlation:.3f}"
    )
    axis.set_xlabel("Final flux")
    axis.set_ylabel("Final capacity")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_learning_curves(data, output_path, dpi):
    required = {"train_loss", "test_loss", "train_acc", "test_acc"}
    if not required.issubset(data.keys()):
        return

    train_loss = np.asarray(data["train_loss"], dtype=float)
    test_loss = np.asarray(data["test_loss"], dtype=float)
    train_acc = np.asarray(data["train_acc"], dtype=float)
    test_acc = np.asarray(data["test_acc"], dtype=float)
    epochs = np.arange(1, len(train_loss) + 1)

    figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(epochs, train_loss, marker="o", label="Train")
    axes[0].plot(epochs, test_loss, marker="o", label="Test")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Learning Curves")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="Train")
    axes[1].plot(epochs, test_acc, marker="o", label="Test")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xticks(epochs)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()

    stats_path = Path(args.stats)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_statistics(stats_path)

    flux = to_numpy(data["flux"], "flux")
    capacity = to_numpy(data["capacity"], "capacity")
    activation = to_numpy(data["activation"], "activation")
    weights = data["weights"].cpu().numpy()
    if flux.shape != capacity.shape or flux.shape != activation.shape:
        raise ValueError(
            "Flux, capacity, and activation must have identical shapes. "
            f"Got {flux.shape}, {capacity.shape}, {activation.shape}."
        )

    dataset = str(data.get("dataset", stats_path.stem))
    prefix = dataset.lower().replace(" ", "_")

    save_heatmap(
        flux,
        f"{dataset}: Neuron Flux",
        "Flux",
        output_dir / f"{prefix}_flux_heatmap.png",
        args.dpi,
    )
    save_heatmap(
        capacity,
        f"{dataset}: Neuron Capacity",
        "Capacity",
        output_dir / f"{prefix}_capacity_heatmap.png",
        args.dpi,
    )
    save_heatmap(
        activation,
        f"{dataset}: Mean Neuron Activation",
        "Mean activation",
        output_dir / f"{prefix}_activation_heatmap.png",
        args.dpi,
    )
    save_epoch_curve(
        flux,
        f"{dataset}: Mean Flux Across Neurons",
        "Flux",
        output_dir / f"{prefix}_mean_flux.png",
        args.dpi,
    )
    save_epoch_curve(
        capacity,
        f"{dataset}: Mean Capacity Across Neurons",
        "Capacity",
        output_dir / f"{prefix}_mean_capacity.png",
        args.dpi,
    )
    save_final_histogram(
        capacity,
        f"{dataset}: Final Capacity Distribution",
        "Final capacity",
        output_dir / f"{prefix}_capacity_histogram.png",
        args.dpi,
    )
    save_top_capacity(
        capacity,
        args.top_k,
        output_dir / f"{prefix}_top_capacity.png",
        args.dpi,
    )
    save_top_capacity_trajectories(
        capacity,
        min(args.top_k, 10),
        output_dir / f"{prefix}_top_capacity_trajectories.png",
        args.dpi,
    )
    save_flux_capacity_scatter(
        flux,
        capacity,
        output_dir / f"{prefix}_flux_capacity_scatter.png",
        args.dpi,
    )
    save_capacity_backbone(
        capacity,
        weights,
        output_dir / f"{prefix}_capacity_backbone.png",
        )
    save_learning_curves(
        data,
        output_dir / f"{prefix}_learning_curves.png",
        args.dpi,
    )
    save_computational_core(
        capacity,
        weights,
        output_dir / f"{prefix}_computational_core.png",
        )

    print(f"Loaded: {stats_path}")
    print(f"Observation shape: {flux.shape}")
    print(f"Saved figures to: {output_dir.resolve()}")

    for path in sorted(output_dir.glob(f"{prefix}_*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
