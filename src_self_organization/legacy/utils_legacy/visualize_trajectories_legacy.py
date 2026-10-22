from pathlib import Path
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt


# ================================================================
# Arguments
# ================================================================

parser = argparse.ArgumentParser(
    description=(
        "Plot mean ± std neuron-count trajectories "
        "for neurogenesis experiments."
    )
)

parser.add_argument(
    "folder",
    type=str,
    help="Folder containing neurons_*/seed_* experiment directories.",
)

parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Optional output image path.",
)

parser.add_argument(
    "--dpi",
    type=int,
    default=200,
)

args = parser.parse_args()

ROOT = Path(args.folder)

if not ROOT.exists():
    raise FileNotFoundError(
        f"Folder does not exist: {ROOT}"
    )


# ================================================================
# Collect trajectories
# ================================================================

runs_by_initial_n = {}

for stats_file in sorted(
        ROOT.glob("neurons_*/seed_*/statistics.pt")
):

    run_dir = stats_file.parent

    initial_n = int(
        run_dir.parent.name.split("_")[1]
    )

    seed = int(
        run_dir.name.split("_")[1]
    )

    stats = torch.load(
        stats_file,
        map_location="cpu",
        weights_only=False,
    )

    trajectory = stats.get("neuron_count")

    if trajectory is None:
        print(
            f"WARNING: neuron_count missing: {stats_file}"
        )
        continue

    # Convert possible tensors / numpy values to plain floats.
    trajectory = np.asarray(
        [
            float(x.item()) if torch.is_tensor(x) else float(x)
            for x in trajectory
        ],
        dtype=float,
    )

    if len(trajectory) == 0:
        print(
            f"WARNING: empty trajectory: {stats_file}"
        )
        continue

    runs_by_initial_n.setdefault(
        initial_n,
        []
    ).append(
        {
            "seed": seed,
            "trajectory": trajectory,
        }
    )


if not runs_by_initial_n:
    raise RuntimeError(
        f"No usable statistics.pt files found under: {ROOT}"
    )


# ================================================================
# Plot
# ================================================================

fig, ax = plt.subplots(
    figsize=(11, 7)
)

for initial_n in sorted(runs_by_initial_n):

    runs = runs_by_initial_n[initial_n]

    max_epochs = max(
        len(run["trajectory"])
        for run in runs
    )

    means = []
    stds = []
    counts = []

    # ------------------------------------------------------------
    # Calculate epoch-wise statistics.
    #
    # Runs are allowed to have different lengths.
    # At each epoch, only available runs contribute.
    #
    # If n == 1:
    #     std = 0
    #
    # If n >= 2:
    #     sample std (ddof=1)
    # ------------------------------------------------------------

    for epoch in range(max_epochs):

        values = [
            run["trajectory"][epoch]
            for run in runs
            if epoch < len(run["trajectory"])
        ]

        values = np.asarray(
            values,
            dtype=float,
        )

        counts.append(len(values))

        means.append(
            values.mean()
        )

        if len(values) >= 2:
            stds.append(
                values.std(ddof=1)
            )
        else:
            stds.append(0.0)

    means = np.asarray(means)
    stds = np.asarray(stds)
    counts = np.asarray(counts)

    epochs = np.arange(
        1,
        max_epochs + 1,
        )

    # ------------------------------------------------------------
    # Mean trajectory
    # ------------------------------------------------------------

    line, = ax.plot(
        epochs,
        means,
        linewidth=2,
        label=f"Initial N={initial_n}",
    )

    # ------------------------------------------------------------
    # Mean ± STD
    # ------------------------------------------------------------

    ax.fill_between(
        epochs,
        means - stds,
        means + stds,
        alpha=0.15,
        color=line.get_color(),
        )

    # ------------------------------------------------------------
    # Report number of contributing runs
    # ------------------------------------------------------------

    print(
        f"Initial N={initial_n:4d} | "
        f"runs={len(runs):2d} | "
        f"epochs={max_epochs:3d} | "
        f"min contributing runs={counts.min()} | "
        f"max contributing runs={counts.max()}"
    )


# ================================================================
# Reference line / formatting
# ================================================================

ax.set_xlabel(
    "Epoch",
    fontsize=12,
)

ax.set_ylabel(
    "Number of neurons",
    fontsize=12,
)

ax.set_title(
    "Neurogenesis trajectories: mean ± SD",
    fontsize=14,
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend(
    title="Initial network size",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)

fig.tight_layout()


# ================================================================
# Save
# ================================================================

if args.output is None:

    output = (
            ROOT
            / "neurogenesis_trajectories.png"
    )

else:

    output = Path(args.output)


fig.savefig(
    output,
    dpi=args.dpi,
    bbox_inches="tight",
)

print(
    f"\nSaved: {output}"
)

plt.show()