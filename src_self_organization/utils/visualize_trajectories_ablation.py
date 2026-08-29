from pathlib import Path
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description=(
        "Plot neurogenesis trajectories and train/test accuracy for "
        "rate, beta, warmup, threshold, or regime ablations."
    )
)
parser.add_argument(
    "folder",
    type=str,
    help=(
        "Ablation root containing rate_*/beta_*/birth_*/death_*/neurons_*/seed_* "
        "or a single rate_* / beta_* / birth_* / death_* folder."
    ),
)
parser.add_argument(
    "--rates",
    type=int,
    nargs="*",
    default=None,
    help="Optional subset of rates to plot, e.g. --rates 2 4 8 16 32.",
)
parser.add_argument(
    "--warmups",
    type=int,
    nargs="*",
    default=None,
    help="Optional subset of warmup values to plot, e.g. --warmups 0 5 25.",
)
parser.add_argument(
    "--birth-thresholds",
    type=float,
    nargs="*",
    default=None,
    help=(
        "Optional subset of birth thresholds to plot, "
        "e.g. --birth-thresholds 0.05 0.1 0.25 0.5."
    ),
)
parser.add_argument(
    "--death-thresholds",
    type=float,
    nargs="*",
    default=None,
    help=(
        "Optional subset of death thresholds to plot, "
        "e.g. --death-thresholds 0.01 0.05 0.1 0.25."
    ),
)
parser.add_argument(
    "--betas",
    type=float,
    nargs="*",
    default=None,
    help="Optional subset of beta values to plot, e.g. --betas 0.1 0.25 0.5 0.75 1.0.",
)
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--dpi", type=int, default=200)
args = parser.parse_args()

ROOT = Path(args.folder)
if not ROOT.exists():
    raise FileNotFoundError(f"Folder does not exist: {ROOT}")


def to_array(values):
    if values is None:
        return None
    return np.asarray(
        [float(x.item()) if torch.is_tensor(x) else float(x) for x in values],
        dtype=float,
    )


def mean_std(runs, key):
    arrays = [run[key] for run in runs if run.get(key) is not None and len(run[key])]
    if not arrays:
        return None, None, None
    max_len = max(len(a) for a in arrays)
    means, stds, counts = [], [], []
    for i in range(max_len):
        vals = np.asarray([a[i] for a in arrays if i < len(a)], dtype=float)
        means.append(vals.mean())
        stds.append(vals.std(ddof=1) if len(vals) >= 2 else 0.0)
        counts.append(len(vals))
    return np.asarray(means), np.asarray(stds), np.asarray(counts)

#======================================================

if ROOT.name.startswith("rate_"):
    experiment_type = "rate"
    condition_dirs = [ROOT]

elif ROOT.name.startswith("beta_"):
    experiment_type = "beta"
    condition_dirs = [ROOT]

elif ROOT.name.startswith("warmup_"):
    experiment_type = "warmup"
    condition_dirs = [ROOT]

elif ROOT.name.startswith("birth_"):
    experiment_type = "birth"
    condition_dirs = [ROOT]

elif ROOT.name.startswith("death_"):
    experiment_type = "death"
    condition_dirs = [ROOT]

else:
    rate_dirs = [p for p in ROOT.glob("rate_*") if p.is_dir()]
    birth_dirs = [p for p in ROOT.glob("birth_*") if p.is_dir()]
    death_dirs = [p for p in ROOT.glob("death_*") if p.is_dir()]
    beta_dirs = [p for p in ROOT.glob("beta_*") if p.is_dir()]
    warmup_dirs = [p for p in ROOT.glob("warmup_*") if p.is_dir()]
    regime_names = ["full", "no_birth", "no_death", "static"]

    regime_dirs = [
        ROOT / name
        for name in regime_names
        if (ROOT / name).is_dir()
    ]
    if rate_dirs:
        experiment_type = "rate"
        condition_dirs = sorted(
            rate_dirs,
            key=lambda p: int(p.name.split("_", 1)[1]),
        )
    elif beta_dirs:
        experiment_type = "beta"
        condition_dirs = sorted(
            beta_dirs,
            key=lambda p: float(p.name.split("_", 1)[1]),
        )

    elif birth_dirs:
        experiment_type = "birth"
        condition_dirs = sorted(
            birth_dirs,
            key=lambda p: float(p.name.split("_", 1)[1]),
        )

    elif warmup_dirs:
        experiment_type = "warmup"
        condition_dirs = sorted(
            warmup_dirs,
            key=lambda p: int(p.name.split("_", 1)[1]),
        )
    elif death_dirs:
        experiment_type = "death"
        condition_dirs = sorted(
            death_dirs,
            key=lambda p: float(p.name.split("_", 1)[1]),
        )

    elif regime_dirs:
        experiment_type = "regime"
        condition_dirs = regime_dirs

    else:
        raise RuntimeError(
            f"No rate_*, beta_*, warmup_*, birth_*, death_*, "
            f"or regime directories found under: {ROOT}"
        )


# ============================================================
# Optional filtering
# ============================================================

if experiment_type == "rate" and args.rates is not None:
    wanted = set(args.rates)

    condition_dirs = [
        p for p in condition_dirs
        if int(p.name.split("_", 1)[1]) in wanted
    ]


if experiment_type == "birth" and args.birth_thresholds is not None:
    wanted = np.asarray(args.birth_thresholds, dtype=float)

    condition_dirs = [
        p for p in condition_dirs
        if np.any(
            np.isclose(
                float(p.name.split("_", 1)[1]),
                wanted,
            )
        )
    ]
if experiment_type == "death" and args.death_thresholds is not None:
    wanted = np.asarray(args.death_thresholds, dtype=float)

    condition_dirs = [
        p for p in condition_dirs
        if np.any(
            np.isclose(
                float(p.name.split("_", 1)[1]),
                wanted,
            )
        )
    ]
if experiment_type == "warmup" and args.warmups is not None:
    wanted = set(args.warmups)

    condition_dirs = [
        p for p in condition_dirs
        if int(p.name.split("_", 1)[1]) in wanted
    ]

if experiment_type == "beta" and args.betas is not None:
    wanted = np.asarray(args.betas, dtype=float)

    condition_dirs = [
        p for p in condition_dirs
        if np.any(
            np.isclose(
                float(p.name.split("_", 1)[1]),
                wanted,
            )
        )
    ]
if not condition_dirs:
    raise RuntimeError(
        f"No matching experiment directories found under: {ROOT}"
    )
runs_by_condition = {}

for condition_dir in condition_dirs:

    if experiment_type in ("rate", "warmup"):
        condition = int(condition_dir.name.split("_", 1)[1])

    elif experiment_type in ("beta", "birth", "death"):
        condition = float(condition_dir.name.split("_", 1)[1])

    elif experiment_type == "regime":
        condition = condition_dir.name

    by_n = {}

    for stats_file in sorted(
            condition_dir.glob("neurons_*/seed_*/statistics.pt")
    ):
        run_dir = stats_file.parent
        initial_n = int(run_dir.parent.name.split("_")[1])
        seed = int(run_dir.name.split("_")[1])

        stats = torch.load(stats_file, map_location="cpu", weights_only=False)
        neurons = to_array(stats.get("neuron_count"))
        if neurons is None or len(neurons) == 0:
            print(f"WARNING: neuron_count missing/empty: {stats_file}")
            continue

        curves_file = run_dir / "learning_curves.pt"
        train_acc = test_acc = None
        if curves_file.exists():
            curves = torch.load(curves_file, map_location="cpu", weights_only=False)
            train_acc = to_array(curves.get("train_accuracy"))
            test_acc = to_array(curves.get("test_accuracy"))
        else:
            print(f"WARNING: learning_curves.pt missing: {run_dir}")

        by_n.setdefault(initial_n, []).append({
            "seed": seed,
            "neurons": neurons,
            "train_acc": train_acc,
            "test_acc": test_acc,
        })

    if by_n:
        runs_by_condition[condition] = by_n


if not runs_by_condition:
    raise RuntimeError(f"No usable runs found under: {ROOT}")



if experiment_type == "regime":
    regime_order = ["full", "no_birth", "no_death", "static"]
    conditions = [
        r for r in regime_order
        if r in runs_by_condition
    ]
else:
    conditions = sorted(runs_by_condition)
ncols = len(conditions)

fig, axes = plt.subplots(
    2,
    ncols,
    figsize=(5.3 * ncols, 8.2),
    squeeze=False,
    sharex="col",
)

# Keep initial-N colors consistent across all rates and both rows.
all_initial_n = sorted({
    n
    for by_n in runs_by_condition.values()
    for n in by_n
})
color_cycle = [
    "#4E79A7",  # blue
    "#E39C37",  # orange
    "#59A14F",  # green
    "#C95A5A",  # red
    "#4FA3A5",  # teal
    "#EDC948",  # muted yellow
    "#9C755F",  # brown
    "#79706E",  # warm gray
]
colors = {n: color_cycle[i % len(color_cycle)] for i, n in enumerate(all_initial_n)}

for col, condition in enumerate(conditions):
    ax_n = axes[0, col]
    ax_a = axes[1, col]

    by_n = runs_by_condition[condition]

    # --------------------------------------------------------
    # Subtitle for each experimental condition
    # --------------------------------------------------------
    if experiment_type == "rate":
        subtitle = f"Rate = {condition:g}"
    elif experiment_type == "beta":
        subtitle = rf"$\beta$ = {condition:g}"
    elif experiment_type == "warmup":
        subtitle = f"Warm-up = {condition:g}"
    elif experiment_type == "birth":
        subtitle = f"Birth threshold = {condition:g}"

    elif experiment_type == "death":
        subtitle = f"Death threshold = {condition:g}"

    elif experiment_type == "regime":
        regime_titles = {
            "full": "Full",
            "no_birth": "No Birth",
            "no_death": "No Death",
            "static": "Static",
        }
        subtitle = regime_titles[condition]

    ax_n.set_title(subtitle, fontsize=13, pad=10)

    # Separate legend entries for train and test.
    train_handles = []
    train_labels = []
    test_handles = []
    test_labels = []

    # --------------------------------------------------------
    # Plot each initial network size
    # --------------------------------------------------------
    for initial_n in sorted(by_n):
        runs = by_n[initial_n]
        color = colors[initial_n]

        # ====================================================
        # Neuron-count trajectory
        # ====================================================
        n_mean, n_std, n_count = mean_std(runs, "neurons")

        epochs = np.arange(1, len(n_mean) + 1)

        ax_n.plot(
            epochs,
            n_mean,
            linewidth=2.0,
            color=color,
            label=f"N0={initial_n}",
        )

        ax_n.fill_between(
            epochs,
            n_mean - n_std,
            n_mean + n_std,
            alpha=0.10,
            color=color,
            )

        # ====================================================
        # Accuracy trajectories
        # ====================================================
        train_mean, train_std, _ = mean_std(runs, "train_acc")
        test_mean, test_std, _ = mean_std(runs, "test_acc")

        # --------------------
        # Train
        # --------------------
        if train_mean is not None:
            e = np.arange(1, len(train_mean) + 1)

            train_line, = ax_a.plot(
                e,
                train_mean,
                linewidth=1.6,
                linestyle="--",
                alpha=0.9,
                color=color,
            )

            ax_a.fill_between(
                e,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.05,
                color=color,
                )

            train_handles.append(train_line)
            train_labels.append(f"N0={initial_n}")

        # --------------------
        # Test
        # --------------------
        if test_mean is not None:
            e = np.arange(1, len(test_mean) + 1)

            test_line, = ax_a.plot(
                e,
                test_mean,
                linewidth=2.0,
                linestyle="-",
                color=color,
            )

            ax_a.fill_between(
                e,
                test_mean - test_std,
                test_mean + test_std,
                alpha=0.07,
                color=color,
                )

            test_handles.append(test_line)
            test_labels.append(f"N0={initial_n}")

        # ====================================================
        # Console diagnostics
        # ====================================================
        if experiment_type == "rate":
            condition_text = f"rate={condition:g}"
        elif experiment_type == "beta":
            condition_text = f"beta={condition:g}"
        elif experiment_type == "warmup":
            condition_text = f"warmup={condition:g}"
        elif experiment_type == "birth":
             condition_text = f"birth={condition:g}"

        elif experiment_type == "death":
            condition_text = f"death={condition:g}"
        elif experiment_type == "regime":
            condition_text = condition

        print(
            f"{condition_text:12s} | "
            f"N0={initial_n:4d} | "
            f"runs={len(runs):2d} | "
            f"epochs={len(n_mean):3d} | "
            f"min contributing={n_count.min()}"
        )

    # ========================================================
    # Top panel formatting
    # ========================================================
    ax_n.set_ylabel("Number of neurons", fontsize=12)
    ax_n.grid(True, alpha=0.20)

    ax_n.legend(
        fontsize=8,
        loc="upper right",
        frameon=True,
    )

    # ========================================================
    # Bottom panel formatting
    # ========================================================
    ax_a.set_xlabel("Epoch", fontsize=12)
    ax_a.set_ylabel("Accuracy (%)", fontsize=12)
    ax_a.set_ylim(80, 100)
    ax_a.grid(True, alpha=0.20)

    # ========================================================
    # Separate TRAIN and TEST legends
    # ========================================================
    if train_handles:
        train_legend = ax_a.legend(
            train_handles,
            train_labels,
            title="Train",
            fontsize=8,
            title_fontsize=9,
            loc="lower left",
            bbox_to_anchor=(0.02, 0.02),
            frameon=True,
            borderaxespad=0.0,
        )

        # Keep this legend when the second one is added.
        ax_a.add_artist(train_legend)

    if test_handles:
        ax_a.legend(
            test_handles,
            test_labels,
            title="Test",
            fontsize=8,
            title_fontsize=9,
            loc="lower left",
            bbox_to_anchor=(0.27, 0.02),
            frameon=True,
            borderaxespad=0.0,
        )


fig.tight_layout()

if args.output is None:
    output = ROOT / "neurogenesis_trajectories_and_accuracy.png"
else:
    output = Path(args.output)

output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
print(f"\nSaved: {output}")
plt.show()