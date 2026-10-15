"""
Visualize the Fashion-MNIST connectivity (beta) ablation.

Creates two separate figures for a fixed initial network size N0:
  1. Test accuracy vs epoch, mean ± 1 SD across seeds
  2. Number of neurons vs epoch, mean ± 1 SD across seeds

Conditions:
  beta = 0.05, 0.10, 0.20, 0.40
  N0 = 40 by default

Expected layout:
  data/neurogenesis/beta_ablation/
      beta_0.05/
          neurons_40/
              seed_0/
              ...
      beta_0.1/
          neurons_40/
              seed_0/
              ...

The loader also accepts:
  beta_<beta>/N0_<N0>/seed_<seed>/
  beta_<beta>/n0_<N0>/seed_<seed>/
  beta_<beta>/seed_<seed>/   (only if runs are already N0-specific)

Each run should contain learning_curves.pt with per-epoch:
  "epoch", "test_accuracy", and preferably "num_neurons".

If num_neurons is absent, statistics.pt is searched for a recoverable
population trajectory.
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path("data/neurogenesis/data/neurogenesis") / "beta_ablation"
OUT = ROOT / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# Main-paper connectivity comparison: hold initial capacity fixed.
N0 = 40
BETA_VALUES = [0.05, 0.10, 0.20, 0.40]

# Same restrained blue progression as the N0 figure.
COLORS = {
    0.05: "lightsteelblue",
    0.10: "steelblue",
    0.20: "#35688f",
    0.40: "#102f46",
}


def as_1d_numeric(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    try:
        arr = np.asarray(x, dtype=float).reshape(-1)
    except Exception:
        return None
    return arr if arr.size else None


def recover_neuron_history(run_dir, curve):
    for key in ("num_neurons", "neurons", "neuron_history", "population"):
        if key in curve:
            arr = as_1d_numeric(curve[key])
            if arr is not None:
                return arr

    stats_path = run_dir / "statistics.pt"
    if not stats_path.exists():
        return None

    stats = torch.load(stats_path, map_location="cpu", weights_only=False)

    if isinstance(stats, dict):
        for key in (
                "neurons",
                "neuron_history",
                "num_neurons_history",
                "population",
                "population_history",
                "alive_history",
        ):
            if key in stats:
                arr = as_1d_numeric(stats[key])
                if arr is not None:
                    return arr

        for key in ("history", "epoch_history", "development_history"):
            if key in stats and isinstance(stats[key], (list, tuple)):
                rows = stats[key]
                if rows and isinstance(rows[0], dict):
                    for nk in ("neurons", "num_neurons", "population", "alive"):
                        if nk in rows[0]:
                            return np.asarray(
                                [float(row[nk]) for row in rows if nk in row],
                                dtype=float,
                            )

    return None


def beta_dir(beta):
    """Resolve beta directory while tolerating 0.1 vs 0.10 naming."""
    candidates = [
        ROOT / f"beta_{beta:g}",
        ROOT / f"beta_{beta}",
        ROOT / f"beta_{beta:.2f}",
        ]
    for d in candidates:
        if d.exists():
            return d
    return candidates[0]


def find_run_dirs(beta, n0=N0):
    """
    Find completed runs for one beta at the requested N0.

    Prefer layouts that explicitly encode N0 so that results from the
    100-run beta x N0 experiment are never accidentally mixed together.
    """
    base = beta_dir(beta)

    n0_candidates = [
        base / f"neurons_{n0}",
        base / f"N0_{n0}",
        base / f"n0_{n0}",
        base / f"hidden_{n0}",
        base / f"hidden_neurons_{n0}",
        ]

    for parent in n0_candidates:
        dirs = list(parent.glob("seed_*"))
        dirs = [d for d in dirs if (d / "learning_curves.pt").exists()]
        if dirs:
            return sorted(dirs, key=lambda d: int(d.name.split("_")[-1]))

    # Fallback for a beta directory that contains only one N0 condition.
    direct = [
        d for d in base.glob("seed_*")
        if (d / "learning_curves.pt").exists()
    ]
    if direct:
        print(
            f"[WARN] beta={beta:g}: using direct seed_* directories under {base}. "
            f"This is safe only if these runs correspond to N0={n0}."
        )
        return sorted(direct, key=lambda d: int(d.name.split("_")[-1]))

    return []


def load_condition(beta):
    run_dirs = find_run_dirs(beta)

    if not run_dirs:
        print(f"[WARN] beta={beta:g}: no completed N0={N0} runs found")
        return None

    acc_runs = []
    neuron_runs = []
    epoch_runs = []

    for run_dir in run_dirs:
        curve = torch.load(
            run_dir / "learning_curves.pt",
            map_location="cpu",
            weights_only=False,
            )

        epochs = as_1d_numeric(curve.get("epoch"))
        acc = as_1d_numeric(curve.get("test_accuracy"))

        if epochs is None or acc is None:
            print(f"[WARN] Skipping {run_dir}: missing epoch/test_accuracy")
            continue

        L = min(len(epochs), len(acc))
        epochs = epochs[:L]
        acc = acc[:L]

        neurons = recover_neuron_history(run_dir, curve)
        if neurons is not None:
            L2 = min(L, len(neurons))
            neuron_runs.append(neurons[:L2])

        epoch_runs.append(epochs)
        acc_runs.append(acc)

    if not acc_runs:
        return None

    # Use the common completed prefix across seeds.
    acc_len = min(len(x) for x in acc_runs)
    epochs = epoch_runs[0][:acc_len]
    acc = np.stack([x[:acc_len] for x in acc_runs])

    result = {
        "epochs": epochs,
        "acc_mean": acc.mean(axis=0),
        "acc_std": (
            acc.std(axis=0, ddof=1)
            if len(acc_runs) > 1
            else np.zeros(acc_len)
        ),
        "n_acc_runs": len(acc_runs),
        "neuron_mean": None,
        "neuron_std": None,
        "neuron_epochs": None,
        "n_neuron_runs": len(neuron_runs),
    }

    if neuron_runs:
        n_len = min(len(x) for x in neuron_runs)
        neurons = np.stack([x[:n_len] for x in neuron_runs])
        result["neuron_epochs"] = epochs[:n_len]
        result["neuron_mean"] = neurons.mean(axis=0)
        result["neuron_std"] = (
            neurons.std(axis=0, ddof=1)
            if len(neuron_runs) > 1
            else np.zeros(n_len)
        )

    print(
        f"beta={beta:g}, N0={N0}: "
        f"accuracy={len(acc_runs)} run(s), "
        f"neurons={len(neuron_runs)} run(s)"
    )
    return result


results = {beta: load_condition(beta) for beta in BETA_VALUES}
results = {beta: r for beta, r in results.items() if r is not None}

if not results:
    raise RuntimeError(
        f"No completed beta-ablation runs found under {ROOT} for N0={N0}. "
        "Expected beta_<beta>/neurons_<N0>/seed_<seed>/ (or one of the "
        "supported alternative layouts)."
    )


# ============================================================
# Figure 1: Test accuracy
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 4.8))

for beta in BETA_VALUES:
    if beta not in results:
        continue

    r = results[beta]
    x = r["epochs"]
    mean = r["acc_mean"]
    std = r["acc_std"]

    ax.plot(
        x,
        mean,
        linewidth=1.3,
        color=COLORS[beta],
        label=rf"$\beta={beta:g}$",
    )

    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=COLORS[beta],
        alpha=0.065,
        linewidth=0,
        )

ax.set_xlabel("Epoch")
ax.set_ylabel("Test accuracy (%)")
ax.grid(alpha=0.20, linestyle="--")
ax.legend(frameon=False)
fig.tight_layout()

fig.savefig(
    OUT / "fashionmnist_beta_test_accuracy.png",
    dpi=300,
    bbox_inches="tight",
    )
fig.savefig(
    OUT / "fashionmnist_beta_test_accuracy.pdf",
    bbox_inches="tight",
    )
plt.close(fig)


# ============================================================
# Figure 2: Number of neurons
# ============================================================

available_neurons = [
    beta for beta in BETA_VALUES
    if beta in results and results[beta]["neuron_mean"] is not None
]

if available_neurons:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for beta in available_neurons:
        r = results[beta]
        x = r["neuron_epochs"]
        mean = r["neuron_mean"]
        std = r["neuron_std"]

        ax.plot(
            x,
            mean,
            linewidth=1.3,
            color=COLORS[beta],
            label=rf"$\beta={beta:g}$",
        )

        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=COLORS[beta],
            alpha=0.065,
            linewidth=0,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Number of neurons")
    ax.grid(alpha=0.20, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(
        OUT / "fashionmnist_beta_neurons.png",
        dpi=300,
        bbox_inches="tight",
        )
    fig.savefig(
        OUT / "fashionmnist_beta_neurons.pdf",
        bbox_inches="tight",
        )
    plt.close(fig)
else:
    print(
        "\n[WARN] No per-epoch neuron histories were found. "
        "The accuracy figure was created, but the neuron figure cannot be "
        "reconstructed from final neuron counts alone.\n"
        "Expected the per-epoch 'num_neurons' field written by "
        "train_neurogenesis_ablation.py."
    )


print(f"\nSaved figures to: {OUT}")
print("  fashionmnist_beta_test_accuracy.png/.pdf")
if available_neurons:
    print("  fashionmnist_beta_neurons.png/.pdf")
