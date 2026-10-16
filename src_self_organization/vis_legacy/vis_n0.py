"""
Visualize the Fashion-MNIST N0 attractor experiment.

Creates two separate figures:
  1. Test accuracy vs epoch, mean ± 1 SD across seeds
  2. Number of neurons vs epoch, mean ± 1 SD across seeds

Conditions:
  N0 = 10, 20, 40, 80, 160
Colors:
  20  -> red
  40  -> steelblue
  80  -> orange
  160 -> darkorange

The script supports either:
  A) learning_curves.pt containing per-epoch "epoch", "test_accuracy", and "num_neurons", OR
  B) statistics.pt containing a recoverable per-epoch population trajectory.

If neuron trajectories cannot be recovered, the accuracy plot is still produced
and the script prints a precise warning for the neuron plot.
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path("data/neurogenesis") / "N0_ablation"
OUT = ROOT / "plots"
OUT.mkdir(parents=True, exist_ok=True)

N0_VALUES = [10, 20, 40, 80, 160]

COLORS = {
    10: "lightsteelblue",
    20: "steelblue",
    40: "#35688f",
    80: "#234a68",
    160: "#102f46",
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
    # Preferred: saved directly in learning_curves.pt
    for key in ("num_neurons", "neurons", "neuron_history", "population"):
        if key in curve:
            arr = as_1d_numeric(curve[key])
            if arr is not None:
                return arr

    # Fallback: inspect developmental statistics.
    stats_path = run_dir / "statistics.pt"
    if not stats_path.exists():
        return None

    stats = torch.load(stats_path, map_location="cpu", weights_only=False)

    if isinstance(stats, dict):
        # Direct likely keys.
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

        # Some implementations store an epoch-wise list of dictionaries.
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


def find_run_dirs(n0):
    # New N0-ablation layout:
    # data/neurogenesis/N0_ablation/neurons_<N0>/seed_<seed>/
    primary = ROOT / f"neurons_{n0}"
    dirs = sorted(primary.glob("seed_*"))
    return sorted(
        [d for d in dirs if (d / "learning_curves.pt").exists()],
        key=lambda d: int(d.name.split("_")[-1]),
    )


def load_condition(n0):
    run_dirs = find_run_dirs(n0)
    if not run_dirs:
        print(f"[WARN] N0={n0}: no completed runs found")
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

    # Common prefix makes the script robust to interrupted runs.
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
        f"N0={n0}: accuracy={len(acc_runs)} run(s), "
        f"neurons={len(neuron_runs)} run(s)"
    )
    return result


results = {n0: load_condition(n0) for n0 in N0_VALUES}
results = {n0: r for n0, r in results.items() if r is not None}

if not results:
    raise RuntimeError(
        f"No learning_curves.pt files found under {ROOT}. "
        f"Expected neurons_<N0>/seed_<seed>/."
    )


# ============================================================
# Figure 1: Test accuracy
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 4.8))

for n0 in N0_VALUES:
    if n0 not in results:
        continue

    r = results[n0]
    x = r["epochs"]
    mean = r["acc_mean"]
    std = r["acc_std"]

    ax.plot(
        x,
        mean,
        linewidth=1.3,
        color=COLORS[n0],
        label=rf"$N_0={n0}$",
    )

    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=COLORS[n0],
        alpha=0.065,
        linewidth=0,
        )

ax.set_xlabel("Epoch")
ax.set_ylabel("Test accuracy (%)")
ax.grid(alpha=0.20, linestyle="--")
ax.legend(frameon=False)
fig.tight_layout()

fig.savefig(
    OUT / "fashionmnist_n0_test_accuracy.png",
    dpi=300,
    bbox_inches="tight",
    )
fig.savefig(
    OUT / "fashionmnist_n0_test_accuracy.pdf",
    bbox_inches="tight",
    )
plt.close(fig)


# ============================================================
# Figure 2: Number of neurons
# ============================================================

available_neurons = [
    n0 for n0 in N0_VALUES
    if n0 in results and results[n0]["neuron_mean"] is not None
]

if available_neurons:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for n0 in available_neurons:
        r = results[n0]
        x = r["neuron_epochs"]
        mean = r["neuron_mean"]
        std = r["neuron_std"]

        ax.plot(
            x,
            mean,
            linewidth=1.3,
            color=COLORS[n0],
            label=rf"$N_0={n0}$",
        )

        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=COLORS[n0],
            alpha=0.065,
            linewidth=0,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Number of neurons")
    ax.grid(alpha=0.20, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(
        OUT / "fashionmnist_n0_neurons.png",
        dpi=300,
        bbox_inches="tight",
        )
    fig.savefig(
        OUT / "fashionmnist_n0_neurons.pdf",
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
print("  fashionmnist_n0_test_accuracy.png/.pdf")
if available_neurons:
    print("  fashionmnist_n0_neurons.png/.pdf")