#!/usr/bin/env python3
"""Four separate neuron-population plots for run_regime_n_sweep.sh.

Usage:
  python vis_regime_n_sweep.py --root data/neurogenesis/regime_n_sweep

Saves one PNG and PDF per regime under <root>/plots.
Each panel is a regime; curves are mean +/- SD over available seeds.
"""
from pathlib import Path
import argparse
import re
import numpy as np
import torch
import matplotlib.pyplot as plt

REGIMES = ("full", "no_birth", "no_death", "static")
N0S = (20, 40, 80, 160)
COLORS = {20: "black", 40: "steelblue", 80: "orange", 160: "red"}
NEURON_KEYS = ("num_neurons", "neurons", "neuron_history", "num_neurons_history",
               "population", "population_history", "alive_history")

def array(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    try:
        a = np.asarray(x, dtype=float).reshape(-1)
        return a if a.size else None
    except (ValueError, TypeError):
        return None

def extract(obj):
    if not isinstance(obj, dict):
        return None
    for key in NEURON_KEYS:
        if key in obj:
            a = array(obj[key])
            if a is not None:
                return a
    for key in ("history", "epoch_history", "development_history"):
        rows = obj.get(key)
        if isinstance(rows, (list, tuple)) and rows and isinstance(rows[0], dict):
            for nk in NEURON_KEYS:
                if nk in rows[0]:
                    return array([r[nk] for r in rows if nk in r])
    return None

def load_run(folder):
    curve_path = folder / "learning_curves.pt"
    curve = torch.load(curve_path, map_location="cpu", weights_only=False)
    values = extract(curve)
    if values is None and (folder / "statistics.pt").exists():
        values = extract(torch.load(folder / "statistics.pt", map_location="cpu", weights_only=False))
    if values is None:
        return None
    epochs = array(curve.get("epoch")) if isinstance(curve, dict) else None
    if epochs is None:
        epochs = np.arange(1, len(values) + 1, dtype=float)
    length = min(len(epochs), len(values))
    return epochs[:length], values[:length]

def locate(root, regime, n0):
    """Accept both dataset/regime/<value>/neurons_N/seed_K and other nested layouts."""
    matches = []
    for p in root.rglob("learning_curves.pt"):
        parts = p.relative_to(root).parts[:-1]
        tokens = [v.lower() for v in parts]
        if f"value_{regime}" not in tokens or f"neurons_{n0}" not in tokens:
            continue
        if not any(re.fullmatch(r"seed_?\d+", v) for v in tokens):
            continue
        matches.append(p.parent)
    return sorted(matches)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/neurogenesis/regime_n_sweep"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dataset", default="fashionmnist", help="Used in figure title")
    args = parser.parse_args()
    root = args.root.expanduser()
    out = args.out.expanduser() if args.out else root / "plots"
    out.mkdir(parents=True, exist_ok=True)

    any_data = False
    for regime in REGIMES:
        fig, ax = plt.subplots(figsize=(8, 5))
        regime_has_data = False
        for n0 in N0S:
            dirs = locate(root, regime, n0)
            runs = []
            for folder in dirs:
                try:
                    run = load_run(folder)
                    if run is not None:
                        runs.append(run)
                    else:
                        print(f"[WARN] No neuron trajectory in {folder}")
                except Exception as exc:
                    print(f"[WARN] Could not read {folder}: {exc}")
            if not runs:
                print(f"[MISSING] regime={regime}, N0={n0}")
                continue
            common = min(len(y) for _, y in runs)
            x = runs[0][0][:common]
            valid = [y[:common] for e, y in runs if len(e) >= common and np.allclose(e[:common], x)]
            if not valid:
                print(f"[WARN] Epoch grids differ for regime={regime}, N0={n0}")
                continue
            ys = np.stack(valid)
            mean = ys.mean(axis=0)
            std = ys.std(axis=0, ddof=1) if len(valid) > 1 else np.zeros_like(mean)
            ax.plot(x, mean, color=COLORS[n0], linewidth=1.8, label=rf"$N_0={n0}$")
            ax.fill_between(x, mean - std, mean + std, color=COLORS[n0], alpha=0.13, linewidth=0)
            print(f"[OK] {regime:8s} N0={n0:3d}: {len(valid)} seed(s), {common} epochs")
            regime_has_data = True
            any_data = True
        if not regime_has_data:
            plt.close(fig)
            continue
        ax.set_title({"full": "Full (birth + death)", "no_birth": "No birth",
                      "no_death": "No death", "static": "Static"}[regime])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Number of neurons")
        ax.grid(alpha=0.20, linestyle="--")
        ax.legend(frameon=False, fontsize=10)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            target = out / f"regime_{regime}_neurons.{ext}"
            fig.savefig(target, dpi=300, bbox_inches="tight")
            print(f"Saved {target}")
        plt.close(fig)
    if not any_data:
        raise SystemExit(f"No neuron trajectories found under {root}; check saved folder layout")

if __name__ == "__main__":
    main()
