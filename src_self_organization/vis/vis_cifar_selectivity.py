#!/usr/bin/env python3
"""FlowGenesis CIFAR-10 neuron/class selectivity and top-5 real-image visualization.

Outputs:
  class_heatmaps.png        Step 1 vs final, neuron order fixed by final-step profiles
  top5_real_images.png      Five highest-activating test images for selected neurons
  class_response.npz        Raw/normalized class responses and counts
  neuron_class_scores.csv   Class means and selectivity for every neuron/step
  temporal_profile.csv      Per-neuron step-1 vs final class-profile correlations

Run from the interferenceNN repository root.
"""
import argparse
import csv
from importlib import import_module
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def symbol(name, modules):
    errors = []
    for module in modules:
        try:
            return getattr(import_module(module), name)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module}: {exc}")
    raise ImportError("\n".join(errors))


def load_model(path, device, override_steps):
    Config = symbol("Config", ["src_self_organization.config_fm"])
    NeuralContainer = symbol("NeuralContainer", [
        "src_self_organization.model_neurogenesis_new",
        "src_self_organization.model_neurogenesis"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    cfg = Config()
    saved = checkpoint.get("config", checkpoint.get("cfg")) if "state_dict" in checkpoint else None
    if saved is not None:
        entries = saved.items() if isinstance(saved, dict) else vars(saved).items()
        for key, value in entries:
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    else:
        print("WARNING: no saved config; using config_fm.Config for relaxation_steps.")
    cfg.hidden_neurons = int(state["W"].shape[0])
    cfg.input_dim = int(state["input_layer.weight"].shape[1])
    cfg.output_dim = int(state["output_layer.weight"].shape[0])
    if cfg.input_dim != 3072:
        raise ValueError(f"Expected CIFAR-10 input_dim=3072; got {cfg.input_dim}")
    if override_steps is not None:
        cfg.relaxation_steps = override_steps
    model = NeuralContainer(cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def states_at_steps(model, x, steps):
    """Matches NeuralContainer.relax, without changing model statistics."""
    drive = model.input_layer(x)
    h = torch.zeros_like(drive)
    weight = (model.W * model.mask) / model.degree
    found = {}
    for t in range(1, max(steps) + 1):
        h = 0.8 * h + 0.2 * F.relu(drive + F.linear(h, weight, model.bias))
        if t in steps:
            found[t] = h
    return found


def normalize_rows(means, eps=1e-8):
    center = means.mean(axis=1, keepdims=True)
    spread = means.std(axis=1, keepdims=True)
    return (means - center) / np.maximum(spread, eps)


def order_neurons(z):
    # Hierarchical clustering on the FINAL normalized class-response profiles.
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
        if len(z) < 3:
            return np.arange(len(z))
        distances = pdist(z, metric="euclidean")
        if np.all(distances < 1e-10):
            return np.arange(len(z))
        return leaves_list(linkage(distances, method="average", optimal_ordering=True))
    except ImportError:
        print("scipy not available: sorting neurons by preferred class instead.")
        return np.lexsort((-z.max(axis=1), z.argmax(axis=1)))


def plot_heatmaps(out, steps, z, order, dpi):
    fig, axes = plt.subplots(1, len(steps), figsize=(6.3 * len(steps), max(7, len(order) * .17)),
                             sharey=True, squeeze=False)
    vmax = max(1.0, float(np.max(np.abs(np.stack([z[t] for t in steps])))))
    for ax, t in zip(axes[0], steps):
        im = ax.imshow(z[t][order], aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Recurrent step {t}")
        ax.set_xticks(np.arange(10), CLASSES, rotation=55, ha="right", fontsize=9)
        yticks = np.arange(0, len(order), max(1, len(order) // 16))
        ax.set_yticks(yticks, order[yticks], fontsize=8)
        ax.set_xlabel("CIFAR-10 class")
    axes[0][0].set_ylabel("Neuron ID (same clustered order)")
    fig.colorbar(im, ax=list(axes[0]), shrink=.7, label="Within-neuron class z-score",
                 pad=.02)
    fig.suptitle("FlowGenesis: neuron-class response profiles", fontsize=14)
    fig.subplots_adjust(left=.08, right=.90, top=.93, bottom=.16, wspace=.15)
    fig.savefig(out / "class_heatmaps.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_top_images(out, chosen, best_images, best_labels, final_means, dpi):
    fig, axes = plt.subplots(len(chosen), 5,
                             figsize=(10, max(2.0, len(chosen) * 1.85)),
                             squeeze=False)
    for row, neuron in enumerate(chosen):
        pref = int(np.argmax(final_means[neuron]))
        for col in range(5):
            ax = axes[row, col]
            ax.axis("off")
            if best_images[neuron, col] is not None:
                ax.imshow(best_images[neuron, col])
                ax.set_title(CLASSES[int(best_labels[neuron, col])], fontsize=8)
            if col == 0:
                ax.set_ylabel(f"Neuron {neuron}\nmean peak: {CLASSES[pref]}",
                              fontsize=9, rotation=0, labelpad=75, va="center")
    fig.suptitle("Top five real CIFAR-10 test images per neuron (final step)", fontsize=13)
    fig.subplots_adjust(left=.19, right=.98, top=.94, bottom=.03, hspace=.5)
    fig.savefig(out / "top5_real_images.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--folder")
    source.add_argument("--checkpoint")
    parser.add_argument("--model-name", default="final_model.pt")
    parser.add_argument("--output-dir")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=0,
                        help="0 means entire CIFAR-10 test set (recommended)")
    parser.add_argument("--relaxation-steps", type=int, default=None)
    parser.add_argument("--time-steps", type=int, nargs="+", default=None,
                        help="Default: steps 1 and final; e.g. --time-steps 1 2 5 10")
    parser.add_argument("--neurons", type=int, nargs="+", default=None,
                        help="Neuron IDs for top-five images; default: 10 diverse neurons")
    parser.add_argument("--top-neurons", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1 or args.top_neurons < 1 or args.max_images < 0:
        parser.error("Invalid batch-size, top-neurons, or max-images")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device_name = ("cuda" if torch.cuda.is_available() else
                   "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device_name if args.device == "auto" else args.device)
    path = Path(args.checkpoint if args.checkpoint else
                Path(args.folder) / args.model_name).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    out = (Path(args.output_dir).expanduser() if args.output_dir else
           path.parent / "neuron_class_selectivity")
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(path, device, args.relaxation_steps)
    total_steps = int(model.steps)
    steps = sorted(set(args.time_steps if args.time_steps else [1, total_steps]))
    if min(steps) < 1 or max(steps) > total_steps:
        parser.error(f"Time steps must be between 1 and {total_steps}")
    print(f"Model: {path}\nDevice: {device}; neurons: {model.num_neurons}; "
          f"relaxation steps: {total_steps}; evaluated: {steps}", flush=True)

    with torch.no_grad():
        test = torch.rand(2, 3072, device=device)
        discrepancy = (states_at_steps(model, test, [total_steps])[total_steps]
                       - model.relax(test)).abs().max().item()
        if discrepancy > 1e-4:
            raise RuntimeError(f"Unrolled recurrence disagrees with relax(): {discrepancy}")

    ds = datasets.CIFAR10(args.data_dir, train=False, download=True,
                          transform=transforms.ToTensor())
    if args.max_images:
        rng = np.random.default_rng(args.seed)
        ids = rng.choice(len(ds), min(args.max_images, len(ds)), replace=False)
        ds = torch.utils.data.Subset(ds, sorted(ids.tolist()))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)
    n = model.num_neurons
    sums = {t: np.zeros((n, 10), dtype=np.float64) for t in steps}
    counts = np.zeros(10, dtype=np.int64)
    final_step = steps[-1]
    # Retain top-five *real test images* for each neuron.
    top_scores = np.full((n, 5), -np.inf, dtype=np.float32)
    top_labels = np.full((n, 5), -1, dtype=np.int64)
    top_images = np.empty((n, 5), dtype=object)
    top_images[:] = None

    for batch_idx, (images, labels) in enumerate(loader):
        x = images.flatten(1).to(device)
        labels_np = labels.numpy()
        hidden = states_at_steps(model, x, steps)
        for c in range(10):
            mask = labels_np == c
            counts[c] += int(mask.sum())
            if not mask.any():
                continue
            for t in steps:
                sums[t][:, c] += hidden[t][mask].sum(dim=0).cpu().numpy()
        H = hidden[final_step].cpu().numpy()
        rgb = images.permute(0, 2, 3, 1).numpy()
        # Per neuron, merge top-5 of this batch with current top-5.
        for neuron in range(n):
            k = min(5, len(H))
            local = np.argpartition(H[:, neuron], -k)[-k:]
            candidates = [(float(top_scores[neuron, j]),
                           int(top_labels[neuron, j]), top_images[neuron, j])
                          for j in range(5) if np.isfinite(top_scores[neuron, j])]
            candidates += [(float(H[idx, neuron]), int(labels_np[idx]), rgb[idx].copy())
                           for idx in local]
            candidates.sort(key=lambda item: item[0], reverse=True)
            for j, (score, label, img) in enumerate(candidates[:5]):
                top_scores[neuron, j] = score
                top_labels[neuron, j] = label
                top_images[neuron, j] = img
        print(f"Processed {min((batch_idx + 1) * args.batch_size, len(ds))}/{len(ds)}",
              end="\r", flush=True)
    print()
    if (counts == 0).any():
        raise ValueError("One or more classes are missing; increase --max-images")

    means = {t: sums[t] / counts[None, :] for t in steps}
    z = {t: normalize_rows(means[t]) for t in steps}
    order = order_neurons(z[final_step])
    plot_heatmaps(out, steps, z, order, args.dpi)

    if args.neurons is not None:
        chosen = args.neurons
    else:
        # Choose distinct strongly class-preferring neurons, rather than
        # multiple near-duplicates from the same response cluster.
        chosen = []
        for c in range(10):
            candidates = np.argsort(-z[final_step][:, c])
            candidate = next((int(i) for i in candidates if i not in chosen), None)
            if candidate is not None:
                chosen.append(candidate)
            if len(chosen) >= args.top_neurons:
                break
        for neuron in np.argsort(-z[final_step].max(axis=1)):
            if len(chosen) >= min(args.top_neurons, n):
                break
            if int(neuron) not in chosen:
                chosen.append(int(neuron))
    if any(i < 0 or i >= n for i in chosen):
        parser.error("Neuron index out of range")
    plot_top_images(out, chosen, top_images, top_labels, means[final_step], args.dpi)

    np.savez_compressed(out / "class_response.npz",
                        steps=np.asarray(steps), class_names=np.asarray(CLASSES),
                        counts=counts, neuron_order=order,
                        raw_means=np.stack([means[t] for t in steps]),
                        z_scores=np.stack([z[t] for t in steps]),
                        top_scores=top_scores, top_labels=top_labels,
                        top_images=np.stack([[np.asarray(top_images[i, j])
                                              for j in range(5)] for i in range(n)]))
    with (out / "neuron_class_scores.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "neuron", "class", "mean_activation",
                         "within_neuron_z", "class_count"])
        for t in steps:
            for i in range(n):
                for c in range(10):
                    writer.writerow([t, i, CLASSES[c], means[t][i, c],
                                     z[t][i, c], counts[c]])
    with (out / "temporal_profile.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["neuron", "step_first", "step_last", "class_profile_pearson"])
        for i in range(n):
            a, b = z[steps[0]][i], z[steps[-1]][i]
            corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) \
                if np.linalg.norm(a) > 1e-8 and np.linalg.norm(b) > 1e-8 else float("nan")
            writer.writerow([i, steps[0], steps[-1], corr])
    print(f"Saved: {out}")
    print(f"Top-image neuron IDs: {chosen}")
    print("Outputs: class_heatmaps.png, top5_real_images.png, class_response.npz, "
          "neuron_class_scores.csv, temporal_profile.csv")


if __name__ == "__main__":
    main()
