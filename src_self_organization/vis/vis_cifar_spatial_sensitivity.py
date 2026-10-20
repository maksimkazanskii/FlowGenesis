#!/usr/bin/env python3
"""Spatial sensitivity of saved FlowGenesis neurons on real CIFAR-10 test images.

Run from the interferenceNN repository root, with this file at
src_self_organization/vis/vis_cifar_spatial_sensitivity.py.

For each neuron, find its highest-activating test images and compute
sum_c |d h_i^(t) / d x_c| by backpropagating through recurrent steps.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from src_self_organization.vis.vis_cifar_selectivity import (
        load_model, CLASSES,
    )
except ImportError as exc:
    raise ImportError('Place this script alongside vis_cifar_selectivity.py') from exc


def differentiable_states_at_steps(model, x, steps):
    """Exact recurrent update from vis_cifar_selectivity, with autograd enabled.

    Do not call states_at_steps from vis_cifar_selectivity here: that helper
    is decorated with @torch.no_grad() and disconnects input gradients.
    """
    drive = model.input_layer(x)
    h = torch.zeros_like(drive)
    weight = (model.W * model.mask) / model.degree
    found = {}
    requested = set(steps)
    for t in range(1, max(requested) + 1):
        h = 0.8 * h + 0.2 * F.relu(drive + F.linear(h, weight, model.bias))
        if t in requested:
            found[t] = h
    return found


def select_images(model, loader, neuron_ids, step, k, device):
    """Return top-k examples per neuron, with dataset indices and scores."""
    best = {n: [] for n in neuron_ids}
    offset = 0
    with torch.no_grad():
        for images, labels in loader:
            x = images.flatten(1).to(device)
            h = differentiable_states_at_steps(model, x, [step])[step]
            for n in neuron_ids:
                scores = h[:, n].cpu().numpy()
                local = np.argsort(scores)[-min(k, len(scores)):][::-1]
                candidates = best[n] + [
                    (float(scores[j]), offset + int(j), int(labels[j])) for j in local
                ]
                candidates.sort(reverse=True)
                best[n] = candidates[:k]
            offset += len(images)
    return best


def gradient_map(model, image, neuron, step, device, method='abs', smooth_samples=16,
                 smooth_sigma=0.08):
    """Gradient saliency with respect to raw [0,1] RGB pixels."""
    x0 = image.unsqueeze(0).to(device)
    maps = []
    reps = smooth_samples if method == 'smoothgrad' else 1
    for _ in range(reps):
        if method == 'smoothgrad':
            x = (x0 + torch.randn_like(x0) * smooth_sigma).clamp(0, 1).detach()
        else:
            x = x0.detach()
        x.requires_grad_(True)
        h = differentiable_states_at_steps(model, x.flatten(1), [step])[step]
        if not h.requires_grad:
            raise RuntimeError("Recurrent state is detached from the input")
        grad = torch.autograd.grad(h[0, neuron], x)[0]
        maps.append(grad.detach().cpu())
    # SmoothGrad: average squared gradients, then take square root.
    if method == 'smoothgrad':
        saliency = torch.stack(maps).square().mean(0).sqrt().sum(dim=1)[0]
    else:
        saliency = maps[0].abs().sum(dim=1)[0]
    return saliency.numpy()


def normalize_saliency(s, percentile=99.0):
    scale = np.percentile(s, percentile)
    if scale <= 1e-12:
        return np.zeros_like(s)
    return np.clip(s / scale, 0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', required=True)
    p.add_argument('--model-name', default='final_model.pt')
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    p.add_argument('--neurons', type=int, nargs='+', default=[0, 3, 7, 12, 20, 30, 40, 50])
    p.add_argument('--top-k', type=int, default=2)
    p.add_argument('--step', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--max-images', type=int, default=10000)
    p.add_argument('--method', choices=['abs', 'smoothgrad'], default='smoothgrad')
    p.add_argument('--smooth-samples', type=int, default=16)
    p.add_argument('--smooth-sigma', type=float, default=0.08)
    p.add_argument('--overlay-alpha', type=float, default=0.55)
    p.add_argument('--dpi', type=int, default=180)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'auto':
        dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        dev = args.device
    device = torch.device(dev)
    checkpoint = Path(args.folder).expanduser() / args.model_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    out = Path(args.output_dir).expanduser() if args.output_dir else checkpoint.parent / 'spatial_sensitivity'
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint, device, override_steps=None)
    if not 1 <= args.step <= model.steps:
        raise ValueError(f'--step must be between 1 and {model.steps}')
    if any(n < 0 or n >= model.num_neurons for n in args.neurons):
        raise ValueError(f'Neuron IDs must be in [0,{model.num_neurons - 1}]')
    if args.top_k < 1 or args.max_images < 1:
        raise ValueError('top-k and max-images must be positive')
    ds = datasets.CIFAR10(args.data_dir, train=False, download=True,
                          transform=transforms.ToTensor())
    ds = torch.utils.data.Subset(ds, range(min(args.max_images, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'Checkpoint: {checkpoint}\nDevice: {device}; recurrent step: {args.step}')
    print('Finding highest-activating real images...', flush=True)
    chosen = select_images(model, loader, args.neurons, args.step, args.top_k, device)

    # One row per neuron-image pair, three columns: image / saliency / overlay.
    rows = [(n, score, index, label) for n in args.neurons
            for score, index, label in chosen[n]]
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, max(3, 2.45 * len(rows))),
                             squeeze=False)
    for row, (neuron, score, index, label) in enumerate(rows):
        image, _ = ds[index]
        rgb = image.permute(1, 2, 0).numpy()
        sal = gradient_map(model, image, neuron, args.step, device,
                           args.method, args.smooth_samples, args.smooth_sigma)
        sal = normalize_saliency(sal)
        axes[row, 0].imshow(rgb, interpolation='nearest')
        axes[row, 1].imshow(sal, cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
        axes[row, 2].imshow(rgb, interpolation='nearest')
        axes[row, 2].imshow(sal, cmap='inferno', vmin=0, vmax=1,
                            alpha=args.overlay_alpha * sal, interpolation='nearest')
        axes[row, 0].set_ylabel(f'N{neuron} | {CLASSES[label]}\na={score:.2f}',
                                fontsize=9, rotation=0, labelpad=55, va='center')
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        print(f'Neuron {neuron}, test image {index}, class {CLASSES[label]}', flush=True)
    for ax, name in zip(axes[0], ['Original CIFAR-10 image', 'Gradient sensitivity', 'Overlay']):
        ax.set_title(name, fontsize=12)
    fig.suptitle(f'FlowGenesis CIFAR-10 — spatial sensitivity at recurrent step {args.step}\n'
                 f'{args.method} (per-image 99th percentile scaling)', fontsize=13)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.01, hspace=0.25, wspace=0.12)
    target = out / f'spatial_sensitivity_step{args.step}_{args.method}.png'
    fig.savefig(target, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {target}')


if __name__ == '__main__':
    main()
