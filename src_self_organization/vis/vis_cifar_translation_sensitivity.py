#!/usr/bin/env python3
"""FlowGenesis CIFAR-10: does neuron sensitivity move with translated images?

Put next to vis_cifar_spatial_sensitivity.py and vis_cifar_selectivity.py.
Uses saved checkpoint and top-activating real CIFAR-10 test images.
No model retraining. Zero-fill translations, without wraparound.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src_self_organization.vis.vis_cifar_selectivity import load_model, CLASSES
from src_self_organization.vis.vis_cifar_spatial_sensitivity import (
    select_images, gradient_map, normalize_saliency,
)


def translate(tensor, dx, dy):
    """Translate CHW or HW tensor, filling new pixels with zero."""
    height, width = tensor.shape[-2:]
    result = torch.zeros_like(tensor)
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        result[..., src_y0 + dy:src_y1 + dy, src_x0 + dx:src_x1 + dx] = (
            tensor[..., src_y0:src_y1, src_x0:src_x1]
        )
    return result


def valid_mask(height, width, dx, dy):
    """Region occupied by translated source pixels."""
    return translate(torch.ones(height, width), dx, dy).numpy() > 0


def corr(a, b, mask):
    a = np.asarray(a)[mask].astype(np.float64)
    b = np.asarray(b)[mask].astype(np.float64)
    if a.size < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def neuron_response(model, image, neuron, step, device):
    # No-gradient only for scalar activation evaluation, not saliency.
    from src_self_organization.vis.vis_cifar_spatial_sensitivity import differentiable_states_at_steps
    with torch.no_grad():
        x = image.unsqueeze(0).to(device).flatten(1)
        return float(differentiable_states_at_steps(model, x, [step])[step][0, neuron].item())


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
    p.add_argument('--shifts', type=int, nargs='+', default=[-4, 0, 4],
                   help='Horizontal translations in pixels (dy=0). Include 0.')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--max-images', type=int, default=10000)
    p.add_argument('--method', choices=['abs', 'smoothgrad'], default='smoothgrad')
    p.add_argument('--smooth-samples', type=int, default=16)
    p.add_argument('--smooth-sigma', type=float, default=0.08)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dpi', type=int, default=160)
    args = p.parse_args()

    if 0 not in args.shifts:
        p.error('--shifts must include 0')
    if len(set(args.shifts)) != len(args.shifts) or any(abs(s) >= 32 for s in args.shifts):
        p.error('Shifts must be unique and have magnitude < 32')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'auto':
        name = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        name = args.device
    device = torch.device(name)
    checkpoint = Path(args.folder).expanduser() / args.model_name
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    out = Path(args.output_dir).expanduser() if args.output_dir else checkpoint.parent / 'translation_sensitivity'
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint, device, override_steps=None)
    model.eval()
    if not 1 <= args.step <= model.steps:
        p.error(f'--step must be between 1 and {model.steps}')
    if any(n < 0 or n >= model.num_neurons for n in args.neurons):
        p.error(f'Neuron IDs must be between 0 and {model.num_neurons - 1}')

    ds_full = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transforms.ToTensor())
    ds = Subset(ds_full, range(min(args.max_images, len(ds_full))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print('Selecting highest-activating test images...', flush=True)
    chosen = select_images(model, loader, args.neurons, args.step, args.top_k, device)

    rows = []
    shifts = sorted(args.shifts)
    for neuron in args.neurons:
        for _, idx, label in chosen[neuron]:
            image, _ = ds[idx]
            base = gradient_map(model, image, neuron, args.step, device,
                                args.method, args.smooth_samples, args.smooth_sigma)
            base_score = neuron_response(model, image, neuron, args.step, device)
            fig, axes = plt.subplots(2, len(shifts), figsize=(3.1 * len(shifts), 6.5), squeeze=False)
            for col, dx in enumerate(shifts):
                shifted = translate(image, dx, 0)
                sal = base if dx == 0 else gradient_map(
                    model, shifted, neuron, args.step, device,
                    args.method, args.smooth_samples, args.smooth_sigma)
                score = neuron_response(model, shifted, neuron, args.step, device)
                # Compare normalized spatial patterns on their overlapping valid regions.
                # Following-object comparison aligns translated sensitivity back to source coords.
                mask = valid_mask(32, 32, dx, 0)
                moved_base = translate(torch.from_numpy(base), dx, 0).numpy()
                follow_corr = corr(sal, moved_base, mask)
                fixed_corr = corr(sal, base, mask)
                rows.append(dict(neuron=neuron, image_index=idx, class_name=CLASSES[label],
                                 dx=dx, baseline_activation=base_score, activation=score,
                                 activation_ratio=score / base_score if abs(base_score) > 1e-10 else float('nan'),
                                 object_aligned_corr=follow_corr, fixed_coordinate_corr=fixed_corr))
                rgb = shifted.permute(1, 2, 0).numpy()
                axes[0, col].imshow(rgb, interpolation='nearest')
                axes[0, col].set_title(f'dx={dx:+d} px | activation={score:.2f}', fontsize=10)
                axes[1, col].imshow(normalize_saliency(sal), cmap='inferno', vmin=0, vmax=1,
                                    interpolation='nearest')
                axes[1, col].set_title(f'follow={follow_corr:.2f} | fixed={fixed_corr:.2f}', fontsize=9)
                for ax in (axes[0, col], axes[1, col]):
                    ax.set_xticks([])
                    ax.set_yticks([])
            axes[0, 0].set_ylabel('Translated image')
            axes[1, 0].set_ylabel('Gradient sensitivity')
            fig.suptitle(f'FlowGenesis neuron {neuron} | CIFAR-10 {CLASSES[label]} | test #{idx}\n'
                         f'Step {args.step} | {args.method} | baseline activation={base_score:.2f}', fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, 0.90))
            target = out / f'neuron_{neuron:03d}_image_{idx:05d}_translation.png'
            fig.savefig(target, dpi=args.dpi)
            plt.close(fig)
            print(f'Saved {target}', flush=True)

    csv_path = out / 'translation_metrics.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {csv_path}')
    print('Interpretation: follow > fixed suggests sensitivity moves with image; '
          'fixed > follow suggests position-anchored sensitivity. Both are descriptive, '
          'not proof of object detection. Zero padding can introduce border artifacts.')


if __name__ == '__main__':
    main()
