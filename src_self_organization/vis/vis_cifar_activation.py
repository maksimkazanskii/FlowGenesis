#!/usr/bin/env python3
"""CIFAR-10 activation maximization at several FlowGenesis recurrent time steps.

Run from repository root:
python -m src_self_organization.vis.vis_cifar_activation_steps \
 --folder data/neurogenesis/cifar10/so/n0_ablation/cifar10/neurons_200/seed_2 \
 --model-name final_model.pt

Each step has its OWN top activating real CIFAR-10 starts and activation scale.
The recurrent weights are shared, and gradients pass through the unrolled steps.
"""
import argparse
import csv
import math
from importlib import import_module
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def symbol(name, modules):
    failures = []
    for module in modules:
        try:
            return getattr(import_module(module), name)
        except (ImportError, AttributeError) as exc:
            failures.append(f'{module}: {exc}')
    raise ImportError('\n'.join(failures))


def load_model(path, device, override_steps):
    Config = symbol('Config', ['src_self_organization.config_fm'])
    NeuralContainer = symbol('NeuralContainer', [
        'src_self_organization.model_neurogenesis_new',
        'src_self_organization.model_neurogenesis'])
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint['state_dict']
    cfg = Config()
    saved = checkpoint.get('config', checkpoint.get('cfg'))
    if saved is not None:
        entries = saved.items() if isinstance(saved, dict) else vars(saved).items()
        for k, v in entries:
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    else:
        print('WARNING: checkpoint has no config. Verify relaxation_steps in config_fm.')
    cfg.hidden_neurons = int(state['W'].shape[0])
    cfg.input_dim = int(state['input_layer.weight'].shape[1])
    cfg.output_dim = int(state['output_layer.weight'].shape[0])
    if cfg.input_dim != 3072:
        raise ValueError(f'Expected CIFAR-10 3072 input features; found {cfg.input_dim}')
    if override_steps is not None:
        cfg.relaxation_steps = override_steps
    model = NeuralContainer(cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def hidden_at_steps(model, x, requested):
    """Exactly reproduce model.relax recurrence, collecting requested states.

    Does not call model.relax, so no flux/statistics are changed.
    """
    input_drive = model.input_layer(x)
    h = torch.zeros_like(input_drive)
    recurrent_weight = (model.W * model.mask) / model.degree
    found = {}
    for t in range(1, max(requested) + 1):
        candidate = F.relu(input_drive + F.linear(h, recurrent_weight, model.bias))
        h = 0.8 * h + 0.2 * candidate
        if t in requested:
            found[t] = h
    return found


def tv_per_image(images):
    return (images[:, :, 1:, :].sub(images[:, :, :-1, :]).abs().mean((1, 2, 3)) +
            images[:, :, :, 1:].sub(images[:, :, :, :-1]).abs().mean((1, 2, 3)))


def probe(model, requested, args, device):
    ds = datasets.CIFAR10(args.data_dir, train=True, download=True,
                          transform=transforms.ToTensor())
    if args.probe_count and args.probe_count < len(ds):
        rng = np.random.default_rng(args.seed)
        ds = Subset(ds, rng.choice(len(ds), args.probe_count, replace=False).tolist())
    loader = DataLoader(ds, batch_size=args.probe_batch_size, shuffle=False, num_workers=0)
    images_all = []
    activations = {t: [] for t in requested}
    with torch.no_grad():
        for images, _ in loader:
            x = images.flatten(1).to(device)
            hs = hidden_at_steps(model, x, requested)
            images_all.append(x.cpu())
            for t in requested:
                activations[t].append(hs[t].cpu())
    X = torch.cat(images_all)
    seeds, scales, maxima = {}, {}, {}
    for t in requested:
        H = torch.cat(activations[t])
        k = min(args.restarts, len(X))
        best = torch.topk(H, k=k, dim=0).indices.T
        seeds[t] = X[best].reshape(-1, 3072).to(device)
        scales[t] = torch.quantile(H, 0.95, dim=0).clamp_min(1e-6).to(device)
        maxima[t] = H.max(dim=0).values.numpy()
        print(f'Step {t}: {len(X)} probe images, {(H.max(0).values <= 1e-8).sum().item()} inactive neurons')
    return seeds, scales, maxima, k


def optimize_step(model, t, seeds, scale, args, device, k):
    n = model.num_neurons
    optimized = np.zeros((n, 32, 32, 3), dtype=np.float32)
    originals = np.zeros_like(optimized)
    optimized_act = np.zeros(n, dtype=np.float32)
    original_act = np.zeros(n, dtype=np.float32)
    for begin in range(0, n, args.neurons_per_batch):
        end = min(n, begin + args.neurons_per_batch)
        neuron_ids = torch.arange(begin, end, device=device).repeat_interleave(k)
        x0 = seeds[begin*k:end*k].clone()
        delta = torch.nn.Parameter(torch.zeros_like(x0))
        opt = torch.optim.Adam([delta], lr=args.lr)
        rows = torch.arange(len(neuron_ids), device=device)
        for _ in range(args.iterations):
            opt.zero_grad(set_to_none=True)
            x = (x0 + args.max_change * torch.tanh(delta)).clamp(0, 1)
            activation = hidden_at_steps(model, x, [t])[t][rows, neuron_ids]
            smooth = tv_per_image(x.reshape(-1, 3, 32, 32))
            deviation = (x - x0).square().mean(dim=1)
            objective = activation / scale[neuron_ids] - args.tv * smooth - args.anchor * deviation
            (-objective.mean()).backward()
            opt.step()
        with torch.no_grad():
            x = (x0 + args.max_change * torch.tanh(delta)).clamp(0, 1)
            a = hidden_at_steps(model, x, [t])[t][rows, neuron_ids]
            a0 = hidden_at_steps(model, x0, [t])[t][rows, neuron_ids]
            score = (a / scale[neuron_ids] - args.tv * tv_per_image(x.reshape(-1, 3, 32, 32))
                     - args.anchor * (x - x0).square().mean(dim=1))
            for neuron in range(begin, end):
                start = (neuron - begin)*k
                best = start + int(score[start:start+k].argmax())
                optimized[neuron] = x[best].reshape(3, 32, 32).permute(1, 2, 0).cpu().numpy()
                originals[neuron] = x0[best].reshape(3, 32, 32).permute(1, 2, 0).cpu().numpy()
                optimized_act[neuron] = float(a[best])
                original_act[neuron] = float(a0[best])
        print(f'Step {t}: optimized {end}/{n} neurons', flush=True)
    return originals, optimized, original_act, optimized_act


def save_comparison(path, results, requested, selected, optimized, dpi):
    # Rows are neurons, columns are recurrent steps; all use same ordering.
    fig, axes = plt.subplots(len(selected), len(requested),
                             figsize=(len(requested)*2.0, len(selected)*2.0),
                             squeeze=False)
    for row, neuron in enumerate(selected):
        for col, t in enumerate(requested):
            original, image, a0, a = results[t]
            ax = axes[row, col]
            ax.imshow((image if optimized else original)[neuron], vmin=0, vmax=1)
            ax.axis('off')
            ax.set_title(f'N{neuron} / t={t}\na={a[neuron] if optimized else a0[neuron]:.2f}', fontsize=9)
    fig.suptitle('FlowGenesis CIFAR-10 — ' + ('optimized images' if optimized else 'real starting images') +
                 ' across recurrent steps', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_step_grid(path, images, acts, t, cols, dpi):
    n = len(images)
    cols = min(cols, n)
    fig, axes = plt.subplots(math.ceil(n/cols), cols,
                             figsize=(cols*1.5, math.ceil(n/cols)*1.7), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i < n:
            ax.imshow(images[i], vmin=0, vmax=1)
            ax.set_title(f'Neuron {i}\na={acts[i]:.2f}', fontsize=8)
    fig.suptitle(f'FlowGenesis CIFAR-10 — step {t}', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--folder')
    group.add_argument('--checkpoint')
    p.add_argument('--model-name', default='final_model.pt')
    p.add_argument('--output-dir')
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--device', choices=['auto','cpu','cuda','mps'], default='auto')
    p.add_argument('--relaxation-steps', type=int, help='Override current config only if training used a different value')
    p.add_argument('--time-steps', type=int, nargs='+', default=None,
                   help='Steps to visualize, e.g. --time-steps 1 5 10 20 (default: 1, middle, final)')
    p.add_argument('--neurons', type=int, nargs='+', default=None,
                   help='Neuron IDs for cross-time comparison; default: 0 3 7 12 20 30 40 50 if present')
    p.add_argument('--all-neurons', action='store_true', help='Compare every neuron in cross-time figure (very tall)')
    p.add_argument('--probe-count', type=int, default=5000)
    p.add_argument('--probe-batch-size', type=int, default=256)
    p.add_argument('--restarts', type=int, default=3)
    p.add_argument('--neurons-per-batch', type=int, default=8)
    p.add_argument('--iterations', type=int, default=300)
    p.add_argument('--lr', type=float, default=.06)
    p.add_argument('--max-change', type=float, default=.25)
    p.add_argument('--tv', type=float, default=.5)
    p.add_argument('--anchor', type=float, default=3.)
    p.add_argument('--cols', type=int, default=8)
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()
    if min(args.restarts, args.iterations, args.probe_batch_size, args.neurons_per_batch, args.cols) < 1:
        p.error('restarts, iterations, probe-batch-size, neurons-per-batch and cols must be positive')
    if args.probe_count < 0 or args.tv < 0 or args.anchor < 0 or not 0 < args.max_change <= 1:
        p.error('invalid regularization or probe settings')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    path = Path(args.checkpoint if args.checkpoint else Path(args.folder)/args.model_name).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    out = Path(args.output_dir).expanduser() if args.output_dir else path.parent/'activation_max_by_step'
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(path, device, args.relaxation_steps)
    S = int(model.steps)
    requested = sorted(set(args.time_steps if args.time_steps else [1, max(1, S//4), max(1, S//2), S]))
    if min(requested) < 1 or max(requested) > S:
        p.error(f'--time-steps must be between 1 and {S}, model relaxation_steps')
    print(f'Checkpoint: {path}\nDevice: {device}\nNeurons: {model.num_neurons}; model steps: {S}; requested: {requested}')
    # Confirm unrolling matches the model implementation at the final step.
    with torch.no_grad():
        test = torch.rand(2, 3072, device=device)
        error = (hidden_at_steps(model, test, [S])[S] - model.relax(test)).abs().max().item()
        if error > 1e-4:
            raise RuntimeError(f'Unrolled dynamics disagree with model.relax: max difference {error}')
    seeds, scales, maxima, k = probe(model, requested, args, device)
    results = {}
    with (out/'activation_scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step','neuron','starting_activation','optimized_activation','probe_max'])
        for t in requested:
            results[t] = optimize_step(model, t, seeds[t], scales[t], args, device, k)
            original, image, a0, a = results[t]
            for neuron in range(model.num_neurons):
                writer.writerow([t, neuron, float(a0[neuron]), float(a[neuron]), float(maxima[t][neuron])])
            save_step_grid(out/f'step_{t:03d}_original.png', original, a0, t, args.cols, args.dpi)
            save_step_grid(out/f'step_{t:03d}_optimized.png', image, a, t, args.cols, args.dpi)
    selected = list(range(model.num_neurons)) if args.all_neurons else (
        args.neurons if args.neurons is not None else
        [i for i in [0, 3, 7, 12, 20, 30, 40, 50] if i < model.num_neurons])
    if not selected or any(i < 0 or i >= model.num_neurons for i in selected):
        p.error('invalid --neurons selection')
    save_comparison(out/'comparison_optimized.png', results, requested, selected, True, args.dpi)
    save_comparison(out/'comparison_original.png', results, requested, selected, False, args.dpi)
    np.savez_compressed(out/'activation_max_by_step.npz', steps=np.asarray(requested),
                        original=np.stack([results[t][0] for t in requested]),
                        optimized=np.stack([results[t][1] for t in requested]),
                        original_activation=np.stack([results[t][2] for t in requested]),
                        optimized_activation=np.stack([results[t][3] for t in requested]))
    print(f'\nSaved results to: {out}')


if __name__ == '__main__':
    main()
