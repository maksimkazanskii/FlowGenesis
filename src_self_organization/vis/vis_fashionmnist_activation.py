#!/usr/bin/env python3
"""Fashion-MNIST activation maximization for saved recurrent FlowGenesis models.

Run from repository root:
python -m src_self_organization.vis.vis_fashionmnist_activation \
  --folder /Users/maksimkazanskii2/Desktop/interferenceNN/data/neurogenesis/fashionmnist/neurons_160/seed_1 \
  --model-name final_model.pt

Images are optimized near high-activating REAL Fashion-MNIST examples.
"""
import argparse
import csv
import math
from pathlib import Path
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def symbol(name, modules):
    failures = []
    for module in modules:
        try:
            return getattr(import_module(module), name)
        except (ImportError, AttributeError) as exc:
            failures.append(f'{module}: {exc}')
    raise ImportError('\n'.join(failures))


def checkpoint_path(args):
    if args.checkpoint:
        p = Path(args.checkpoint).expanduser()
    else:
        p = Path(args.folder).expanduser() / args.model_name
    if not p.is_file():
        raise FileNotFoundError(f'Checkpoint does not exist: {p}')
    return p.resolve()


def load_model(path, device, steps):
    Config = symbol('Config', ['src_self_organization.config_fm'])
    NeuralContainer = symbol('NeuralContainer', [
        'src_self_organization.model_neurogenesis_new',
        'src_self_organization.model_neurogenesis'])
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt['state_dict']
    cfg = Config()
    saved = ckpt.get('config', ckpt.get('cfg'))
    if saved is not None:
        entries = saved.items() if isinstance(saved, dict) else vars(saved).items()
        for k, v in entries:
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    else:
        print('WARNING: checkpoint does not contain training config; using config_fm defaults.')
    cfg.hidden_neurons = int(state['W'].shape[0])
    cfg.input_dim = int(state['input_layer.weight'].shape[1])
    cfg.output_dim = int(state['output_layer.weight'].shape[0])
    cfg.dataset = 'fashionmnist'
    if cfg.input_dim != 784:
        raise ValueError(f'Expected 784-dimensional Fashion-MNIST input, got {cfg.input_dim}')
    if steps is not None:
        cfg.relaxation_steps = steps
    model = NeuralContainer(cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()  # prevents modification of flux statistics during visualization
    for param in model.parameters():
        param.requires_grad_(False)
    print(f'Loaded {path}\nNeurons={model.num_neurons}; relaxation steps={model.steps}; device={device}')
    return model


def get_seeds(model, args, device):
    """Rank real examples by final recurrent neuron activation; estimate neuron scales."""
    ds = datasets.FashionMNIST(args.data_dir, train=True, download=True,
                               transform=transforms.ToTensor())
    if args.probe_count > 0 and args.probe_count < len(ds):
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(ds), size=args.probe_count, replace=False).tolist()
        ds = torch.utils.data.Subset(ds, indices)
    loader = DataLoader(ds, batch_size=args.probe_batch_size, shuffle=False, num_workers=0)
    H, X = [], []
    with torch.no_grad():
        for images, _ in loader:
            x = images.flatten(1).to(device)
            H.append(model.relax(x).cpu())
            X.append(x.cpu())
    H = torch.cat(H, dim=0)
    X = torch.cat(X, dim=0)
    # 95th percentile is a robust activation scale; zero-scale neurons are handled separately.
    scale = torch.quantile(H, 0.95, dim=0).clamp_min(1e-6)
    k = min(args.restarts, len(X))
    best = torch.topk(H, k=k, dim=0).indices.T  # [neurons, restarts]
    print(f'Probe images={len(X)}; inactive on probe={(H.max(dim=0).values <= 1e-8).sum().item()}')
    return X[best].reshape(-1, 784).to(device), scale.to(device), H.max(dim=0).values.numpy()


def total_variation(image):
    return ((image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
            + (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(dim=(1, 2, 3)))


def optimize(model, seeds, scale, args, device):
    n = model.num_neurons
    k = args.restarts
    output = np.zeros((n, 28, 28), dtype=np.float32)
    original = np.zeros_like(output)
    activation_out = np.zeros(n, dtype=np.float32)
    objective_out = np.full(n, -np.inf, dtype=np.float32)
    for begin in range(0, n, args.neurons_per_batch):
        end = min(n, begin + args.neurons_per_batch)
        neuron_ids = torch.arange(begin, end, device=device).repeat_interleave(k)
        x0 = seeds[begin * k:end * k].clone()
        # Optimize a bounded perturbation of real images, rather than arbitrary pixels.
        delta = torch.nn.Parameter(torch.zeros_like(x0))
        opt = torch.optim.Adam([delta], lr=args.lr)
        rows = torch.arange(len(neuron_ids), device=device)
        for _ in range(args.steps):
            opt.zero_grad(set_to_none=True)
            x = (x0 + args.max_change * torch.tanh(delta)).clamp(0.0, 1.0)
            h = model.relax(x)
            act = h[rows, neuron_ids]
            image = x.view(-1, 1, 28, 28)
            tv = total_variation(image)
            deviation = ((x - x0) ** 2).mean(dim=1)
            obj = act / scale[neuron_ids] - args.tv * tv - args.anchor * deviation
            (-obj.mean()).backward()
            opt.step()
        with torch.no_grad():
            x = (x0 + args.max_change * torch.tanh(delta)).clamp(0.0, 1.0)
            act = model.relax(x)[rows, neuron_ids]
            tv = total_variation(x.view(-1, 1, 28, 28))
            deviation = ((x - x0) ** 2).mean(dim=1)
            score = act / scale[neuron_ids] - args.tv * tv - args.anchor * deviation
            for neuron in range(begin, end):
                local = (neuron - begin) * k
                best = local + int(torch.argmax(score[local:local + k]).item())
                output[neuron] = x[best].view(28, 28).cpu().numpy()
                original[neuron] = x0[best].view(28, 28).cpu().numpy()
                activation_out[neuron] = float(act[best].item())
                objective_out[neuron] = float(score[best].item())
        print(f'Optimized {end}/{n} neurons', flush=True)
    return output, original, activation_out, objective_out


def plot_grid(images, scores, output, cols, dpi, title):
    n = len(images)
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.7), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i < n:
            ax.imshow(images[i], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(f'Neuron {i}\na={scores[i]:.2f}', fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint')
    group.add_argument('--folder')
    p.add_argument('--model-name', choices=['best_model.pt', 'final_model.pt'], default='final_model.pt')
    p.add_argument('--output-dir')
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    p.add_argument('--relaxation-steps', type=int)
    p.add_argument('--probe-count', type=int, default=5000, help='Real training images used to find starting images')
    p.add_argument('--probe-batch-size', type=int, default=256)
    p.add_argument('--restarts', type=int, default=3, help='Top real images used per neuron')
    p.add_argument('--neurons-per-batch', type=int, default=8)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--lr', type=float, default=0.06)
    p.add_argument('--max-change', type=float, default=0.25, help='Maximum absolute pixel change from real image')
    p.add_argument('--tv', type=float, default=0.5, help='Total variation penalty on optimized image')
    p.add_argument('--anchor', type=float, default=3.0, help='Penalty on squared deviation from original image')
    p.add_argument('--cols', type=int, default=8)
    p.add_argument('--dpi', type=int, default=160)
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()
    if min(args.restarts, args.steps, args.neurons_per_batch, args.probe_batch_size, args.cols) <= 0:
        p.error('restarts, steps, neurons-per-batch, probe-batch-size, cols must be positive')
    if not 0 < args.max_change <= 1:
        p.error('max-change must be in (0,1]')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    path = checkpoint_path(args)
    out = Path(args.output_dir) if args.output_dir else path.parent / 'activation_max_regularized'
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(path, device, args.relaxation_steps)
    seeds, scale, probe_max = get_seeds(model, args, device)
    images, originals, activations, scores = optimize(model, seeds, scale, args, device)
    plot_grid(images, activations, out / 'activation_max_grid.png', args.cols, args.dpi,
              'FlowGenesis — constrained neuron activation maximization')
    plot_grid(originals, activations, out / 'starting_images_grid.png', args.cols, args.dpi,
              'FlowGenesis — original Fashion-MNIST starting images')
    np.savez_compressed(out / 'activation_max.npz', images=images, starting_images=originals,
                        activations=activations, objectives=scores, probe_max=probe_max,
                        activation_scale=scale.cpu().numpy())
    with (out / 'activation_max_scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['neuron', 'optimized_activation', 'probe_max_activation', 'regularized_objective'])
        writer.writerows((i, float(activations[i]), float(probe_max[i]), float(scores[i])) for i in range(len(images)))
    print(f'Output: {out.resolve()}')


if __name__ == '__main__':
    main()
