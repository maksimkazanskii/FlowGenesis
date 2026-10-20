"""Continuous Rotating MNIST: matched-growth Flux vs Activity vs Static.

Run from repository root:
 python -u src_self_organization/train_rotating_mnist.py \\
   --model-file src_self_organization/model_neurogenesis_new.py \\
   --seeds 0 1 2 3 4 --cycles 3

The dynamic policies share *exactly the same growth event schedule* within each
seed: at each structural window, Flux's threshold/persistence determines the
number of births; Flux selects parents by flux, Activity by activity. This
isolates the neuron-selection signal rather than growth timing/amount.
Pruning is disabled for this controlled experiment. A stationary stream is
available with --control stationary.
"""
from __future__ import annotations
import argparse, copy, csv, importlib.util, json, math, random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST


def load_model(path):
    spec = importlib.util.spec_from_file_location('flowgenesis_model_rotating', path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.NeuralContainer


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def rotate_batch(x, angles):
    # Rotate tensor on selected device, using identical transformation for all methods.
    radians = angles * (math.pi / 180.0)
    c, s = radians.cos(), radians.sin()
    theta = torch.zeros(x.size(0), 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = c; theta[:, 0, 1] = -s
    theta[:, 1, 0] = s; theta[:, 1, 1] = c
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def balanced_order(labels, seed):
    # Exactly 10 samples per class per 100 examples when possible; deterministic.
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(labels == cls)) for cls in range(10)]
    n = min(len(g) for g in groups)
    return np.stack([g[:n] for g in groups], axis=1).reshape(-1)


def angle_at(indices, samples_per_cycle, control):
    if control == 'stationary':
        return torch.zeros_like(indices, dtype=torch.float32)
    phase = (indices % samples_per_cycle).float() / samples_per_cycle
    return 180.0 * (1.0 - torch.abs(2.0 * phase - 1.0))


def make_cfg(args, method):
    return SimpleNamespace(
        input_dim=784, output_dim=10, hidden_neurons=args.initial_neurons,
        relaxation_steps=args.relaxation_steps,
        recurrent_connection_probability=args.connection_probability,
        self_connections=False,
        use_neurogenesis=(method != 'static'), use_pruning=False,
        birth_threshold=args.birth_threshold,
        birth_persistence=args.birth_persistence,
        death_persistence=10**12,
        max_births_per_epoch=args.max_births_per_window,
        max_deaths_per_epoch=0,
        min_neurons=args.initial_neurons, max_neurons=args.max_neurons,
        structure_warmup_examples=args.warmup_examples,
        structure_cooldown_examples=args.cooldown_examples,
        split_noise_std=args.split_noise_std,
        flux_decay=0.90, activity_decay=0.90, capacity_decay=0.99,
    )


def save_rows(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)


def run_seed(args, seed, NeuralContainer, images, labels, device):
    seed_everything(seed)
    # Initialize all methods from the exact same parameters.
    reference = NeuralContainer(make_cfg(args, 'flux')).to(device)
    models = {'flux': reference, 'activity': copy.deepcopy(reference), 'static': copy.deepcopy(reference)}
    for method, model in models.items():
        model.use_neurogenesis = method != 'static'
        model.use_pruning = False
        model.train()
    optimizers = {m: torch.optim.Adam(model.parameters(), lr=args.lr) for m, model in models.items()}
    order = balanced_order(labels, seed + 12345)
    samples_per_cycle = len(order)
    rows = []
    total_correct = {m: 0 for m in models}
    total_seen = {m: 0 for m in models}
    weighted_neurons = {m: 0 for m in models}
    # Use one deterministic digit order in each cycle. The exact same transformed
    # mini-batches are presented to all three methods.
    global_step = 0
    for cycle in range(args.cycles):
        for offset in range(0, samples_per_cycle, args.batch_size):
            indices = order[offset:offset + args.batch_size]
            if len(indices) == 0: continue
            global_indices = torch.arange(cycle * samples_per_cycle + offset,
                                          cycle * samples_per_cycle + offset + len(indices), device=device)
            angles = angle_at(global_indices, samples_per_cycle, args.control)
            x = images[indices].to(device, non_blocking=True)
            y = labels[indices].to(device, non_blocking=True)
            x = rotate_batch(x, angles)
            window_trigger = False
            # Update one batch for each method, with pre-update prediction.
            for method, model in models.items():
                optimizer = optimizers[method]
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                correct = int((logits.argmax(dim=1) == y).sum().item())
                n = len(indices)
                neurons_before = model.num_neurons
                total_correct[method] += correct
                total_seen[method] += n
                weighted_neurons[method] += neurons_before * n
                loss.backward(); optimizer.step()
                rows.append(dict(seed=seed, method=method, cycle=cycle + 1,
                                 step=global_step, examples_seen=total_seen[method],
                                 angle_deg=float(angles.mean().item()),
                                 batch_accuracy=100 * correct / n,
                                 cumulative_preq_accuracy=100 * total_correct[method] / total_seen[method],
                                 loss=float(loss.item()), neurons=neurons_before,
                                 births=0, mean_flux=float(model.flux.mean().item()),
                                 mean_activity=float(model.activity.mean().item())))
            # Structural update after complete fixed-size example windows.
            window_trigger = (models['flux'].window_examples >= args.structure_window)
            if window_trigger:
                # end_epoch updates flux/activity streaks in the supplied model.
                stats = {m: model.end_epoch() for m, model in models.items()}
                flux = models['flux']
                # Use the same flux-derived birth schedule for both methods;
                # change only which neuron is selected for splitting.
                eligible = torch.where(flux.birth_streak >= flux.birth_persistence)[0]
                warm = (flux.examples_seen >= flux.structure_warmup_examples and
                        flux.examples_seen - flux.last_structure_examples >= flux.structure_cooldown_examples)
                births = min(args.max_births_per_window, len(eligible),
                             args.max_neurons - flux.num_neurons) if warm else 0
                if births > 0:
                    # Select from all neurons for both policies, ensuring identical
                    # birth counts/timing, with ranking as the only intervention.
                    for method in ('flux', 'activity'):
                        model = models[method]
                        score = stats[method]['mean_flux'] if method == 'flux' else stats[method]['mean_activity']
                        parents = torch.argsort(score, descending=True)[:births].tolist()
                        for parent in parents:
                            model._split_neuron(int(parent))
                        model.last_structure_examples = model.examples_seen
                        # Parameter objects change after splitting.
                        optimizers[method] = torch.optim.Adam(model.parameters(), lr=args.lr)
                        rows[-(3 - list(models).index(method))]['births'] = births
                    # Reset triggering streaks after growth; avoid perpetually
                    # triggering on the same persistent overload.
                    for method in ('flux', 'activity'):
                        models[method].birth_streak.zero_()
                if args.verbose:
                    print(f'[seed={seed} cycle={cycle+1} examples={flux.examples_seen}] '
                          f'flux={stats["flux"]["network_mean_flux"]:.4f} '
                          f'births={births} N={flux.num_neurons}', flush=True)
            global_step += 1
        print(f'[seed={seed} cycle={cycle+1}/{args.cycles} control={args.control}] ' +
              ' | '.join(f'{m}: preq={100*total_correct[m]/total_seen[m]:.2f}% '
                         f'avgN={weighted_neurons[m]/total_seen[m]:.1f} finalN={models[m].num_neurons}'
                         for m in models), flush=True)
        save_rows(args.output / args.control / f'seed_{seed}' / 'stream.csv', rows,
                  list(rows[0].keys()))
    summary = []
    for method in models:
        summary.append(dict(seed=seed, control=args.control, method=method,
                            prequential_accuracy=100*total_correct[method]/total_seen[method],
                            average_neurons=weighted_neurons[method]/total_seen[method],
                            final_neurons=models[method].num_neurons))
    save_rows(args.output / args.control / f'seed_{seed}' / 'summary.csv', summary,
              list(summary[0].keys()))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model-file', type=Path, default=Path('src_self_organization/model_neurogenesis_new.py'))
    p.add_argument('--data-root', default='./data')
    p.add_argument('--output', type=Path, default=Path('data/neurogenesis/rotating_mnist'))
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4])
    p.add_argument('--cycles', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--device', default='auto')
    p.add_argument('--control', choices=['rotating', 'stationary'], default='rotating')
    p.add_argument('--initial-neurons', type=int, default=80)
    p.add_argument('--max-neurons', type=int, default=160)
    p.add_argument('--relaxation-steps', type=int, default=10)
    p.add_argument('--connection-probability', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--structure-window', type=int, default=5000)
    p.add_argument('--warmup-examples', type=int, default=10000)
    p.add_argument('--cooldown-examples', type=int, default=5000)
    p.add_argument('--birth-threshold', type=float, default=0.25)
    p.add_argument('--birth-persistence', type=int, default=10000)
    p.add_argument('--max-births-per-window', type=int, default=2)
    p.add_argument('--split-noise-std', type=float, default=0.01)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    if args.batch_size % 10: p.error('--batch-size must be divisible by 10 for balanced batches')
    if args.max_neurons < args.initial_neurons: p.error('max neurons must be >= initial neurons')
    device = torch.device(('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    print('device:', device, flush=True)
    ds = MNIST(args.data_root, train=True, download=True)
    images = ds.data.unsqueeze(1).float().div_(255.0)
    labels = ds.targets.long()
    NeuralContainer = load_model(args.model_file)
    all_summaries = []
    for seed in args.seeds:
        all_summaries.extend(run_seed(args, seed, NeuralContainer, images, labels, device))
        save_rows(args.output / args.control / 'all_seeds.csv', all_summaries,
                  list(all_summaries[0].keys()))
    print('Saved:', args.output / args.control / 'all_seeds.csv')

if __name__ == '__main__':
    main()
