#!/usr/bin/env python3
"""Fashion-MNIST structural-damage recovery: flux, activity, weight-only.

Uses the supplied NeuralContainer unchanged. Each seed shares one trained
checkpoint and one neuron-deletion mask across the three recovery methods.
"""
import argparse
import copy
import csv
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_model_class(path):
    spec = importlib.util.spec_from_file_location('flowgenesis_model', str(path))
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.NeuralContainer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loaders(args, seed):
    tfm = transforms.ToTensor()
    train = datasets.FashionMNIST(args.data_dir, train=True, download=True, transform=tfm)
    test = datasets.FashionMNIST(args.data_dir, train=False, download=True, transform=tfm)
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=g,
                       num_workers=args.workers),
            DataLoader(test, batch_size=args.batch_size * 2, shuffle=False,
                       num_workers=args.workers))


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return 100 * correct / total, loss_sum / total


def train_epoch(model, loader, optimizer, device):
    model.train()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    model.current_epoch += 1
    stats = model.end_epoch()
    return 100 * correct / total, stats


def reset_structure_statistics(model):
    # Damage causes a discontinuity in neuron identities and hidden states.
    for name in ('previous_state', 'flux', 'flux_ema', 'activity', 'activity_ema',
                 'capacity', 'epoch_flux_sum', 'epoch_activity_sum',
                 'birth_streak', 'death_streak'):
        getattr(model, name).zero_()
    model.epoch_batch_count = 0
    model.window_examples = 0
    model.examples_seen = 0
    model.last_structure_examples = -10**15
    model.current_epoch = 0
    model.flux_history.clear()
    model.activity_history.clear()
    model.capacity_history.clear()
    model.activation_history.clear()
    model.neuron_count_history.clear()
    model.birth_history.clear()
    model.death_history.clear()


def damage(model, fraction, seed):
    n = model.num_neurons
    count = max(1, round(n * fraction))
    if n - count < 2:
        raise ValueError('Too few neurons remain after damage')
    rng = np.random.default_rng(seed)
    removed = sorted(rng.choice(n, size=count, replace=False).tolist(), reverse=True)
    # Temporarily relax min_neurons to permit the externally imposed lesion.
    previous_min = model.min_neurons
    model.min_neurons = 1
    for index in removed:
        model._prune_neuron(int(index))
    model.min_neurons = min(previous_min, model.num_neurons)
    reset_structure_statistics(model)
    return removed


def growth_step(model, method, args):
    if method == 'weight_only':
        return 0
    if model.examples_seen < args.recovery_warmup_examples:
        return 0
    if model.examples_seen - model.last_structure_examples < args.cooldown_examples:
        return 0
    if model.num_neurons >= model.max_neurons:
        return 0

    if method == 'flux':
        candidates = torch.where(model.birth_streak >= model.birth_persistence)[0]
        scores = model.flux_ema
    elif method == 'activity':
        # Activity-only ablation: replace flux overload with persistently high
        # activity relative to the current population, with identical persistence.
        candidates = torch.where(model.birth_streak >= model.birth_persistence)[0]
        scores = model.activity_ema
    else:
        raise ValueError(method)
    if candidates.numel() == 0:
        return 0
    chosen = candidates[torch.argsort(scores[candidates], descending=True)]
    limit = min(args.max_births_per_epoch, len(chosen), model.max_neurons - model.num_neurons)
    parents = chosen[:limit].tolist()
    for p in parents:
        model._split_neuron(int(p))
    if parents:
        model.last_structure_examples = model.examples_seen
    return len(parents)



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-file', type=Path, default=Path(__file__).with_name('model_neurogenesis_new(20261018-190934).py'))
    p.add_argument('--out', type=Path, default=Path('data/neurogenesis/structural_recovery'))
    p.add_argument('--data-dir', default='data')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--pretrain-epochs', type=int, default=30)
    p.add_argument('--recovery-epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--initial-neurons', type=int, default=80)
    p.add_argument('--max-neurons', type=int, default=300)
    p.add_argument('--min-neurons', type=int, default=20)
    p.add_argument('--relaxation-steps', type=int, default=5)
    p.add_argument('--connection-probability', type=float, default=0.2)
    p.add_argument('--birth-threshold', type=float, default=0.25)
    p.add_argument('--birth-persistence', type=int, default=3)
    p.add_argument('--max-births-per-epoch', type=int, default=2)
    p.add_argument('--damage-fraction', type=float, nargs='+', default=[0.30])
    p.add_argument('--recovery-warmup-examples', type=int, default=0)
    p.add_argument('--cooldown-examples', type=int, default=1000)
    p.add_argument('--activity-quantile', type=float, default=0.75)
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    if any(not (0 < x < 1) for x in args.damage_fraction):
        p.error('--damage-fraction values must be between 0 and 1')
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    NeuralContainer = load_model_class(args.model_file)
    cfg = SimpleNamespace(input_dim=784, output_dim=10,
                          hidden_neurons=args.initial_neurons, relaxation_steps=args.relaxation_steps,
                          recurrent_connection_probability=args.connection_probability,
                          self_connections=False, use_neurogenesis=True, use_pruning=True,
                          birth_threshold=args.birth_threshold, birth_persistence=args.birth_persistence,
                          max_births_per_epoch=args.max_births_per_epoch, max_deaths_per_epoch=2,
                          min_neurons=args.min_neurons, max_neurons=args.max_neurons,
                          structure_warmup_examples=5000, structure_cooldown_examples=args.cooldown_examples)
    with (args.out / 'config.json').open('w') as f:
        json.dump({**vars(args), 'model_file': str(args.model_file), 'out': str(args.out)}, f, indent=2, default=str)

    methods = ('flux', 'activity', 'weight_only')
    for seed in args.seeds:
        seed_everything(seed)
        train_loader, test_loader = loaders(args, seed)
        model = NeuralContainer(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        seed_dir = args.out / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f'[seed={seed}] pretraining', flush=True)
        for epoch in range(args.pretrain_epochs):
            train_acc, _ = train_epoch(model, train_loader, optimizer, device)
            upd = model.apply_structural_updates()
            if upd.changed:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            test_acc, _ = evaluate(model, test_loader, device)
            print(f'[seed={seed}] pretrain {epoch+1} train={train_acc:.2f} test={test_acc:.2f} N={model.num_neurons}', flush=True)
        before_acc, _ = evaluate(model, test_loader, device)
        torch.save({'cfg': vars(cfg), 'state_dict': model.state_dict(), 'neurons': model.num_neurons,
                    'pre_damage_accuracy': before_acc}, seed_dir / 'pre_damage.pt')
        for fraction in args.damage_fraction:
            damage_dir = seed_dir / f'damage_{fraction:.2f}'
            damage_dir.mkdir(parents=True, exist_ok=True)
            # Deepcopy preserves all dynamic shape-dependent buffers and histories.
            damaged = copy.deepcopy(model)
            removed = damage(damaged, fraction, seed + 100000 + int(round(fraction * 10000)))
            damaged_acc, _ = evaluate(damaged, test_loader, device)
            print(f'[seed={seed} damage={fraction:.2f}] lesion N={model.num_neurons}->{damaged.num_neurons} acc={before_acc:.2f}->{damaged_acc:.2f}', flush=True)
            torch.save({'state_dict': damaged.state_dict(), 'neurons': damaged.num_neurons,
                        'removed': removed, 'accuracy': damaged_acc}, damage_dir / 'damaged.pt')
            (damage_dir / 'removed_indices.json').write_text(json.dumps(removed))

            # Save the actual epoch-wise shuffle order once and replay it for each method.
            # Loader's generator state advances on iteration; restore before each method.
            loader_state = train_loader.generator.get_state().clone()
            for method in methods:
                seed_everything(seed + 200000)
                train_loader.generator.set_state(loader_state.clone())
                run = copy.deepcopy(damaged)
                run.use_pruning = False
                run.use_neurogenesis = method != 'weight_only'
                # Regrowth cap is pre-damage N, so methods cannot grow beyond the original model.
                run.max_neurons = model.num_neurons
                optimizer = torch.optim.Adam(run.parameters(), lr=args.lr)
                rows = []
                for epoch in range(args.recovery_epochs + 1):
                    if epoch > 0:
                        train_acc, stats = train_epoch(run, train_loader, optimizer, device)
                        if method == 'activity':
                            # Flux streak from end_epoch() must not determine activity growth.
                            # Keep a separate activity persistence streak across epochs.
                            activity = stats['mean_activity']
                            threshold = torch.quantile(activity, args.activity_quantile)
                            active = (activity >= threshold) & (activity > 0)
                            previous = activity_streak
                            activity_streak = torch.where(active, previous + len(train_loader.dataset),
                                                          torch.zeros_like(previous))
                            run.birth_streak.copy_(activity_streak)
                        births = growth_step(run, method, args)
                        if births:
                            optimizer = torch.optim.Adam(run.parameters(), lr=args.lr)
                            if method == 'activity':
                                activity_streak = run.birth_streak.clone()
                        if method == 'activity' and not births:
                            activity_streak = run.birth_streak.clone()
                    else:
                        train_acc, births = float('nan'), 0
                        activity_streak = torch.zeros(run.num_neurons, dtype=torch.long, device=device)
                    test_acc, test_loss = evaluate(run, test_loader, device)
                    row = dict(seed=seed, damage_fraction=fraction, method=method, epoch=epoch, train_acc=train_acc,
                               test_acc=test_acc, test_loss=test_loss, neurons=run.num_neurons,
                               births=births, pre_damage_acc=before_acc, post_damage_acc=damaged_acc)
                    rows.append(row)
                    print(f'[seed={seed} damage={fraction:.2f} {method}] epoch={epoch} test={test_acc:.2f} N={run.num_neurons} births={births}', flush=True)
                method_dir = damage_dir / method
                method_dir.mkdir(exist_ok=True)
                with (method_dir / 'metrics.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                torch.save({'state_dict': run.state_dict(), 'neurons': run.num_neurons}, method_dir / 'final.pt')
    print('Finished:', args.out, flush=True)


if __name__ == '__main__':
    main()
