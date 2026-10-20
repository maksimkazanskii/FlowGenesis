#!/usr/bin/env python3
"""CIFAR-100 cyclic class-diversity experiment using existing FlowGenesis modules.

Place beside config_cifar.py and model_neurogenesis_new.py.
The original train.py is NOT imported (it runs progressive columns).
"""
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, RandomSampler
from torchvision import datasets, transforms

from config_cifar import Config
from model_neurogenesis_new import NeuralContainer

DEFAULT_SCHEDULE = [5, 10, 20, 50, 100, 50, 20, 10, 5,
                    10, 20, 50, 100, 50, 20, 10, 5]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class FeatureEncoder(nn.Module):
    """Same CNN structure as in supplied progressive train.py."""
    def __init__(self, feature_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, feature_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class CyclicModel(nn.Module):
    def __init__(self, cfg, feature_dim):
        super().__init__()
        self.encoder = FeatureEncoder(feature_dim)
        self.container = NeuralContainer(cfg)

    def forward(self, x):
        return self.container(self.encoder(x))


def active_parameters(model):
    container = model.container
    return (sum(p.numel() for p in model.encoder.parameters())
            + sum(p.numel() for name, p in container.named_parameters() if name != 'W')
            + int(torch.count_nonzero(container.mask).item()))


def mask_logits(logits, allowed):
    # Keep output dimension at 100 and suppress inactive classes.
    return logits.masked_fill(~allowed.unsqueeze(0), torch.finfo(logits.dtype).min)


@torch.no_grad()
def evaluate(model, loader, device, allowed=None):
    model.eval()
    good = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if allowed is not None:
            logits = mask_logits(logits, allowed)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction='sum').item()
        good += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return 100.0 * good / max(1, total), loss_sum / max(1, total)


def prepare_data(args, seed, schedule):
    tf = transforms.ToTensor()
    train = datasets.CIFAR100(args.data_root, train=True, download=True, transform=tf)
    test = datasets.CIFAR100(args.data_root, train=False, download=True, transform=tf)
    rng = np.random.default_rng(seed)
    order = rng.permutation(100).tolist()
    labels = np.asarray(train.targets)
    test_labels = np.asarray(test.targets)
    train_parts, val_parts, test_parts = {}, {}, {}
    for k in sorted(set(schedule)):
        classes = order[:k]
        tr_ids, val_ids = [], []
        # Same per-class train/validation split across all nested subsets.
        for c in classes:
            ids = np.where(labels == c)[0]
            crng = np.random.default_rng(seed * 1009 + int(c))
            ids = crng.permutation(ids)
            nv = max(1, int(round(len(ids) * args.val_fraction)))
            val_ids.extend(ids[:nv].tolist())
            tr_ids.extend(ids[nv:].tolist())
        te_ids = np.where(np.isin(test_labels, classes))[0].tolist()
        train_parts[k] = Subset(train, tr_ids)
        val_parts[k] = Subset(train, val_ids)
        test_parts[k] = Subset(test, te_ids)
    return order, train_parts, val_parts, test_parts


def make_eval_loader(ds, args):
    return DataLoader(ds, batch_size=args.batch, shuffle=False,
                      num_workers=args.workers, pin_memory=torch.cuda.is_available())


def make_train_loader(ds, args, seed):
    # Fixed steps per epoch for all K; replacement also works for K=5.
    g = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(ds, replacement=True,
                            num_samples=args.steps_per_epoch * args.batch,
                            generator=g)
    return DataLoader(ds, batch_size=args.batch, sampler=sampler,
                      num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                      drop_last=True)


def create_config(args, method):
    cfg = Config()
    cfg.input_dim = args.feature_dim
    cfg.output_dim = 100
    cfg.hidden_neurons = args.flow_width if method == 'flowgenesis' else int(method.split('_')[1])
    cfg.use_neurogenesis = method == 'flowgenesis'
    cfg.use_pruning = method == 'flowgenesis'
    cfg.min_neurons = args.min_neurons
    cfg.max_neurons = args.max_neurons
    cfg.recurrent_connection_probability = args.connectivity
    cfg.relaxation_steps = args.relaxation_steps
    cfg.structure_warmup_examples = args.warmup
    cfg.structure_cooldown_examples = args.cooldown
    cfg.structure_check_every_examples = args.check_every
    cfg.birth_threshold = args.birth_threshold
    cfg.birth_persistence = args.birth_persistence
    cfg.death_persistence = args.death_persistence
    cfg.max_births_per_epoch = args.max_births
    cfg.max_deaths_per_epoch = args.max_deaths
    cfg.birth_fraction = args.birth_fraction
    cfg.death_fraction = args.death_fraction
    # The supplied model uses these *different* names for its death criteria.
    cfg.death_activity_fraction = args.death_activity_fraction
    cfg.death_flux_fraction = args.death_flux_fraction
    cfg.death_flux_max = args.death_flux_max
    return cfg


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_one(args, method, seed, data, device):
    seed_all(seed)
    order, train_parts, val_parts, test_parts = data
    cfg = create_config(args, method)
    model = CyclicModel(cfg, args.feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    run_dir = Path(args.output) / method / f'seed_{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'settings.json').write_text(json.dumps({**vars(args),
        'method': method, 'seed': seed, 'class_order': order}, indent=2))
    history = []
    events = []
    completed = 0
    ckpt_path = run_dir / 'latest.pt'
    if args.resume and ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        if saved['method'] != method or saved['seed'] != seed or saved['schedule'] != args.schedule:
            raise ValueError('Checkpoint does not match this run')
        cfg.hidden_neurons = saved['neurons']
        model = CyclicModel(cfg, args.feature_dim).to(device)
        model.load_state_dict(saved['model'])
        # Restore non-state_dict counters used by structural logic.
        c = model.container
        for name, value in saved['structural_state'].items():
            setattr(c, name, value)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        optimizer.load_state_dict(saved['optimizer'])
        history = saved['history']
        events = saved['events']
        completed = saved['completed_phases']
        # Restore RNG state for reproducibility.
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if torch.cuda.is_available() and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        print(f'RESUMED {method} seed={seed} after phase {completed}', flush=True)

    eval_loaders = {k: (make_eval_loader(val_parts[k], args),
                        make_eval_loader(test_parts[k], args))
                    for k in set(args.schedule)}
    for phase_idx in range(completed, len(args.schedule)):
        k = args.schedule[phase_idx]
        allowed = torch.zeros(100, dtype=torch.bool, device=device)
        allowed[order[:k]] = True
        # Independent per-phase sampling stream; repeatable after resume.
        train_loader = make_train_loader(train_parts[k], args,
                                         seed * 100000 + phase_idx)
        val_loader, test_loader = eval_loaders[k]
        start = time.time()
        for epoch in range(1, args.epochs_per_phase + 1):
            model.train()
            model.container.current_epoch = phase_idx * args.epochs_per_phase + epoch
            correct = total = 0
            loss_total = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = mask_logits(model(x), allowed)
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                correct += (logits.argmax(1) == y).sum().item()
                total += y.numel()
                loss_total += loss.item() * y.numel()
                c = model.container
                if method == 'flowgenesis' and c.window_examples >= cfg.structure_check_every_examples:
                    stats = c.end_epoch()
                    update = c.apply_structural_updates()
                    if update.changed:
                        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                                     weight_decay=args.weight_decay)
                    events.append({'phase': phase_idx + 1, 'epoch': epoch,
                                   'examples_seen': c.examples_seen,
                                   'classes': k, 'neurons': c.num_neurons,
                                   'births': update.births, 'deaths': update.deaths,
                                   'flux': stats['network_mean_flux']})
            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs_per_phase:
                val_acc, _ = evaluate(model, val_loader, device, allowed)
                row = {'phase': phase_idx + 1, 'classes': k, 'epoch': epoch,
                       'global_epoch': phase_idx * args.epochs_per_phase + epoch,
                       'train_accuracy': 100 * correct / max(1, total),
                       'train_loss': loss_total / max(1, total),
                       'val_accuracy': val_acc, 'neurons': model.container.num_neurons,
                       'active_parameters': active_parameters(model)}
                history.append(row)
                save_csv(run_dir / 'learning_curves.csv', history)
                print(f'{method} seed={seed} phase={phase_idx+1:02d}/{len(args.schedule)} '
                      f'K={k:3d} epoch={epoch:03d}/{args.epochs_per_phase} '
                      f'train={row["train_accuracy"]:.2f} val={val_acc:.2f} '
                      f'N={row["neurons"]}', flush=True)
        test_acc, test_loss = evaluate(model, test_loader, device, allowed)
        full_acc, full_loss = evaluate(model, eval_loaders[100][1], device, None)
        phase_row = {'phase': phase_idx + 1, 'classes': k,
                     'neurons': model.container.num_neurons,
                     'active_parameters': active_parameters(model),
                     'test_accuracy': test_acc, 'test_loss': test_loss,
                     'full_100_unmasked_accuracy': full_acc,
                     'full_100_unmasked_loss': full_loss,
                     'phase_seconds': time.time() - start,
                     'births': sum(e['births'] for e in events if e['phase'] == phase_idx + 1),
                     'deaths': sum(e['deaths'] for e in events if e['phase'] == phase_idx + 1)}
        previous = []
        if (run_dir / 'phase_results.json').exists():
            previous = json.loads((run_dir / 'phase_results.json').read_text())
        previous.append(phase_row)
        (run_dir / 'phase_results.json').write_text(json.dumps(previous, indent=2))
        save_csv(run_dir / 'phase_results.csv', previous)
        (run_dir / 'events.json').write_text(json.dumps(events, indent=2))
        c = model.container
        structural_state = {name: getattr(c, name) for name in
                            ('examples_seen', 'last_structure_examples', 'window_examples',
                             'current_epoch', 'epoch_batch_count')}
        state = {'method': method, 'seed': seed, 'schedule': args.schedule,
                 'completed_phases': phase_idx + 1, 'neurons': c.num_neurons,
                 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'structural_state': structural_state,
                 'history': history, 'events': events,
                 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        torch.save(state, ckpt_path)
        print(f'PHASE DONE {method} seed={seed} phase={phase_idx+1} '
              f'K={k} test={test_acc:.2f} N={c.num_neurons}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['all', 'flowgenesis', 'static_50',
                                        'static_100', 'static_200'], default='all')
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--schedule', type=int, nargs='+', default=DEFAULT_SCHEDULE)
    p.add_argument('--epochs-per-phase', type=int, default=250)
    p.add_argument('--steps-per-epoch', type=int, default=100)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--feature-dim', type=int, default=128)
    p.add_argument('--flow-width', type=int, default=100)
    p.add_argument('--min-neurons', type=int, default=5)
    p.add_argument('--max-neurons', type=int, default=900)
    p.add_argument('--connectivity', type=float, default=0.2)
    p.add_argument('--relaxation-steps', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--warmup', type=int, default=20000)
    p.add_argument('--cooldown', type=int, default=10000)
    p.add_argument('--check-every', type=int, default=128)
    p.add_argument('--birth-threshold', type=float, default=0.35)
    p.add_argument('--birth-persistence', type=int, default=5000)
    p.add_argument('--death-persistence', type=int, default=30000)
    p.add_argument('--birth-fraction', type=float, default=0.2)
    p.add_argument('--death-fraction', type=float, default=0.2)
    p.add_argument('--death-activity-fraction', type=float, default=0.1)
    p.add_argument('--death-flux-fraction', type=float, default=0.2)
    p.add_argument('--death-flux-max', type=float, default=0.13)
    p.add_argument('--max-births', type=int, default=32)
    p.add_argument('--max-deaths', type=int, default=32)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--data-root', default='./data')
    p.add_argument('--output', default='data/cifar100_complexity_cycle')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if 100 not in args.schedule:
        p.error('Schedule must contain 100 for full-dataset retention evaluation')
    if any(k < 2 or k > 100 for k in args.schedule):
        p.error('All class counts must be between 2 and 100')
    if min(args.epochs_per_phase, args.steps_per_epoch, args.batch, args.log_every) < 1:
        p.error('Epochs, steps, batch size and logging interval must be positive')
    if args.min_neurons > args.flow_width or args.max_neurons < args.flow_width:
        p.error('FlowGenesis initial width must be between min and max')
    device = choose_device()
    print('Device:', device, 'Schedule:', args.schedule, flush=True)
    methods = ['flowgenesis', 'static_50', 'static_100', 'static_200'] if args.method == 'all' else [args.method]
    for seed in args.seeds:
        data = prepare_data(args, seed, args.schedule)
        for method in methods:
            run_one(args, method, seed, data, device)


if __name__ == '__main__':
    main()
