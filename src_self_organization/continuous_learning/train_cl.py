"""Replay-free class-incremental Split CIFAR-10 with FlowGenesis.

Run from project root:
python -u -m src_self_organization.continuous_learning.train_cl --model full --seed 0
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from .config_cl import MODELS, make_config
from ..model_neurogenesis_new import NeuralContainer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def class_order(seed):
    # Shared across model variants for each seed.
    return np.random.default_rng(seed + 20261017).permutation(10).tolist()


def subset_for_classes(dataset, classes):
    targets = np.asarray(dataset.targets)
    indices = np.flatnonzero(np.isin(targets, classes)).tolist()
    return Subset(dataset, indices)


@torch.no_grad()
def evaluate(model, loader, device, allowed_classes):
    model.eval()
    correct, count = 0, 0
    allowed = torch.as_tensor(allowed_classes, dtype=torch.long, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        # Seen-class CIL: restrict to all seen classes, NEVER to one task.
        selected = logits.index_select(1, allowed)
        pred = allowed[selected.argmax(dim=1)]
        correct += (pred == y).sum().item()
        count += y.numel()
    return 100.0 * correct / count if count else float('nan')


def save_atomic(payload, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=sorted(MODELS))
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--epochs-per-task', type=int, default=20)
    parser.add_argument('--output-dir', default='data/so/continual/cifar10')
    parser.add_argument('--data-dir', default='./data')
    args = parser.parse_args()
    if args.epochs_per_task < 1:
        parser.error('--epochs-per-task must be positive')

    seed_all(args.seed)
    cfg = make_config(args.model, args.seed, args.epochs_per_task)
    device = get_device()
    cfg.device = str(device)
    order = class_order(args.seed)
    tasks = [order[i:i + 2] for i in range(0, 10, 2)]
    output = Path(args.output_dir) / args.model / f'seed_{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'task_order.json').write_text(json.dumps({'seed': args.seed, 'tasks': tasks}, indent=2))

    train_set = CIFAR10(args.data_dir, train=True, download=True, transform=ToTensor())
    test_set = CIFAR10(args.data_dir, train=False, download=True, transform=ToTensor())
    train_loaders, test_loaders = [], []
    for task_classes in tasks:
        train_loaders.append(DataLoader(subset_for_classes(train_set, task_classes),
                                        batch_size=cfg.batch_size, shuffle=True,
                                        num_workers=cfg.num_workers))
        test_loaders.append(DataLoader(subset_for_classes(test_set, task_classes),
                                       batch_size=cfg.batch_size, shuffle=False,
                                       num_workers=cfg.num_workers))

    model = NeuralContainer(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    matrix = np.full((5, 5), np.nan, dtype=np.float64)
    epochs = []
    structural_events = []
    global_epoch = 0

    print(f'Model={args.model} seed={args.seed} N0={cfg.hidden_neurons} device={device} tasks={tasks}', flush=True)
    print('Class-incremental; no replay; task identity unavailable at inference.', flush=True)

    for task_id, task_classes in enumerate(tasks):
        loader = train_loaders[task_id]
        seen_classes = [c for task in tasks[:task_id + 1] for c in task]
        print(f'=== TASK {task_id+1}/5 classes={task_classes} ===', flush=True)
        for local_epoch in range(args.epochs_per_task):
            global_epoch += 1
            model.current_epoch = global_epoch
            model.train()
            train_correct = train_count = 0
            train_loss_sum = 0.0
            event_start = len(structural_events)
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                # Standard full-10-way CE; no task-specific output masking.
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_correct += (logits.argmax(1) == y).sum().item()
                train_count += y.numel()
                train_loss_sum += loss.item() * y.numel()
                if model.window_examples >= cfg.structure_check_every_examples:
                    stats = model.end_epoch()
                    update = model.apply_structural_updates()
                    structural_events.append({
                        'task': task_id + 1, 'epoch': global_epoch,
                        'examples_seen': model.examples_seen,
                        'mean_flux': stats['network_mean_flux'],
                        'births': update.births, 'deaths': update.deaths,
                        'neurons': update.after,
                    })
                    if update.changed:
                        # Matches existing FlowGenesis behavior after parameter resizing.
                        optimizer = torch.optim.Adam(model.parameters(),
                                                     lr=cfg.learning_rate,
                                                     weight_decay=cfg.weight_decay)
            # End-of-epoch evaluation on ALL seen classes (not only current task).
            # No evaluation-induced flux changes: model.eval() disables tracking.
            per_task = [evaluate(model, test_loaders[j], device, seen_classes)
                        for j in range(task_id + 1)]
            epoch_events = structural_events[event_start:]
            record = {
                'epoch': global_epoch, 'task': task_id + 1,
                'task_epoch': local_epoch + 1,
                'train_loss': train_loss_sum / train_count,
                'train_accuracy': 100 * train_correct / train_count,
                'per_task_accuracy': per_task,
                'seen_accuracy': float(np.mean(per_task)),
                'num_neurons': model.num_neurons,
                'births': sum(e['births'] for e in epoch_events),
                'deaths': sum(e['deaths'] for e in epoch_events),
                'total_born': len(model.birth_history),
                'total_pruned': len(model.death_history),
            }
            epochs.append(record)
            print(f"epoch={global_epoch:03d} task={task_id+1} "
                  f"train={record['train_accuracy']:.2f}% seen={record['seen_accuracy']:.2f}% "
                  f"neurons={model.num_neurons} births={record['births']} deaths={record['deaths']}",
                  flush=True)
            save_atomic({'epochs': epochs, 'structural_events': structural_events,
                         'tasks': tasks, 'model': args.model, 'seed': args.seed},
                        output / 'learning_curves.pt')

        # Evaluate each seen task with same seen-class classifier.
        for j in range(task_id + 1):
            matrix[task_id, j] = evaluate(model, test_loaders[j], device, seen_classes)
        save_atomic({'accuracy_matrix': matrix, 'tasks': tasks, 'model': args.model,
                     'seed': args.seed}, output / 'accuracy_matrix.pt')
        print(f'After task {task_id+1}: {matrix[task_id, :task_id+1].round(2).tolist()}', flush=True)

    final_accuracy = float(np.nanmean(matrix[-1]))
    forgetting = [float(np.nanmax(matrix[j:4, j]) - matrix[4, j]) for j in range(4)]
    summary = {
        'model': args.model, 'seed': args.seed, 'tasks': tasks,
        'epochs_per_task': args.epochs_per_task,
        'accuracy_matrix': matrix,
        'final_average_accuracy': final_accuracy,
        'average_forgetting': float(np.mean(forgetting)),
        'forgetting_by_task': forgetting,
        'final_neurons': model.num_neurons,
        'total_births': len(model.birth_history),
        'total_deaths': len(model.death_history),
        'final_parameters': sum(p.numel() for p in model.parameters()),
    }
    save_atomic(summary, output / 'summary.pt')
    save_atomic({'state_dict': model.state_dict(), 'cfg': vars(cfg), 'tasks': tasks},
                output / 'final_model.pt')
    model.save_statistics(output / 'statistics.pt')
    print(f'DONE {args.model} seed={args.seed} final_ACC={final_accuracy:.2f} '
          f'forgetting={summary["average_forgetting"]:.2f} neurons={model.num_neurons}', flush=True)


if __name__ == '__main__':
    main()
