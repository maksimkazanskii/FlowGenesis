"""Five-task Split CIFAR-10: static vs activity vs full FlowGenesis.

Run from project root:
python -m src_self_organization.continuous_learning.train_cl_selforg --method full --seed 0
Uses the existing config_cl.py and model_neurogenesis_new.py in the project.
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, ConcatDataset, WeightedRandomSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from .config_cl import make_config
from ..model_neurogenesis_cl_selforg import NeuralContainer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset(ds, classes):
    return Subset(ds, np.flatnonzero(np.isin(np.asarray(ds.targets), classes)).tolist())


def device_choice():
    if torch.cuda.is_available(): return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


@torch.no_grad()
def accuracy(model, loader, device, seen):
    model.eval()
    allowed = torch.tensor(seen, device=device)
    correct = total = 0
    for x, y in loader:
        logits = model(x.to(device)).index_select(1, allowed)
        pred = allowed[logits.argmax(1)]
        correct += (pred == y.to(device)).sum().item()
        total += len(y)
    return 100.0*correct/max(1,total)


def save(obj, path):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['static','activity','full'], required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs-per-task', type=int, default=100)
    p.add_argument('--task1-max-neurons', type=int, default=400)
    p.add_argument('--birth-budget', type=int, default=100)
    p.add_argument('--replay-per-class', type=int, default=20)
    p.add_argument('--output-dir', default='data/so/continual/cifar10_selforg')
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--batch-size', type=int, default=None)
    a = p.parse_args()
    if a.epochs_per_task < 100 or a.birth_budget < 1 or a.replay_per_class < 0:
        p.error('Require >=100 epochs, positive birth budget, and nonnegative replay size')
    seed_all(a.seed)
    cfg = make_config('full', a.seed, a.epochs_per_task)
    cfg.max_neurons = a.task1_max_neurons
    if a.batch_size is not None: cfg.batch_size = a.batch_size
    device = device_choice()
    cfg.device = str(device)
    order = np.random.default_rng(a.seed+20261017).permutation(10).tolist()
    tasks = [order[2*i:2*i+2] for i in range(5)]
    out = Path(a.output_dir)/a.method/f'seed_{a.seed}'
    out.mkdir(parents=True, exist_ok=True)
    (out/'configuration.json').write_text(json.dumps({**vars(a),'tasks':tasks,'device':str(device)}, indent=2))
    train = CIFAR10(a.data_dir, train=True, download=True, transform=ToTensor())
    test = CIFAR10(a.data_dir, train=False, download=True, transform=ToTensor())
    task_train = [subset(train, t) for t in tasks]
    task_test = [DataLoader(subset(test,t), batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers) for t in tasks]
    model = NeuralContainer(cfg, method=a.method).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(a.seed+9876)
    replay_indices = []
    history = []
    matrix = np.full((5,5), np.nan)
    epoch_global = 0
    for task_id, classes in enumerate(tasks):
        if task_id:
            model.begin_task(task_id, birth_budget=a.birth_budget)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        seen = [c for t in tasks[:task_id+1] for c in t]
        # Same balanced exemplar indices for all methods with a given seed.
        replay = Subset(train, replay_indices)
        training_ds = ConcatDataset([task_train[task_id], replay]) if replay_indices else task_train[task_id]
        if replay_indices:
            # Equal expected sampling mass for current-task and replay examples;
            # otherwise a 20/class buffer would be nearly invisible in 10k examples.
            current_n, old_n = len(task_train[task_id]), len(replay_indices)
            weights = torch.cat([torch.full((current_n,), 0.5/current_n),
                                 torch.full((old_n,), 0.5/old_n)])
            sampler = WeightedRandomSampler(weights, num_samples=2*current_n, replacement=True)
            loader = DataLoader(training_ds, batch_size=cfg.batch_size, sampler=sampler,
                                num_workers=cfg.num_workers)
        else:
            loader = DataLoader(training_ds, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=cfg.num_workers)
        print(f'TASK {task_id+1} method={a.method} classes={classes} N={model.hidden_dim}', flush=True)
        for epoch in range(a.epochs_per_task):
            epoch_global += 1
            model.current_epoch = epoch_global
            model.train()
            loss_sum = total = 0
            for x,y in loader:
                x,y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                # Restrict predictions to classes seen so far, both during training
                # and evaluation. Replay examples are included in this same loss.
                seen_t = torch.tensor(seen, device=device)
                targets = torch.empty_like(y)
                for index, cls in enumerate(seen):
                    targets[y == cls] = index
                loss = criterion(logits.index_select(1,seen_t), targets)
                loss.backward()
                model.protect_gradients()
                optimizer.step()
                model.restore_frozen()
                loss_sum += loss.item()*len(y)
                total += len(y)
                if model.window_examples >= cfg.structure_check_every_examples:
                    model.end_epoch()
                    update = model.apply_structural_updates()
                    if update.changed:
                        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
                        model.restore_frozen()
            per_task = [accuracy(model,task_test[j],device,seen) for j in range(task_id+1)]
            row = dict(task=task_id+1,epoch=epoch+1,global_epoch=epoch_global,
                       loss=loss_sum/max(1,total), per_task_accuracy=per_task,
                       average_accuracy=float(np.mean(per_task)),neurons=model.hidden_dim,
                       births_this_task=model.births_this_task)
            history.append(row)
            print(f'  epoch={epoch+1:03d} N={model.hidden_dim} acc={row["average_accuracy"]:.2f}', flush=True)
            save(dict(history=history, tasks=tasks, structure_events=model.structure_events), out/'learning_curves.pt')
        matrix[task_id,:task_id+1] = per_task
        save(dict(matrix=matrix,tasks=tasks),out/'accuracy_matrix.pt')
        save(dict(state_dict=model.state_dict(), task=task_id+1, frozen_n=model.frozen_n,
                  method=a.method, cfg=vars(cfg)),out/f'checkpoint_task_{task_id+1}.pt')
        for cls in classes:
            all_idx = np.flatnonzero(np.asarray(train.targets)==cls)
            replay_indices.extend(rng.choice(all_idx, size=min(a.replay_per_class,len(all_idx)), replace=False).tolist())
    fgt = [float(np.nanmax(matrix[i:,i])-matrix[-1,i]) for i in range(4)]
    summary = dict(method=a.method,seed=a.seed,final_average_accuracy=float(np.nanmean(matrix[-1])),
                   average_forgetting=float(np.mean(fgt)), forgetting=fgt,
                   final_neurons=model.hidden_dim, matrix=matrix,
                   structure_events=model.structure_events)
    save(summary,out/'summary.pt')
    print('DONE',summary['final_average_accuracy'],summary['final_neurons'],flush=True)

if __name__=='__main__': main()
