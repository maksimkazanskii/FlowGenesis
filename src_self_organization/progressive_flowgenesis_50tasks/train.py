#!/usr/bin/env python3
"""20/50-task Split CIFAR-100: fixed-width PNN vs Progressive FlowGenesis.
Run: python train_50tasks.py --num-tasks 50 --method both --epochs 40 --seeds 1 2 3 4 5
"""
import argparse, copy, csv, json, random, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from model_neurogenesis_new import NeuralContainer
from config_cifar import Config


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def device_auto():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


class FeatureEncoder(nn.Module):
    """Shared encoder, learned on Task 1 then frozen for both methods."""
    def __init__(self, feature_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(128,feature_dim),nn.ReLU())
    def forward(self,x): return self.net(x)


class ProgressiveModel(nn.Module):
    def __init__(self, cfg, method, feature_dim=128, lateral_dim=32):
        super().__init__()
        self.encoder=FeatureEncoder(feature_dim)
        self.columns=nn.ModuleList()
        self.cfg=cfg; self.method=method
        self.feature_dim=feature_dim; self.lateral_dim=lateral_dim

    def append_column(self):
        task=len(self.columns)
        config=copy.copy(self.cfg)
        config.input_dim=self.feature_dim+task*self.lateral_dim
        config.output_dim=self.cfg.classes_per_task
        config.hidden_neurons=(self.cfg.pnn_width if self.method=='pnn' else self.cfg.hidden_neurons)
        config.use_neurogenesis=(self.method=='flowgenesis')
        config.use_pruning=(self.method=='flowgenesis')
        col=NeuralContainer(config)
        self.columns.append(col.to(next(self.encoder.parameters()).device))
        return col

    def freeze_completed(self):
        for col in self.columns[:-1]:
            col.eval()
            for p in col.parameters(): p.requires_grad_(False)
        if len(self.columns)>1:
            self.encoder.eval()
            for p in self.encoder.parameters(): p.requires_grad_(False)

    @staticmethod
    def pool_features(hidden, dim):
        # Fixed, parameter-free lateral interface; identical for both methods.
        return torch.nn.functional.adaptive_avg_pool1d(hidden.unsqueeze(1),dim).squeeze(1)

    def forward_task(self,x,task):
        # No gradients or state-statistics updates in frozen older modules.
        if task>0:
            with torch.no_grad(): base=self.encoder(x)
        else: base=self.encoder(x)
        previous=[]
        for j in range(task):
            with torch.no_grad():
                old_in=torch.cat([base]+previous,dim=1)
                old_hidden=self.columns[j].relax(old_in)
                previous.append(self.pool_features(old_hidden,self.lateral_dim))
        current_in=torch.cat([base]+previous,dim=1)
        return self.columns[task](current_in)


def count_active(model, through):
    # Includes stored encoder, all task heads, lateral input weights and masked recurrent edges.
    total=sum(p.numel() for p in model.encoder.parameters())
    for col in model.columns[:through+1]:
        total+=sum(p.numel() for name,p in col.named_parameters() if name!='W')
        total+=int(torch.count_nonzero(col.mask).item())
    return int(total)


def make_datasets(root, seed, val_fraction, num_tasks):
    tf=transforms.ToTensor()
    train=datasets.CIFAR100(root=root,train=True,download=True,transform=tf)
    test=datasets.CIFAR100(root=root,train=False,download=True,transform=tf)
    rng=np.random.default_rng(seed)
    class_order=rng.permutation(100).tolist()
    labels=np.asarray(train.targets); test_labels=np.asarray(test.targets)
    assert 100 % num_tasks == 0, "Number of tasks must divide 100"
    classes_per_task = 100 // num_tasks
    splits=[]
    for task in range(num_tasks):
        classes=class_order[task*classes_per_task:(task+1)*classes_per_task]
        tr_idx=[]; val_idx=[]
        for cls in classes:
            ids=np.where(labels==cls)[0]; rng.shuffle(ids)
            nv=max(1,int(round(len(ids)*val_fraction)))
            val_idx.extend(ids[:nv]); tr_idx.extend(ids[nv:])
        test_idx=np.where(np.isin(test_labels,classes))[0].tolist()
        splits.append((Subset(train,tr_idx),Subset(train,val_idx),Subset(test,test_idx),classes))
    return splits,class_order


class RemapSubset(torch.utils.data.Dataset):
    def __init__(self, subset, classes):
        self.subset=subset; self.mapping={int(c):i for i,c in enumerate(classes)}
    def __len__(self): return len(self.subset)
    def __getitem__(self,i):
        x,y=self.subset[i]; return x,self.mapping[int(y)]


def loaders(split,batch,workers,seed):
    tr,va,te,classes=split
    g=torch.Generator().manual_seed(seed)
    kwargs=dict(batch_size=batch,num_workers=workers,pin_memory=torch.cuda.is_available())
    return (DataLoader(RemapSubset(tr,classes),shuffle=True,generator=g,**kwargs),
            DataLoader(RemapSubset(va,classes),shuffle=False,**kwargs),
            DataLoader(RemapSubset(te,classes),shuffle=False,**kwargs))


@torch.no_grad()
def accuracy(model, loader, task, device):
    model.eval(); correct=total=0
    for x,y in loader:
        x=x.to(device); y=y.to(device)
        pred=model.forward_task(x,task).argmax(1)
        correct+=(pred==y).sum().item(); total+=len(y)
    return 100*correct/max(total,1)


def train_method(args, method, seed, splits, order, device):
    seed_everything(seed)
    cfg=Config()
    cfg.classes_per_task=100 // args.num_tasks
    cfg.hidden_neurons=args.width
    cfg.pnn_width=args.pnn_width
    cfg.min_neurons=args.min_neurons
    if args.min_neurons > args.width:
        raise ValueError("--min-neurons cannot exceed FlowGenesis --width")
    cfg.max_neurons=args.max_neurons
    cfg.structure_warmup_examples=args.warmup
    cfg.structure_check_every_examples=args.check_every
    cfg.structure_cooldown_examples=args.cooldown
    cfg.birth_fraction=args.birth_fraction
    cfg.death_fraction=args.death_fraction
    cfg.birth_threshold=args.birth_threshold
    cfg.birth_persistence=args.birth_persistence
    cfg.death_persistence=args.death_persistence
    cfg.max_births_per_epoch=args.max_births
    cfg.max_deaths_per_epoch=args.max_deaths
    cfg.learning_rate=args.lr; cfg.weight_decay=args.weight_decay
    cfg.relaxation_steps=args.relaxation_steps
    model=ProgressiveModel(cfg,method,args.feature_dim,args.lateral_dim).to(device)
    out=Path(args.output)/method/f'seed_{seed}'; out.mkdir(parents=True,exist_ok=True)
    (out/'settings.json').write_text(json.dumps(vars(args)|{'method':method,'seed':seed,'class_order':order},indent=2))
    results=[]; test_loaders=[]
    for t,split in enumerate(splits):
        train_loader,val_loader,test_loader=loaders(split,args.batch,args.workers,seed+t)
        test_loaders.append(test_loader)
        col=model.append_column(); model.freeze_completed()
        trainable=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(trainable,lr=args.lr,weight_decay=args.weight_decay)
        best_val=-1.; best_state=None; best_epoch=-1; best_width=None
        events=[]; start=time.time()
        for epoch in range(args.epochs):
            model.eval(); col.train()
            if t==0: model.encoder.train()
            col.current_epoch=epoch+1
            correct=total=0
            for x,y in train_loader:
                x=x.to(device); y=y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits=model.forward_task(x,t)
                loss=nn.functional.cross_entropy(logits,y)
                loss.backward(); optimizer.step()
                correct+=(logits.argmax(1)==y).sum().item(); total+=len(y)
                if method=='flowgenesis' and col.window_examples>=cfg.structure_check_every_examples:
                    stats=col.end_epoch()
                    update=col.apply_structural_updates()
                    events.append(dict(epoch=epoch+1,examples=col.examples_seen,neurons=update.after,
                                       births=update.births,deaths=update.deaths,flux=stats['network_mean_flux']))
                    if update.changed:
                        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                                   lr=args.lr,weight_decay=args.weight_decay)
            val=accuracy(model,val_loader,t,device)
            if val>best_val:
                best_val=val; best_epoch=epoch+1; best_width=col.num_neurons
                best_state=copy.deepcopy(col.state_dict())
                if t==0: best_encoder=copy.deepcopy(model.encoder.state_dict())
            print(f'{method} seed={seed} task={t+1:02d}/{args.num_tasks} epoch={epoch+1:03d} '
                  f'train={100*correct/max(total,1):.2f} val={val:.2f} N={col.num_neurons}',flush=True)
        # Restore validation-selected state; later tasks never alter it.
        # Shape can differ after structural changes, so recreate selected column.
        if col.num_neurons!=best_width:
            old=col
            restore_cfg=copy.copy(cfg)
            restore_cfg.input_dim=old.input_dim; restore_cfg.output_dim=cfg.classes_per_task
            restore_cfg.hidden_neurons=best_width
            model.columns[t]=NeuralContainer(restore_cfg).to(device)
            col=model.columns[t]
        col.load_state_dict(best_state)
        if t==0: model.encoder.load_state_dict(best_encoder)
        col.eval()
        for p in col.parameters(): p.requires_grad_(False)
        if t==0:
            model.encoder.eval()
            for p in model.encoder.parameters(): p.requires_grad_(False)
        accuracies=[accuracy(model,test_loaders[j],j,device) for j in range(t+1)]
        active=count_active(model,t)
        row=dict(task=t+1,classes=' '.join(map(str,split[3])),module_neurons=col.num_neurons,
                 active_parameters=active,new_parameters=active-(results[-1]['active_parameters'] if results else 0),
                 average_accuracy=float(np.mean(accuracies)),current_task_accuracy=accuracies[-1],
                 all_task_accuracies=accuracies,best_validation_accuracy=best_val,
                 selected_epoch=best_epoch,elapsed_seconds=time.time()-start,
                 births=sum(e['births'] for e in events),deaths=sum(e['deaths'] for e in events))
        results.append(row)
        torch.save({'encoder':model.encoder.state_dict(),'columns':[c.state_dict() for c in model.columns],
                    'widths':[c.num_neurons for c in model.columns],'class_order':order,
                    'results':results},out/f'after_task_{t+1:02d}.pt')
        (out/f'task_{t+1:02d}_events.json').write_text(json.dumps(events,indent=2))
        (out/'results.json').write_text(json.dumps(results,indent=2))
        with (out/'results.csv').open('w',newline='') as f:
            fields=[k for k in row if k!='all_task_accuracies']; writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader(); writer.writerows([{k:r[k] for k in fields} for r in results])
        print(f'COMPLETED {method} seed={seed} task={t+1}: avg_acc={row["average_accuracy"]:.2f} '
              f'params={active:,} newN={col.num_neurons}',flush=True)
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['both','pnn','flowgenesis'],default='both')
    p.add_argument('--num-tasks',type=int,choices=[20,50],default=50,
                   help='20 tasks = 5 classes/task; 50 tasks = 2 classes/task')
    p.add_argument('--seeds',type=int,nargs='+',default=[1])
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--width',type=int,default=100,help='Initial FlowGenesis module width')
    p.add_argument('--pnn-width',type=int,default=60,help='Fixed PNN module width')
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--feature-dim',type=int,default=128)
    p.add_argument('--lateral-dim',type=int,default=32)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--relaxation-steps',type=int,default=10)
    p.add_argument('--val-fraction',type=float,default=.1)
    p.add_argument('--warmup',type=int,default=20000)
    p.add_argument('--check-every',type=int,default=128,help='Structural statistics window in training examples; 128 checks after each full batch')
    p.add_argument('--cooldown',type=int,default=10000)
    p.add_argument('--birth-fraction',type=float,default=.2)
    p.add_argument('--death-fraction',type=float,default=.2)
    p.add_argument('--birth-threshold',type=float,default=.35)
    p.add_argument('--birth-persistence',type=int,default=5000)
    p.add_argument('--death-persistence',type=int,default=30000)
    p.add_argument('--max-births',type=int,default=32)
    p.add_argument('--max-deaths',type=int,default=32)
    p.add_argument('--min-neurons',type=int,default=10)
    p.add_argument('--max-neurons',type=int,default=900)
    p.add_argument('--data-root',default='./data')
    p.add_argument('--output',default='data/progressive_cifar100_50tasks')
    args=p.parse_args()
    if args.check_every < 1 or args.min_neurons < 1 or args.pnn_width < 1 or args.width < 1:
        p.error("Widths, minimum neurons and check interval must be positive")
    device=device_auto(); print('Device:',device,flush=True)
    for seed in args.seeds:
        splits,order=make_datasets(args.data_root,seed,args.val_fraction,args.num_tasks)
        for method in (['pnn','flowgenesis'] if args.method=='both' else [args.method]):
            train_method(args,method,seed,splits,order,device)

if __name__=='__main__': main()
