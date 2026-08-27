import os,json,torch,pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR="data/continual_independent"
os.makedirs(OUT_DIR,exist_ok=True)
TASKS=[[0,1],[2,3],[4,5],[6,7],[8,9]]
FREQS_PER_TASK=8
N_TASKS=5
MAX_FREQS=FREQS_PER_TASK*N_TASKS

class IndependentOscillatorReservoir(nn.Module):
    def __init__(self,in_features=784,n_neurons=40,max_freqs=40,n_steps=128,n_classes=10,temperature=0.1):
        super().__init__()
        self.n_neurons=n_neurons
        self.max_freqs=max_freqs
        self.n_steps=n_steps
        self.temperature=temperature
        self.input_proj=nn.Linear(in_features,n_neurons)
        self.freqs=nn.Parameter(torch.randn(n_neurons,max_freqs))
        self.phases=nn.Parameter(torch.randn(n_neurons,max_freqs))
        self.amps=nn.Parameter(torch.randn(n_neurons,max_freqs))
        self.prototypes=nn.Parameter(torch.randn(n_classes,n_steps))
    def forward_block(self,x,task_id):
        start=task_id*FREQS_PER_TASK
        end=(task_id+1)*FREQS_PER_TASK
        b=x.shape[0]
        x=x.view(b,-1)
        h=torch.tanh(self.input_proj(x)).view(b,self.n_neurons,1,1)
        t=torch.linspace(0.0,1.0,self.n_steps,device=x.device).view(1,1,1,self.n_steps)
        f=F.softplus(self.freqs[:,start:end]).view(1,self.n_neurons,FREQS_PER_TASK,1)
        p=self.phases[:,start:end].view(1,self.n_neurons,FREQS_PER_TASK,1)
        a=self.amps[:,start:end].view(1,self.n_neurons,FREQS_PER_TASK,1)
        waves=h*a*torch.sin(2.0*torch.pi*f*t+p)
        s=F.normalize(waves.sum(dim=(1,2)),dim=1)
        prototypes=F.normalize(self.prototypes,dim=1)
        logits=(s@prototypes.T)/self.temperature
        mask=torch.full_like(logits,-1e9)
        mask[:,TASKS[task_id]]=logits[:,TASKS[task_id]]
        return mask,s
    def forward(self,x,task_id):
        return self.forward_block(x,task_id)
    def predict_all_classes(self,x):
        b=x.shape[0]
        x=x.view(b,-1)

        h=torch.tanh(
            self.input_proj(x)
        ).view(
            b,
            self.n_neurons,
            1,
            1
        )

        t=torch.linspace(
            0.0,
            1.0,
            self.n_steps,
            device=x.device
        ).view(
            1,
            1,
            1,
            self.n_steps
        )

        prototypes=F.normalize(
            self.prototypes,
            dim=1
        )

        final_logits=torch.full(
            (b,10),
            -1e9,
            device=x.device
        )

        for task_id in range(N_TASKS):

            start=task_id*FREQS_PER_TASK
            end=(task_id+1)*FREQS_PER_TASK

            f=F.softplus(
                self.freqs[:,start:end]
            ).view(
                1,
                self.n_neurons,
                FREQS_PER_TASK,
                1
            )

            p=self.phases[:,start:end].view(
                1,
                self.n_neurons,
                FREQS_PER_TASK,
                1
            )

            a=self.amps[:,start:end].view(
                1,
                self.n_neurons,
                FREQS_PER_TASK,
                1
            )

            waves=h*a*torch.sin(
                2.0*torch.pi*f*t+p
            )

            s=F.normalize(
                waves.sum(dim=(1,2)),
                dim=1
            )

            logits=(s@prototypes.T)/self.temperature

            for cls in TASKS[task_id]:
                final_logits[:,cls]=logits[:,cls]

        return final_logits

def prototype_orthogonality_loss(model):
    p = F.normalize(
        model.prototypes,
        dim=1
    )

    gram = p @ p.T

    target = torch.eye(
        gram.shape[0],
        device=gram.device
    )

    return F.mse_loss(
        gram,
        target
    )
def filter_dataset(dataset,digits):
    return Subset(dataset,[i for i,(_,y) in enumerate(dataset) if int(y) in digits])

def make_loaders():
    tr=transforms.ToTensor()
    train=datasets.MNIST("./data",train=True,download=True,transform=tr)
    test=datasets.MNIST("./data",train=False,download=True,transform=tr)
    return [DataLoader(filter_dataset(train,d),batch_size=128,shuffle=True) for d in TASKS],[DataLoader(filter_dataset(test,d),batch_size=256,shuffle=False) for d in TASKS]

def apply_gradient_masks(model,task_id):
    start=task_id*FREQS_PER_TASK
    end=(task_id+1)*FREQS_PER_TASK
    fm=torch.zeros_like(model.freqs)
    fm[:,start:end]=1.0
    pm=torch.zeros_like(model.prototypes)
    pm[TASKS[task_id],:]=1.0
    model.freqs.register_hook(lambda g:g*fm.to(g.device))
    model.amps.register_hook(lambda g:g*fm.to(g.device))
    model.phases.register_hook(lambda g:g*fm.to(g.device))
    model.prototypes.register_hook(lambda g:g*pm.to(g.device))

def train_epoch(model,loader,opt,task_id):
    model.train()
    loss_sum,correct,total=0,0,0
    for x,y in loader:
        x,y=x.to(DEVICE),y.to(DEVICE)
        opt.zero_grad()
        logits,_=model(x,task_id)
        ce_loss = F.cross_entropy(
            logits,
            y
        )

        ortho_loss = prototype_orthogonality_loss(
            model
        )

        loss = (
                ce_loss
                +
                0.05 * ortho_loss
        )
        loss.backward()
        opt.step()
        loss_sum+=loss.item()
        correct+=(logits.argmax(1)==y).sum().item()
        total+=y.size(0)
    return loss_sum/len(loader),100*correct/total

@torch.no_grad()
def evaluate(model,loader,task_id):
    model.eval()
    correct,total=0,0
    for x,y in loader:
        x,y=x.to(DEVICE),y.to(DEVICE)
        logits,_=model(x,task_id)
        correct+=(logits.argmax(1)==y).sum().item()
        total+=y.size(0)
    return 100*correct/total

@torch.no_grad()
def evaluate_class_incremental(
        model,
        loaders
):
    model.eval()

    correct=0
    total=0

    per_digit_correct=[0]*10
    per_digit_total=[0]*10

    for loader in loaders:

        for x,y in loader:

            x=x.to(DEVICE)
            y=y.to(DEVICE)

            logits=model.predict_all_classes(x)

            pred=logits.argmax(1)

            correct += (
                    pred==y
            ).sum().item()

            total += y.size(0)

            for d in range(10):

                mask=(y==d)

                if mask.sum()>0:

                    per_digit_correct[d] += (
                            pred[mask]==y[mask]
                    ).sum().item()

                    per_digit_total[d] += (
                        mask.sum().item()
                    )

    overall=100*correct/total

    print("\n"+"="*60)
    print("CLASS INCREMENTAL")
    print("="*60)

    for d in range(10):

        acc=100*per_digit_correct[d]/per_digit_total[d]

        print(
            f"Digit {d}: "
            f"{acc:.2f}%"
        )

    print(
        f"\nOverall: "
        f"{overall:.2f}%"
    )

    return overall

def forgetting(acc):
    vals=[]
    for j in range(len(TASKS)-1):
        best=max(acc[i][j] for i in range(j,len(TASKS)) if acc[i][j] is not None)
        vals.append(best-acc[-1][j])
    return sum(vals)/len(vals)

def main():
    train_loaders,test_loaders=make_loaders()
    model=IndependentOscillatorReservoir(max_freqs=MAX_FREQS).to(DEVICE)
    acc=[[None for _ in TASKS] for _ in TASKS]
    rows=[]
    for task_id,loader in enumerate(train_loaders):

        if task_id == 1:
            print("\nFREEZING INPUT PROJECTION\n")
            for p in model.input_proj.parameters():
                p.requires_grad = False

        apply_gradient_masks(model,task_id)

        opt=torch.optim.Adam(
            filter(
                lambda p:p.requires_grad,
                model.parameters()
            ),
            lr=1e-3
        )
        print("\n"+"="*60)
        print(f"Task {task_id+1} {TASKS[task_id]} | frequency block {task_id*8}:{(task_id+1)*8}")
        print("="*60)
        for ep in range(5):
            loss,tr=train_epoch(model,loader,opt,task_id)
            print(f"Task={task_id+1} | Epoch={ep+1:02d} | Loss={loss:.4f} | Train={tr:.2f}%")
        for j in range(task_id+1):
            a=evaluate(model,test_loaders[j],j)
            acc[task_id][j]=a
            rows.append({"after_task":task_id+1,"eval_task":j+1,"digits":str(TASKS[j]),"freq_block":f"{j*8}:{(j+1)*8}","accuracy":a})
            print(f"Eval after task {task_id+1} on task {j+1} {TASKS[j]}: {a:.2f}%")
        torch.save(model.state_dict(),f"{OUT_DIR}/model_after_task_{task_id+1}.pt")
        torch.save(F.softplus(model.freqs).detach().cpu(),f"{OUT_DIR}/freqs_after_task_{task_id+1}.pt")
        torch.save(model.amps.detach().cpu(),f"{OUT_DIR}/amps_after_task_{task_id+1}.pt")
        torch.save(model.phases.detach().cpu(),f"{OUT_DIR}/phases_after_task_{task_id+1}.pt")
        torch.save(model.prototypes.detach().cpu(),f"{OUT_DIR}/prototypes_after_task_{task_id+1}.pt")

    final_avg=sum(acc[-1])/len(TASKS)
    avg_forgetting=forgetting(acc)
    ci_acc=evaluate_class_incremental(
        model,
        test_loaders
    )

    print(
        "\n"+
        "="*60
    )

    print(
        f"FINAL CLASS-INCREMENTAL ACCURACY: "
        f"{ci_acc:.2f}%"
    )

    print(
        "="*60
    )
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/results.csv",index=False)
    summary={
        "final_avg_accuracy":final_avg,
        "avg_forgetting":avg_forgetting,
        "class_incremental_accuracy":ci_acc,
        "acc_matrix":acc
    }

    with open(
            f"{OUT_DIR}/summary.json",
            "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=4
        )
    print("\nFINAL MATRIX")
    for r in acc:
        print(r)
    print(f"\nFinal average accuracy: {final_avg:.2f}%")
    print(f"Average forgetting: {avg_forgetting:.2f}%")

if __name__=="__main__":
    main()