import os,json,torch,pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR="data/continual_spectral"
os.makedirs(OUT_DIR,exist_ok=True)
TASKS=[[0,1],[2,3],[4,5],[6,7],[8,9]]

class SpectralContinual(nn.Module):
    def __init__(self,n_neurons=40,n_freqs=32,n_steps=128,n_classes=10,temperature=0.1):
        super().__init__()
        self.n_neurons=n_neurons; self.n_freqs=n_freqs; self.n_steps=n_steps; self.temperature=temperature
        self.to_amp=nn.Linear(784,n_neurons*n_freqs)
        self.freqs=nn.Parameter(torch.randn(n_neurons,n_freqs))
        self.phases=nn.Parameter(torch.randn(n_neurons,n_freqs))
        self.prototypes=nn.Parameter(torch.randn(n_classes,n_steps))
    def reservoir(self,x):
        b=x.shape[0]; x=x.view(b,-1)
        amp=torch.tanh(self.to_amp(x)).view(b,self.n_neurons,self.n_freqs,1)
        t=torch.linspace(0,1,self.n_steps,device=x.device).view(1,1,1,self.n_steps)
        f=F.softplus(self.freqs).view(1,self.n_neurons,self.n_freqs,1)
        p=self.phases.view(1,self.n_neurons,self.n_freqs,1)
        s=(amp*torch.sin(2*torch.pi*f*t+p)).sum(dim=(1,2))
        return F.normalize(s,dim=1)
    def forward(self,x):
        s=self.reservoir(x); p=F.normalize(self.prototypes,dim=1)
        return (s@p.T)/self.temperature,s

def filter_dataset(ds,digits): return Subset(ds,[i for i,(_,y) in enumerate(ds) if int(y) in digits])
def mask_logits(logits,digits):
    out=torch.full_like(logits,-1e9); out[:,digits]=logits[:,digits]; return out
def prototype_ortho(model):
    p=F.normalize(model.prototypes,dim=1)
    return F.mse_loss(p@p.T,torch.eye(p.shape[0],device=p.device))
def make_loaders():
    tr=transforms.ToTensor()
    train=datasets.MNIST("./data",train=True,download=True,transform=tr)
    test=datasets.MNIST("./data",train=False,download=True,transform=tr)
    return [DataLoader(filter_dataset(train,d),128,shuffle=True) for d in TASKS],[DataLoader(filter_dataset(test,d),256,shuffle=False) for d in TASKS]
def apply_proto_grad_mask(model,task_id):
    m=torch.zeros_like(model.prototypes); m[TASKS[task_id],:]=1
    model.prototypes.grad*=m.to(model.prototypes.grad.device)

def train_epoch(model,loader,opt,task_id):
    model.train(); loss_sum=correct=total=0
    for x,y in loader:
        x,y=x.to(DEVICE),y.to(DEVICE)
        opt.zero_grad()
        logits,_=model(x)
        logits_task=mask_logits(logits,TASKS[task_id])
        loss=F.cross_entropy(logits_task,y)+0.05*prototype_ortho(model)
        loss.backward()
        apply_proto_grad_mask(model,task_id)
        opt.step()
        loss_sum+=loss.item()
        correct+=(logits_task.argmax(1)==y).sum().item()
        total+=y.size(0)
    return loss_sum/len(loader),100*correct/total

@torch.no_grad()
def evaluate_task(model,loader,task_id):
    model.eval(); correct=total=0
    for x,y in loader:
        x,y=x.to(DEVICE),y.to(DEVICE)
        logits,_=model(x)
        logits=mask_logits(logits,TASKS[task_id])
        correct+=(logits.argmax(1)==y).sum().item()
        total+=y.size(0)
    return 100*correct/total

@torch.no_grad()
def evaluate_ci(model,loaders,tag="CLASS INCREMENTAL"):
    model.eval(); correct=total=0; dc=[0]*10; dt=[0]*10
    for loader in loaders:
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            logits,_=model(x); pred=logits.argmax(1)
            correct+=(pred==y).sum().item(); total+=y.size(0)
            for d in range(10):
                m=(y==d)
                if m.sum(): dc[d]+=(pred[m]==y[m]).sum().item(); dt[d]+=m.sum().item()
    print("\n"+"="*60); print(tag); print("="*60)
    for d in range(10):
        if dt[d]>0: print(f"Digit {d}: {100*dc[d]/dt[d]:.2f}%")
    overall=100*correct/total
    print(f"\nOverall: {overall:.2f}%")
    return overall

@torch.no_grad()
def pred_hist(model,loader,name):
    model.eval(); h=torch.zeros(10,dtype=torch.long)
    for x,y in loader:
        logits,_=model(x.to(DEVICE))
        h+=torch.bincount(logits.argmax(1).cpu(),minlength=10)
    print(f"{name} prediction histogram:",h.tolist())

def forgetting(acc):
    vals=[]
    for j in range(len(TASKS)-1):
        best=max(acc[i][j] for i in range(j,len(TASKS)) if acc[i][j] is not None)
        vals.append(best-acc[-1][j])
    return sum(vals)/len(vals)

def count_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    train_loaders,test_loaders=make_loaders()
    model=SpectralContinual(40,128,256).to(DEVICE)
    print(f"\nParameters: {count_params(model):,}")
    acc=[[None for _ in TASKS] for _ in TASKS]; rows=[]; ci_rows=[]
    for task_id,loader in enumerate(train_loaders):
        opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        print("\n"+"="*60); print(f"Task {task_id+1} {TASKS[task_id]}"); print("="*60)
        for ep in range(5):
            loss,tr=train_epoch(model,loader,opt,task_id)
            print(f"Epoch={ep+1:02d} | Loss={loss:.4f} | Train={tr:.2f}%")
        for j in range(task_id+1):
            a=evaluate_task(model,test_loaders[j],j)
            acc[task_id][j]=a
            rows.append({"after_task":task_id+1,"eval_task":j+1,"digits":str(TASKS[j]),"task_aware_accuracy":a})
            print(f"Task-aware eval task {j+1} {TASKS[j]}: {a:.2f}%")
            pred_hist(model,test_loaders[j],f"After task {task_id+1}, eval {TASKS[j]}")
        ci=evaluate_ci(model,test_loaders[:task_id+1],tag=f"CLASS INCREMENTAL AFTER TASK {task_id+1}")
        ci_rows.append({"after_task":task_id+1,"ci_accuracy":ci})
        torch.save(model.state_dict(),f"{OUT_DIR}/model_after_task_{task_id+1}.pt")
    final_avg=sum(acc[-1])/len(TASKS); avg_forgetting=forgetting(acc); final_ci=ci_rows[-1]["ci_accuracy"]
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/task_aware_results.csv",index=False)
    pd.DataFrame(ci_rows).to_csv(f"{OUT_DIR}/class_incremental_results.csv",index=False)
    with open(f"{OUT_DIR}/summary.json","w") as f:
        json.dump({"final_task_aware_avg":final_avg,"avg_forgetting":avg_forgetting,"final_class_incremental":final_ci,"acc_matrix":acc},f,indent=4)
    torch.save(model.state_dict(),f"{OUT_DIR}/final_model.pt")
    print("\nFINAL MATRIX")
    for r in acc: print(r)
    print(f"\nFinal task-aware average: {final_avg:.2f}%")
    print(f"Average forgetting: {avg_forgetting:.2f}%")
    print(f"Final class-incremental accuracy: {final_ci:.2f}%")

if __name__=="__main__":
    main()