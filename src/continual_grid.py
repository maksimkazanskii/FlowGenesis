import os,json,torch,pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR="data/grid_search"
os.makedirs(OUT_DIR,exist_ok=True)

TASKS=[[0,1],[2,3],[4,5],[6,7],[8,9]]
FREQS=[4,16]
RATIOS=[2,4,8,16]
CONFIGS=[(f,f*r,r) for f in FREQS for r in RATIOS]

class OscillatorContinual(nn.Module):
    def __init__(self,n_neurons=40,n_freqs=32,n_steps=128,n_classes=10,temperature=0.1):
        super().__init__()
        self.n_neurons=n_neurons; self.n_freqs=n_freqs; self.n_steps=n_steps; self.temperature=temperature
        self.to_amp=nn.Linear(784,n_neurons*n_freqs)
        self.freqs=nn.Parameter(torch.randn(n_neurons,n_freqs))
        self.phases=nn.Parameter(torch.randn(n_neurons,n_freqs))
        self.prototypes=nn.Parameter(torch.randn(n_classes,n_steps))
    def forward(self,x):
        b=x.shape[0]; x=x.view(b,-1)
        amp=torch.tanh(self.to_amp(x)).view(b,self.n_neurons,self.n_freqs,1)
        t=torch.linspace(0,1,self.n_steps,device=x.device).view(1,1,1,self.n_steps)
        f=F.softplus(self.freqs).view(1,self.n_neurons,self.n_freqs,1)
        p=self.phases.view(1,self.n_neurons,self.n_freqs,1)
        s=(amp*torch.sin(2*torch.pi*f*t+p)).sum(dim=(1,2))
        s=F.normalize(s,dim=1)
        proto=F.normalize(self.prototypes,dim=1)
        return (s@proto.T)/self.temperature,s

class MLPContinual(nn.Module):
    def __init__(self,target_params):
        super().__init__()
        hidden=max(4,int(target_params/794))
        self.hidden=hidden
        self.net=nn.Sequential(nn.Linear(784,hidden),nn.ReLU(),nn.Linear(hidden,10))
    def forward(self,x):
        return self.net(x.view(x.size(0),-1)),None

def filter_dataset(ds,digits): return Subset(ds,[i for i,(_,y) in enumerate(ds) if int(y) in digits])

def mask_logits(logits,digits):
    out=torch.full_like(logits,-1e9)
    out[:,digits]=logits[:,digits]
    return out

def count_params(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_loaders():
    tr=transforms.ToTensor()
    train=datasets.MNIST("./data",train=True,download=True,transform=tr)
    test=datasets.MNIST("./data",train=False,download=True,transform=tr)
    return [DataLoader(filter_dataset(train,d),batch_size=128,shuffle=True) for d in TASKS],[DataLoader(filter_dataset(test,d),batch_size=256,shuffle=False) for d in TASKS]

def train_epoch(model,loader,opt,task_id):
    model.train(); loss_sum=correct=total=0
    for x,y in loader:
        x,y=x.to(DEVICE),y.to(DEVICE)
        opt.zero_grad()
        logits,_=model(x)
        logits=mask_logits(logits,TASKS[task_id])
        loss=F.cross_entropy(logits,y)
        loss.backward()
        opt.step()
        loss_sum+=loss.item()
        correct+=(logits.argmax(1)==y).sum().item()
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
def evaluate_ci(model,loaders):
    model.eval(); correct=total=0; dc=[0]*10; dt=[0]*10
    for loader in loaders:
        for x,y in loader:
            x,y=x.to(DEVICE),y.to(DEVICE)
            logits,_=model(x)
            pred=logits.argmax(1)
            correct+=(pred==y).sum().item()
            total+=y.size(0)
            for d in range(10):
                m=(y==d)
                if m.sum():
                    dc[d]+=(pred[m]==y[m]).sum().item()
                    dt[d]+=m.sum().item()
    overall=100*correct/total
    for d in range(10):
        if dt[d]>0:
            print(f"Digit {d}: {100*dc[d]/dt[d]:.2f}%")
    print(f"Overall: {overall:.2f}%")
    return overall

def forgetting(acc):
    vals=[]
    for j in range(len(TASKS)-1):
        best=max(acc[i][j] for i in range(j,len(TASKS)) if acc[i][j] is not None)
        vals.append(best-acc[-1][j])
    return sum(vals)/len(vals)

def run_model(model,train_loaders,test_loaders,epochs=10):
    acc=[[None for _ in TASKS] for _ in TASKS]
    for task_id,loader in enumerate(train_loaders):
        opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        for _ in range(epochs):
            train_epoch(model,loader,opt,task_id)
        for j in range(task_id+1):
            acc[task_id][j]=evaluate_task(model,test_loaders[j],j)
    ci=evaluate_ci(model,test_loaders)
    return {"task_aware":sum(acc[-1])/len(TASKS),"forgetting":forgetting(acc),"ci":ci,"acc_matrix":acc}

def add_row(rows,model_name,n_freqs,n_steps,ratio,params,hidden,res):
    rows.append({
        "model":model_name,
        "n_freqs":n_freqs,
        "n_steps":n_steps,
        "ratio":ratio,
        "params":params,
        "hidden":hidden,
        "task_aware":res["task_aware"],
        "forgetting":res["forgetting"],
        "ci":res["ci"],
        "ci_gap":res["task_aware"]-res["ci"],
        "acc_matrix":json.dumps(res["acc_matrix"])
    })

def main():
    train_loaders,test_loaders=make_loaders()
    rows=[]
    for n_freqs,n_steps,ratio in CONFIGS:
        print("\n"+"="*80)
        print(f"Freqs={n_freqs} Steps={n_steps} Ratio=1:{ratio}")
        print("="*80)

        osc=OscillatorContinual(n_neurons=40,n_freqs=n_freqs,n_steps=n_steps).to(DEVICE)
        osc_params=count_params(osc)
        print(f"\nOSC params: {osc_params:,}")
        osc_res=run_model(osc,train_loaders,test_loaders,epochs=10)
        add_row(rows,"OSC",n_freqs,n_steps,ratio,osc_params,None,osc_res)

        mlp=MLPContinual(osc_params).to(DEVICE)
        mlp_params=count_params(mlp)
        print(f"\nMLP params: {mlp_params:,} | hidden={mlp.hidden}")
        mlp_res=run_model(mlp,train_loaders,test_loaders,epochs=10)
        add_row(rows,"MLP",n_freqs,n_steps,ratio,mlp_params,mlp.hidden,mlp_res)

        df=pd.DataFrame(rows)
        df.to_csv(f"{OUT_DIR}/grid_results.csv",index=False)
        df.sort_values("ci",ascending=False).to_csv(f"{OUT_DIR}/grid_results_sorted.csv",index=False)
        print(rows[-2])
        print(rows[-1])

    df=pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/grid_results.csv",index=False)
    df.sort_values("ci",ascending=False).to_csv(f"{OUT_DIR}/grid_results_sorted.csv",index=False)
    print("\nTOP RESULTS")
    print(df.sort_values("ci",ascending=False).head(20))

if __name__=="__main__":
    main()