
import os,torch,pandas as pd
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from utils import OscillatorReservoir
from src_coupling.memory import SpectralMemory

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("data/checkpoint",exist_ok=True)

def classify(memory,spectrum):
    spec=F.normalize(spectrum,dim=1)
    cent=F.normalize(memory.centroids+1e-8,dim=1)
    return spec@cent.T

def spectral_orthogonality_loss(spectrum,labels):
    means=[]
    for c in labels.unique():
        means.append(spectrum[labels==c].mean(0))
    if len(means)<2:
        return 0.0*spectrum.mean()
    means=F.normalize(torch.stack(means),dim=1)
    sim=means@means.T
    sim=sim-torch.eye(sim.size(0),device=sim.device)
    return (sim**2).mean()

def train_epoch(model,memory,loader,opt):
    model.train()
    loss_sum=correct=total=0

    for images,labels in loader:

        images,labels=images.to(DEVICE),labels.to(DEVICE)

        opt.zero_grad()

        spectrum,_=model(images)

        #
        # bootstrap first pass
        #
        if memory.counts.sum()==0:
            memory.update(
                spectrum.detach(),
                labels
            )

        logits=classify(
            memory,
            spectrum
        )

        ce=F.cross_entropy(
            logits,
            labels
        )

        orth=spectral_orthogonality_loss(
            spectrum,
            labels
        )

        loss=ce+0.1*orth

        loss.backward()
        opt.step()

        memory.update(
            spectrum.detach(),
            labels
        )

        loss_sum+=loss.item()

        pred=logits.argmax(1)

        correct+=(pred==labels).sum().item()
        total+=labels.size(0)

    return loss_sum/len(loader),100*correct/total

@torch.no_grad()
def evaluate(model,memory,loader):

    model.eval()

    correct=total=0

    for images,labels in loader:

        images,labels=images.to(DEVICE),labels.to(DEVICE)

        spectrum,_=model(images)

        logits=classify(
            memory,
            spectrum
        )

        pred=logits.argmax(1)

        correct+=(pred==labels).sum().item()
        total+=labels.size(0)

    return 100*correct/total

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

def main():

    tr=transforms.ToTensor()

    train_loader=DataLoader(
        datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=tr
        ),
        batch_size=128,
        shuffle=True
    )

    test_loader=DataLoader(
        datasets.MNIST(
            "./data",
            train=False,
            download=True,
            transform=tr
        ),
        batch_size=256,
        shuffle=False
    )

    freq_list=[1,2,4,8,16,32]
    TOTAL_OSCILLATORS=256
    results=[]

    for n_freqs in freq_list:

        n_neurons=max(
            1,
            TOTAL_OSCILLATORS//n_freqs
        )

        print("\n"+"="*60)
        print(
            f"Running experiment with "
            f"n_freqs={n_freqs}, "
            f"n_neurons={n_neurons}"
        )
        print("="*60)

        model=OscillatorReservoir(
            n_neurons=n_neurons,
            n_freqs=n_freqs,
            n_steps=128
        ).to(DEVICE)

        params=count_parameters(model)

        memory=SpectralMemory(
            n_classes=10,
            dim=130,
            device=DEVICE
        )

        opt=torch.optim.Adam(
            model.parameters(),
            lr=1e-3
        )

        epochs=20

        for epoch in range(epochs):

            train_loss,train_acc=train_epoch(
                model,
                memory,
                train_loader,
                opt
            )

            test_acc=evaluate(
                model,
                memory,
                test_loader
            )

            print(
                f"Freqs={n_freqs:02d} | "
                f"Epoch {epoch+1:02d} | "
                f"Loss {train_loss:.4f} | "
                f"Train {train_acc:.2f}% | "
                f"Test {test_acc:.2f}%"
            )

        final_test_acc=evaluate(
            model,
            memory,
            test_loader
        )

        results.append({
            "n_freqs":n_freqs,
            "params":params,
            "test_acc":final_test_acc
        })

    print("="*60)
    print("FINAL RESULTS")
    print("="*60)

    df=pd.DataFrame(results)

    print(df)

    df.to_csv(
        "frequency_ablation.csv",
        index=False
    )

if __name__=="__main__":
    main()