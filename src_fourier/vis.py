import os,torch,pandas as pd,matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from utils import OscillatorReservoir

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
MODEL_FREQS=32
VIS_DIR="data/vis"
CKPT_DIR="data/checkpoint"

@torch.no_grad()
def class_centroids(model,loader):
    sums=counts=None

    for images,labels in loader:
        images,labels=images.to(DEVICE),labels.to(DEVICE)

        spectrum,signal=model(images)

        if sums is None:
            sums=torch.zeros(
                10,
                spectrum.shape[1],
                device=DEVICE
            )
            counts=torch.zeros(
                10,
                device=DEVICE
            )

        for c in range(10):
            m=labels==c
            if m.any():
                sums[c]+=spectrum[m].sum(0)
                counts[c]+=m.sum()

    return F.normalize(
        sums/counts[:,None],
        dim=1
    ).cpu()

@torch.no_grad()
def centroid_similarity(model,loader):
    c=class_centroids(model,loader)
    sim=(c@c.T).numpy()

    print("\nSpectral centroid similarity:\n")
    print(pd.DataFrame(sim).round(3))

    plt.figure(figsize=(8,6))
    plt.imshow(sim,vmin=-1,vmax=1)
    plt.colorbar()
    plt.title("Spectral Centroid Similarity")
    plt.tight_layout()
    plt.savefig(f"{VIS_DIR}/spectral_similarity.png",dpi=300)
    plt.close()

@torch.no_grad()
def centroid_spectra(model,loader):

    c=class_centroids(model,loader)

    real=c[:,:65]
    imag=c[:,65:]

    fig,ax=plt.subplots(
        1,
        2,
        figsize=(14,6)
    )

    for cls in range(10):
        ax[0].plot(
            real[cls].numpy(),
            label=f"{cls}"
        )

    ax[0].set_title(
        "Real FFT Centroids"
    )

    ax[0].set_xlabel(
        "Frequency Bin"
    )

    ax[0].set_ylabel(
        "Value"
    )

    for cls in range(10):
        ax[1].plot(
            imag[cls].numpy(),
            label=f"{cls}"
        )

    ax[1].set_title(
        "Imag FFT Centroids"
    )

    ax[1].set_xlabel(
        "Frequency Bin"
    )

    ax[1].set_ylabel(
        "Value"
    )

    ax[1].legend(
        ncol=2
    )

    plt.tight_layout()

    plt.savefig(
        f"{VIS_DIR}/spectral_centroids_split.png",
        dpi=300
    )

    plt.close()

@torch.no_grad()
def trajectory_centroids(model,loader):

    sums=None
    counts=torch.zeros(10,device=DEVICE)

    for images,labels in loader:

        images,labels=images.to(DEVICE),labels.to(DEVICE)

        _,signal=model(images)

        if sums is None:
            sums=torch.zeros(
                10,
                signal.shape[1],
                device=DEVICE
            )

        for c in range(10):
            m=labels==c
            if m.any():
                sums[c]+=signal[m].sum(0)
                counts[c]+=m.sum()

    cent=F.normalize(
        sums/counts[:,None],
        dim=1
    ).cpu()

    plt.figure(figsize=(12,8))

    for cls in range(10):
        plt.plot(
            cent[cls].numpy(),
            label=f"{cls}"
        )

    plt.legend(ncol=2)
    plt.title("Trajectory Centroids")
    plt.tight_layout()
    plt.savefig(f"{VIS_DIR}/trajectory_centroids.png",dpi=300)
    plt.close()

def print_frequency_statistics(model):
    f=F.softplus(model.freqs).cpu()
    print(f"\nMean={f.mean():.4f}")
    print(f"Std ={f.std():.4f}")
    print(f"Min ={f.min():.4f}")
    print(f"Max ={f.max():.4f}")

def load_model():
    ckpt=torch.load(
        f"{CKPT_DIR}/checkpoint_{MODEL_FREQS}.pt",
        map_location=DEVICE
    )

    sd=ckpt["state_dict"]

    sd.pop("prototypes",None)

    model=OscillatorReservoir(
        n_neurons=sd["freqs"].shape[0],
        n_freqs=sd["freqs"].shape[1],
        n_steps=ckpt["n_steps"]
    ).to(DEVICE)

    model.load_state_dict(
        sd,
        strict=False
    )

    model.eval()

    return model

def main():

    os.makedirs(VIS_DIR,exist_ok=True)

    loader=DataLoader(
        datasets.MNIST(
            "./data",
            train=False,
            download=True,
            transform=transforms.ToTensor()
        ),
        batch_size=256,
        shuffle=False
    )

    model=load_model()

    print_frequency_statistics(model)
    centroid_similarity(model,loader)
    centroid_spectra(model,loader)
    trajectory_centroids(model,loader)

if __name__=="__main__":
    main()