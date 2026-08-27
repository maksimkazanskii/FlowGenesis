import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils import OscillatorReservoir


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT = "data/checkpoint/coupled_oscillator.pt"
OUT = "data/vis_coupling"

os.makedirs(OUT, exist_ok=True)


def make_loader():
    return DataLoader(
        datasets.MNIST(
            "./data",
            train=False,
            download=True,
            transform=transforms.ToTensor()
        ),
        batch_size=256,
        shuffle=False
    )


def load_model():
    model = OscillatorReservoir(
        n_neurons=32,
        n_freqs=8,
        n_steps=128
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(CKPT, map_location=DEVICE)
    )

    model.eval()
    return model


@torch.no_grad()
def get_omega(model, x):
    b = x.size(0)
    flat = x.view(b, -1)

    base = F.softplus(
        model.freqs
    ).view(1, -1)

    delta = 0.1 * torch.tanh(
        model.freq_shift(flat)
    )

    omega0 = base + delta

    omegaT = model.evolve_frequencies(
        omega0
    )

    return omega0, omegaT


@torch.no_grad()
def collect(model, loader):
    omega0_all = []
    omegaT_all = []
    labels_all = []

    for x, y in loader:
        x = x.to(DEVICE)

        omega0, omegaT = get_omega(
            model,
            x
        )

        omega0_all.append(
            omega0.cpu()
        )

        omegaT_all.append(
            omegaT.cpu()
        )

        labels_all.append(
            y
        )

    return (
        torch.cat(omega0_all),
        torch.cat(omegaT_all),
        torch.cat(labels_all)
    )


def class_centroids(omega, labels):
    centroids = []

    for c in range(10):
        centroids.append(
            omega[labels == c].mean(0)
        )

    return torch.stack(centroids)


def plot_class_frequency_centroids(centroids):
    plt.figure(figsize=(12, 7))

    for c in range(10):
        plt.plot(
            centroids[c].numpy(),
            label=str(c)
        )

    plt.title("Class Frequency Centroids After Coupling")
    plt.xlabel("Oscillator index")
    plt.ylabel("Frequency state")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(
        f"{OUT}/class_frequency_centroids.png",
        dpi=300
    )
    plt.close()


def plot_centroid_similarity(centroids):
    cent = F.normalize(
        centroids,
        dim=1
    )

    sim = cent @ cent.T

    plt.figure(figsize=(7, 6))
    plt.imshow(
        sim.numpy(),
        vmin=-1,
        vmax=1
    )

    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.title("Class Frequency-Centroid Cosine Similarity")
    plt.tight_layout()
    plt.savefig(
        f"{OUT}/class_frequency_similarity.png",
        dpi=300
    )
    plt.close()

    pd.DataFrame(
        sim.numpy()
    ).round(3).to_csv(
        f"{OUT}/class_frequency_similarity.csv",
        index=False
    )


def plot_frequency_shift(omega0, omegaT):
    shift = (
            omegaT - omega0
    ).abs().mean(0)

    plt.figure(figsize=(12, 5))
    plt.plot(
        shift.numpy()
    )

    plt.title("Mean Absolute Frequency Shift Induced by Coupling")
    plt.xlabel("Oscillator index")
    plt.ylabel("Mean |omega_T - omega_0|")
    plt.tight_layout()
    plt.savefig(
        f"{OUT}/mean_frequency_shift.png",
        dpi=300
    )
    plt.close()


def plot_base_frequencies(model):
    f = F.softplus(
        model.freqs
    ).detach().cpu().view(-1)

    plt.figure(figsize=(8, 5))
    plt.hist(
        f.numpy(),
        bins=30
    )

    plt.title("Learned Base Frequency Distribution")
    plt.xlabel("Frequency")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(
        f"{OUT}/base_frequency_histogram.png",
        dpi=300
    )
    plt.close()


def plot_coupling_matrix(model):
    K = model.K.detach().cpu()

    plt.figure(figsize=(7, 7))
    plt.imshow(
        K.numpy()
    )

    plt.colorbar()
    plt.title("Learned Coupling Matrix K")
    plt.tight_layout()
    plt.savefig(
        f"{OUT}/coupling_matrix.png",
        dpi=300
    )
    plt.close()


def main():
    loader = make_loader()
    model = load_model()

    omega0, omegaT, labels = collect(
        model,
        loader
    )

    centroids = class_centroids(
        omegaT,
        labels
    )

    plot_class_frequency_centroids(
        centroids
    )

    plot_centroid_similarity(
        centroids
    )

    plot_frequency_shift(
        omega0,
        omegaT
    )

    plot_base_frequencies(
        model
    )

    plot_coupling_matrix(
        model
    )

    print(f"Saved visualizations to: {OUT}")


if __name__ == "__main__":
    main()