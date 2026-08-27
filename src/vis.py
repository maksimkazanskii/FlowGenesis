import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import OscillatorReservoir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_FREQS = 16
VIS_DIR = "data/vis"
CKPT_DIR = "data/checkpoint"

@torch.no_grad()
def prototype_time_domain(model):

    p = F.normalize(
        model.prototypes,
        dim=1
    ).cpu()

    plt.figure(
        figsize=(12,8)
    )

    for cls in range(10):

        plt.plot(
            p[cls].numpy(),
            label=f"Class {cls}"
        )

    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")

    plt.title(
        "Class Prototype Waveforms"
    )

    plt.legend(
        ncol=2
    )

    plt.tight_layout()

    plt.savefig(
        f"{VIS_DIR}/prototype_waveforms.png",
        dpi=300
    )

    plt.close()
@torch.no_grad()
def prototype_similarity_matrix(model):

    p = F.normalize(
        model.prototypes,
        dim=1
    ).cpu()

    sim = (
            p @ p.T
    ).numpy()

    plt.figure(
        figsize=(8,6)
    )

    plt.imshow(
        sim,
        vmin=-1,
        vmax=1,
        aspect="equal"
    )

    plt.colorbar()

    plt.xlabel("Class")
    plt.ylabel("Class")

    plt.title(
        "Prototype Cosine Similarity Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        f"{VIS_DIR}/prototype_similarity.png",
        dpi=300
    )

    plt.close()

    print("\nPrototype similarity matrix:\n")
    print(pd.DataFrame(sim).round(3))
@torch.no_grad()
def prototype_frequency_heatmap(model):

    proto = F.normalize(
        model.prototypes,
        dim=1
    ).cpu()

    spectra=[]

    for cls in range(10):

        s=torch.fft.rfft(
            proto[cls]
        ).abs()

        spectra.append(
            s.numpy()
        )

    spectra=torch.tensor(
        spectra
    ).numpy()

    plt.figure(
        figsize=(10,6)
    )

    plt.imshow(
        spectra,
        aspect="auto"
    )

    plt.colorbar()

    plt.xlabel(
        "Frequency Bin"
    )

    plt.ylabel(
        "Class"
    )

    plt.title(
        "Class Anchor Frequency Usage"
    )

    plt.tight_layout()

    plt.savefig(
        f"{VIS_DIR}/prototype_frequency_heatmap.png",
        dpi=300
    )

    plt.close()

@torch.no_grad()
def visualize_class_anchor_frequencies(model):

    proto = F.normalize(
        model.prototypes,
        dim=1
    ).cpu()

    n_classes = proto.shape[0]

    plt.figure(
        figsize=(12,8)
    )

    for cls in range(n_classes):

        signal = proto[cls]

        spectrum = torch.fft.rfft(
            signal
        )

        power = spectrum.abs().numpy()

        plt.plot(
            power,
            label=f"Class {cls}"
        )

    plt.xlabel(
        "Frequency Bin"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        "Prototype Frequency Spectra"
    )

    plt.legend(
        ncol=2
    )

    plt.tight_layout()

    plt.savefig(
        f"{VIS_DIR}/prototype_frequency_spectra.png",
        dpi=300
    )

    plt.close()
@torch.no_grad()
def print_frequency_statistics(model):
    freqs = F.softplus(model.freqs).cpu()

    print("=" * 80)
    print("FREQUENCY ANALYSIS")
    print("=" * 80)
    print("Shape:", freqs.shape)

    print("\nMean per channel:")
    print(freqs.mean(dim=0))

    print("\nStd per channel:")
    print(freqs.std(dim=0))

    print("\nMin per channel:")
    print(freqs.min(dim=0).values)

    print("\nMax per channel:")
    print(freqs.max(dim=0).values)

    print(f"\nGlobal mean = {freqs.mean():.4f}")
    print(f"Global std  = {freqs.std():.4f}")
    print(f"Global min  = {freqs.min():.4f}")
    print(f"Global max  = {freqs.max():.4f}")

    plt.figure(figsize=(10, 6))
    plt.imshow(freqs.numpy(), aspect="auto")
    plt.colorbar()
    plt.xlabel("Frequency")
    plt.ylabel("Neuron")
    plt.title(
        f"Learned Frequencies ({model.n_neurons}x{model.n_freqs})"
    )
    plt.tight_layout()
    plt.savefig(
        f"{VIS_DIR}/frequencies_heatmap.png",
        dpi=300
    )
    plt.close()

@torch.no_grad()
def visualize_max_activations(
        model,
        loader,
        top_k=1
):

    best_val = torch.full(
        (model.n_neurons, model.n_freqs),
        -1e9
    )

    best_img = [[None for _ in range(model.n_freqs)]
                for _ in range(model.n_neurons)]

    for images, _ in loader:

        images = images.to(DEVICE)

        amp = torch.tanh(
            model.to_amp(
                images.view(
                    images.size(0),
                    -1
                )
            )
        )

        amp = amp.view(
            images.size(0),
            model.n_neurons,
            model.n_freqs
        )

        for b in range(images.size(0)):

            for n in range(model.n_neurons):

                for f in range(model.n_freqs):

                    v = amp[b,n,f].item()

                    if v > best_val[n,f]:

                        best_val[n,f] = v

                        best_img[n][f] = (
                            images[b]
                            .cpu()
                            .clone()
                        )

    return best_img,best_val

def save_frequency_receptive_fields(
        model,
        loader
):

    imgs,vals = visualize_max_activations(
        model,
        loader
    )

    for n in range(model.n_neurons):

        fig,ax = plt.subplots(
            model.n_freqs,
            1,
            figsize=(3,2*model.n_freqs)
        )

        for f in range(model.n_freqs):

            ax[f].imshow(
                imgs[n][f].squeeze(),
                cmap="gray"
            )

            ax[f].set_title(
                f"F={f} "
                f"A={vals[n,f]:.2f}"
            )

            ax[f].axis("off")

        plt.tight_layout()

        plt.savefig(
            f"{VIS_DIR}/neuron_{n}.png",
            dpi=300
        )

        plt.close()
def optimize_class(
        model,
        class_id,
        steps=1000,
        lr=0.05
):

    model.eval()

    x = torch.randn(
        1,
        1,
        28,
        28,
        device=DEVICE,
        requires_grad=True
    )

    opt = torch.optim.Adam(
        [x],
        lr=lr
    )

    for _ in range(steps):

        opt.zero_grad()

        logits,_ = model(x)

        score = logits[
            0,
            class_id
        ]

        reg = 1e-4 * (
                x*x
        ).mean()

        loss = -(score - reg)

        loss.backward()

        opt.step()

        with torch.no_grad():

            x.clamp_(0,1)

    return (
        x.detach()
        .cpu()
        .squeeze()
        .numpy(),
        score.item()
    )

def save_class_visualizations(model):

    fig,ax = plt.subplots(
        2,
        5,
        figsize=(12,6)
    )

    ax=ax.flatten()

    for cls in range(10):

        img,val = optimize_class(
            model,
            cls
        )

        ax[cls].imshow(
            img,
            cmap="gray"
        )

        ax[cls].set_title(
            f"{cls}\n{val:.2f}"
        )

        ax[cls].axis("off")

    plt.tight_layout()

    plt.savefig(
        f"{VIS_DIR}/class_visualizations.png",
        dpi=300
    )

    plt.close()

def save_synthetic_receptive_fields(model):

    outdir = (
        f"{VIS_DIR}/synthetic"
    )

    os.makedirs(
        outdir,
        exist_ok=True
    )

    for n in range(model.n_neurons):

        fig,ax = plt.subplots(
            model.n_freqs,
            1,
            figsize=(3,2*model.n_freqs)
        )

        for f in range(model.n_freqs):

            img,val = optimize_neuron_frequency(
                model,
                n,
                f
            )

            ax[f].imshow(
                img,
                cmap="gray"
            )

            ax[f].set_title(
                f"N={n} F={f} A={val:.3f}"
            )

            ax[f].axis("off")

        plt.tight_layout()

        plt.savefig(
            f"{outdir}/neuron_{n}.png",
            dpi=300
        )

        plt.close()


@torch.no_grad()
def save_digit_visualizations(model, loader):

    collected = defaultdict(list)

    for images, labels in loader:

        for img, lbl in zip(images, labels):

            lbl = int(lbl)

            if len(collected[lbl]) < 3:
                collected[lbl].append(img)

        if all(len(collected[d]) >= 3 for d in range(10)):
            break

    print("\nGenerating visualizations...\n")

    for digit in range(10):

        for idx, image in enumerate(collected[digit]):

            x = image.unsqueeze(0).to(DEVICE).view(1, -1)

            #
            # amplitude matrix
            #
            amplitudes = torch.tanh(
                model.to_amp(x)
            )

            amplitudes = amplitudes.view(
                model.n_neurons,
                model.n_freqs
            )

            amp_matrix = (
                amplitudes
                .detach()
                .cpu()
                .numpy()
            )

            plt.figure(figsize=(14, 5))

            #
            # original digit
            #
            plt.subplot(1, 2, 1)

            plt.imshow(
                image.squeeze(),
                cmap="gray"
            )

            plt.axis("off")

            plt.title(
                f"Digit {digit}"
            )

            #
            # amplitude matrix
            #
            plt.subplot(1, 2, 2)

            plt.imshow(
                amp_matrix,
                aspect="auto"
            )

            plt.colorbar()

            plt.xlabel(
                "Frequency"
            )

            plt.ylabel(
                "Neuron"
            )

            plt.title(
                f"Amplitude Matrix "
                f"({model.n_neurons}x{model.n_freqs})"
            )

            plt.tight_layout()

            filename = (
                f"{VIS_DIR}/digit_{digit}_{idx}.png"
            )

            plt.savefig(
                filename,
                dpi=300
            )

            plt.close()

            print(
                f"Saved: {filename}"
            )


def load_model():

    ckpt = torch.load(
        f"{CKPT_DIR}/checkpoint_{MODEL_FREQS}.pt",
        map_location=DEVICE,
    )

    state_dict = ckpt["state_dict"]

    n_neurons = state_dict["freqs"].shape[0]
    n_freqs = state_dict["freqs"].shape[1]

    print(
        f"Detected architecture: "
        f"{n_neurons} x {n_freqs}"
    )

    model = OscillatorReservoir(
        n_neurons=n_neurons,
        n_freqs=n_freqs,
        n_steps=ckpt["n_steps"],
        temperature=ckpt["temperature"],
    ).to(DEVICE)

    model.load_state_dict(state_dict)

    model.eval()

    return model


def main():

    os.makedirs(
        VIS_DIR,
        exist_ok=True
    )

    test_loader = DataLoader(
        datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=256,
        shuffle=False,
    )

    model = load_model()

    print_frequency_statistics(
        model
    )

    save_digit_visualizations(
        model,
        test_loader
    )

    print(
        f"\nOutputs saved to {VIS_DIR}"
    )
    print_frequency_statistics(model)
    visualize_class_anchor_frequencies(model)
    prototype_frequency_heatmap(model)


    print_frequency_statistics(model)
    visualize_class_anchor_frequencies(model)
    prototype_frequency_heatmap(model)
    prototype_time_domain(model)
    prototype_similarity_matrix(model)
    save_digit_visualizations(model, test_loader)
    #save_synthetic_receptive_fields(model)
    save_class_visualizations(model)

if __name__ == "__main__":
    main()