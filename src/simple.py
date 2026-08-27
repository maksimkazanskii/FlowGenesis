import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import OscillatorReservoir
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import os

os.makedirs(
    "data/checkpoint",
    exist_ok=True
)
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return (total_loss / len(loader), 100.0 * correct / total)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        logits, _ = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total

@torch.no_grad()
def prototype_correlation_matrix(
        model,
):
    p = F.normalize(model.prototypes, dim=1)
    return (p @ p.T).cpu()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
    )
    freq_list = [1, 2, 4, 8, 16, 32]
    results = []
    TOTAL_OSCILLATORS = 256
    for n_freqs in freq_list:
        n_neurons = max(1, TOTAL_OSCILLATORS // n_freqs)
        print("\n" + "=" * 60)
        print(f"Running experiment with n_freqs={n_freqs}, n_neurons={n_neurons}")
        print("=" * 60)
        model = OscillatorReservoir(
            n_neurons=n_neurons,
            n_freqs=n_freqs,
            n_steps=128,
            temperature=0.1,
        ).to(DEVICE)
        params = count_parameters(model)
        print(
            f"Trainable parameters: "
            f"{params:,}"
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
        )

        epochs = 20

        for epoch in range(epochs):

            train_loss, train_acc = train_epoch(
                model,
                train_loader,
                optimizer,
            )

            test_acc = evaluate(
                model,
                test_loader,
            )

            print(
                f"Freqs={n_freqs:02d} | "
                f"Epoch {epoch+1:02d} | "
                f"Loss {train_loss:.4f} | "
                f"Train {train_acc:.2f}% | "
                f"Test {test_acc:.2f}%"
            )

        final_test_acc = evaluate(
            model,
            test_loader,
        )

        results.append(
            {
                "n_freqs": n_freqs,
                "params": params,
                "test_acc": final_test_acc,
            }
        )

        freqs = F.softplus(
            model.freqs
        ).detach().cpu()

        torch.save(
            freqs,
            f"data/checkpoint/freqs_{n_freqs}.pt"
        )

        torch.save(
            model.state_dict(),
            f"data/checkpoint/model_{n_freqs}.pt"
        )

        torch.save(
            {
                "n_freqs": n_freqs,
                "n_neurons": 40,
                "n_steps": 128,
                "temperature": 0.1,
                "state_dict": model.state_dict(),
                "test_acc": final_test_acc,
            },
            f"data/checkpoint/checkpoint_{n_freqs}.pt"
        )
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Freqs':>8} {'Params':>12} {'Test Acc':>12}")
    for r in results:
        print(f"{r['n_freqs']:>8} {r['params']:>12,} {r['test_acc']:>11.2f}%")
    df = pd.DataFrame(results)
    print("\n")
    print(df)
    df.to_csv("frequency_ablation.csv", index=False)


if __name__ == "__main__":
    main()