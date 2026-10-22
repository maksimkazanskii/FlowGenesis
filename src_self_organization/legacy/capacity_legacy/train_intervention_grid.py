"""
train.py

Experiment 0

Training script for the Random Neural Container.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from config import Config
from model import NeuralContainer

from pathlib import Path
LAMBDA_GRID = [
    0.0,
    0.333,
    1.0,
    3.33,
    10.0,
    33.3,
]
for capacity_lambda in LAMBDA_GRID:

    run_dir = (
            Path("data")
            / "so"
            / "intervention"
            / "grid"
            / f"lambda={capacity_lambda:.1f}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset
# ============================================================

def get_dataloaders(cfg):

    if cfg.dataset.lower() == "mnist":

        dataset_class = datasets.MNIST

    elif cfg.dataset.lower() == "fashionmnist":

        dataset_class = datasets.FashionMNIST

    else:

        raise ValueError(
            f"Unknown dataset: {cfg.dataset}"
        )

    train_dataset = dataset_class(
        "./data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_dataset = dataset_class(
        "./data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return train_loader, test_loader

# ============================================================
# Training
# ============================================================

def train_epoch(
        model,
        loader,
        optimizer,
        criterion,
        device,
        capacity_lambda,
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)

        loss.backward()

        model.modulate_gradients(
            capacity_lambda
        )

        optimizer.step()

        total_loss += loss.item()

        pred = logits.argmax(1)

        correct += (pred == labels).sum().item()

        total += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        loss = criterion(logits, labels)

        total_loss += loss.item()

        pred = logits.argmax(1)

        correct += (pred == labels).sum().item()

        total += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


# ============================================================
# Main
# ============================================================

def main():

    cfg = Config()

    set_seed(cfg.seed)

    device = torch.device(cfg.device)

    train_loader, test_loader = get_dataloaders(cfg)

    LAMBDA_GRID = [
        0.0,
        0.1,
        0.3,
        0.5,
        1.0,
        2.0,
    ]

    results = []

    for capacity_lambda in LAMBDA_GRID:

        print("\n" + "=" * 70)
        print(f"Capacity λ = {capacity_lambda:.2f}")
        print("=" * 70)

        # ------------------------------------------
        # New model for every lambda
        # ------------------------------------------

        set_seed(cfg.seed)

        model = NeuralContainer(cfg).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        best_acc = 0.0

        for epoch in range(cfg.epochs):

            train_loss, train_acc = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                capacity_lambda,
            )

            test_loss, test_acc = evaluate(
                model,
                test_loader,
                criterion,
                device,
            )

            model.record_statistics()

            print(
                f"[λ={capacity_lambda:.2f}] "
                f"Epoch {epoch+1:02d}/{cfg.epochs} | "
                f"Train {train_acc:.2f}% | "
                f"Test {test_acc:.2f}%"
            )

            if test_acc > best_acc:

                best_acc = test_acc

                torch.save(
                    model.state_dict(),
                    run_dir / "best_model.pt",
                    )

        model.save_statistics(
            run_dir / f"{cfg.dataset.lower()}_statistics.pt"
        )

        results.append(
            (capacity_lambda, best_acc)
        )

        print(
            f"Best accuracy (λ={capacity_lambda:.2f}) = "
            f"{best_acc:.2f}%"
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)

    for lam, acc in results:

        print(
            f"λ = {lam:>4.2f}   "
            f"Best Test Accuracy = {acc:.2f}%"
        )
if __name__ == "__main__":
    main()