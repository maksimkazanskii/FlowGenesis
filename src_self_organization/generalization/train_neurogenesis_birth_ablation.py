"""
train.py

Experiment 0

Training script for the Random Neural Container.
"""
import gc
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from ..model_neurogenesis import NeuralContainer
import argparse
from pathlib import Path


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

    elif cfg.dataset.lower() == "cifar10":
        dataset_class = datasets.CIFAR10

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

def train_epoch(model, loader, optimizer, criterion, device):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        # Release batch tensors as soon as possible
        del images, labels, logits, loss, pred

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

        # Release batch tensors
        del images, labels, logits, loss, pred

    return total_loss / len(loader), 100.0 * correct / total


# ============================================================
# Main
# ============================================================

def main():

    # ============================================================
    # Arguments
    # ============================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["fashionmnist", "cifar10"],
        required=True,
    )

    parser.add_argument(
        "--birth_rate",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    # ============================================================
    # Configuration
    # ============================================================

    if args.dataset == "fashionmnist":
        from ..config import Config

    elif args.dataset == "cifar10":
        from ..config_cifar import Config

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    cfg = Config()

    cfg.dataset = args.dataset

    # Fixed N0 for birth-rate ablation
    cfg.hidden_neurons = 80

    # Ablated structural parameter
    cfg.max_births_per_epoch = args.birth_rate

    cfg.seed = args.seed


    set_seed(cfg.seed)

    # CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    cfg.device = str(device)

    print(
        f"Dataset={cfg.dataset} | "
        f"N0={cfg.hidden_neurons} | "
        f"BirthRate={cfg.max_births_per_epoch} | "
        f"DeathRate={cfg.max_deaths_per_epoch} | "
        f"Seed={cfg.seed} | "
        f"Device={device}"
    )

    device = torch.device(cfg.device)

    # ============================================================
    # Data
    # ============================================================

    train_loader, test_loader = get_dataloaders(cfg)

    # ============================================================
    # Model
    # ============================================================

    model = NeuralContainer(cfg).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # ============================================================
    # Output directory
    # ============================================================

    save_dir = (
            Path("data/so/birth_rate_ablation")
            / cfg.dataset.lower()
            / f"birth_rate_{cfg.max_births_per_epoch}"
            / f"seed_{cfg.seed}"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # Statistics
    # ============================================================

    # Best-performing evaluated architecture
    best_acc = -float("inf")
    best_loss = None
    best_epoch = 0
    best_neurons = model.num_neurons
    best_total_born = 0
    best_total_pruned = 0

    # Final evaluated architecture
    #
    # IMPORTANT:
    # These statistics are recorded BEFORE the structural update
    # of the final epoch, so they correspond exactly to the model
    # that produced final_test_acc.
    final_test_acc = None
    final_test_loss = None
    final_neurons = None
    final_total_born = None
    final_total_pruned = None

    # ============================================================
    # Training
    # ============================================================
    train_loss_history = []
    train_acc_history = []

    test_loss_history = []
    test_acc_history = []
    for epoch in range(cfg.epochs):

        # --------------------------------------------------------
        # Train
        # --------------------------------------------------------

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        # --------------------------------------------------------
        # Test
        #
        # At this point the current architecture is exactly the
        # architecture that produces test_acc.
        # --------------------------------------------------------

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )
        train_loss_history.append(float(train_loss))
        train_acc_history.append(float(train_acc))

        test_loss_history.append(float(test_loss))
        test_acc_history.append(float(test_acc))
        # --------------------------------------------------------
        # Snapshot current evaluated architecture
        # --------------------------------------------------------

        eval_neurons = model.num_neurons
        eval_total_born = len(model.birth_history)
        eval_total_pruned = len(model.death_history)

        # --------------------------------------------------------
        # Save best model + developmental state
        # --------------------------------------------------------

        if test_acc > best_acc:

            best_acc = test_acc
            best_loss = test_loss
            best_epoch = epoch + 1

            best_neurons = eval_neurons
            best_total_born = eval_total_born
            best_total_pruned = eval_total_pruned

            torch.save(
                {
                    "state_dict": model.state_dict(),

                    "dataset": cfg.dataset,
                    "seed": cfg.seed,

                    "initial_hidden_neurons": cfg.hidden_neurons,

                    "epoch": best_epoch,

                    "test_acc": best_acc,
                    "test_loss": best_loss,

                    "hidden_neurons": best_neurons,

                    "total_born": best_total_born,
                    "total_pruned": best_total_pruned,
                },
                save_dir / "best_model.pt",
                )

        # --------------------------------------------------------
        # Save FINAL evaluated state
        #
        # This is BEFORE the final structural update.
        # Therefore all values below refer to exactly the
        # architecture that produced final_test_acc.
        # --------------------------------------------------------

        if epoch == cfg.epochs - 1:

            final_test_acc = test_acc
            final_test_loss = test_loss

            final_neurons = eval_neurons

            final_total_born = eval_total_born
            final_total_pruned = eval_total_pruned

        # --------------------------------------------------------
        # End-of-epoch flux statistics
        # --------------------------------------------------------

        stats = model.end_epoch()

        # --------------------------------------------------------
        # Neurogenesis / pruning
        # --------------------------------------------------------

        # --------------------------------------------------------
        # Neurogenesis / pruning
        # --------------------------------------------------------

        update = model.apply_structural_updates()

        # --------------------------------------------------------
        # Recreate optimizer if network structure changed
        # --------------------------------------------------------

        if update.changed:

            # Explicitly release the old optimizer and its state.
            # Structural changes replace model parameters, so the old
            # Adam state is no longer needed.
            del optimizer

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )

        # --------------------------------------------------------
        # Print epoch statistics
        #
        # NOTE:
        # Alive / Total Born / Total Pruned here are AFTER the
        # structural update of the current epoch.
        # --------------------------------------------------------

        print(
            f"Epoch {epoch+1:02d}/{cfg.epochs} | "
            f"Train {train_acc:.2f}% | "
            f"Test {test_acc:.2f}% | "
            f"Flux {stats['network_mean_flux']:.4f} | "
            f"Birth {update.births} | "
            f"Pruned {update.deaths} | "
            f"Alive {update.after} | "
            f"Total Born {len(model.birth_history)} | "
            f"Total Pruned {len(model.death_history)}"
        )
        # --------------------------------------------------------
        # Memory cleanup
        # --------------------------------------------------------

        del train_loss, train_acc
        del test_loss, test_acc
        del stats, update
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    # ============================================================
    # Save final developmental statistics
    # ============================================================

    model.save_statistics(
        save_dir / "statistics.pt"
    )
    torch.save(
        {
            "epoch": list(range(1, cfg.epochs + 1)),
            "train_loss": train_loss_history,
            "train_accuracy": train_acc_history,
            "test_loss": test_loss_history,
            "test_accuracy": test_acc_history,

            "initial_neurons": cfg.hidden_neurons,
            "birth_rate": cfg.max_births_per_epoch,
            "death_rate": cfg.max_deaths_per_epoch,
            "seed": cfg.seed,
        },
        save_dir / "learning_curves.pt",
        )

    # ============================================================
    # Save final POST-UPDATE network
    #
    # This is kept separately because the model has already
    # undergone the structural update after the final evaluation.
    # ============================================================

    torch.save(
        {
            "state_dict": model.state_dict(),

            "dataset": cfg.dataset,
            "seed": cfg.seed,

            "initial_hidden_neurons": cfg.hidden_neurons,

            "hidden_neurons": model.num_neurons,

            "total_born": len(model.birth_history),
            "total_pruned": len(model.death_history),
        },
        save_dir / "final_model.pt",
        )

    # ============================================================
    # Save compact experiment summary
    # ============================================================

    summary = {

        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "initial_neurons": cfg.hidden_neurons,
        "max_births_per_epoch": cfg.max_births_per_epoch,
        "max_deaths_per_epoch": cfg.max_deaths_per_epoch,
        # --------------------------------------------------------
        # Best evaluated state
        # --------------------------------------------------------

        "best_accuracy": best_acc,
        "best_loss": best_loss,
        "best_epoch": best_epoch,

        "best_neurons": best_neurons,

        "best_total_born": best_total_born,
        "best_total_pruned": best_total_pruned,

        # --------------------------------------------------------
        # Final evaluated state
        # --------------------------------------------------------

        "final_accuracy": final_test_acc,
        "final_loss": final_test_loss,
        "final_epoch": cfg.epochs,

        "final_neurons": final_neurons,

        "final_total_born": final_total_born,
        "final_total_pruned": final_total_pruned,

        # --------------------------------------------------------
        # Post-final-update developmental state
        #
        # Useful diagnostically, but NOT paired with
        # final_accuracy.
        # --------------------------------------------------------

        "post_update_final_neurons": model.num_neurons,
        "post_update_total_born": len(model.birth_history),
        "post_update_total_pruned": len(model.death_history),
    }

    torch.save(
        summary,
        save_dir / "summary.pt",
        )

    # ============================================================
    # Final summary
    # ============================================================

    print("\n========================================")

    print("BEST")
    print(f"Accuracy      : {best_acc:.2f}%")
    print(f"Epoch         : {best_epoch}")
    print(f"Neurons       : {best_neurons}")
    print(f"Total born    : {best_total_born}")
    print(f"Total pruned  : {best_total_pruned}")

    print()

    print("FINAL")
    print(f"Accuracy      : {final_test_acc:.2f}%")
    print(f"Epoch         : {cfg.epochs}")
    print(f"Neurons       : {final_neurons}")
    print(f"Total born    : {final_total_born}")
    print(f"Total pruned  : {final_total_pruned}")

    print()

    print("POST-FINAL STRUCTURAL UPDATE")
    print(f"Neurons       : {model.num_neurons}")
    print(f"Total born    : {len(model.birth_history)}")
    print(f"Total pruned  : {len(model.death_history)}")

    print("========================================")


if __name__ == "__main__":
    main()