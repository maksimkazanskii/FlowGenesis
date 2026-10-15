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
from torchvision import datasets, transforms
from .config_cifar import Config
from .model_neurogenesis import NeuralContainer
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

    if cfg.dataset.lower() == "cifar10":

        dataset_class = datasets.CIFAR10

        # No data augmentation in this first CIFAR-10 experiment:
        # only the dataset changes, while the NeuralContainer and
        # FlowGenesis mechanism remain unchanged.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ])

    else:

        raise ValueError(
            f"Unknown dataset: {cfg.dataset}"
        )

    train_dataset = dataset_class(
        "./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = dataset_class(
        "./data",
        train=False,
        download=True,
        transform=transform,
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

    # ============================================================
    # Arguments
    # ============================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hidden_neurons",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional root output directory. Existing experiment subfolders are appended underneath it.",
    )

    parser.add_argument(
        "--ablation",
        choices=["beta", "birth", "death", "rate", "regime", "warmup"],
        default=None,
        help="Optional ablation to run.",
    )

    parser.add_argument(
        "--value",
        default=None,
        help=(
            "Value for the selected ablation: "
            "beta=float, birth=float, death=float, rate=int, "
            "regime=full/no_birth/no_death/static, warmup=int."
        ),
    )

    args = parser.parse_args()

    # ============================================================
    # Configuration
    # ============================================================

    cfg = Config()

    if args.hidden_neurons is not None:
        cfg.hidden_neurons = args.hidden_neurons

    if args.seed is not None:
        cfg.seed = args.seed

    if args.ablation is not None and args.value is None:
        parser.error("--value is required when --ablation is specified")

    if args.ablation == "beta":
        value = float(args.value)
        if not 0.0 <= value <= 1.0:
            parser.error("--value for beta must be between 0 and 1")
        cfg.max_births_per_epoch = 16
        cfg.max_deaths_per_epoch = 16
        cfg.recurrent_connection_probability = value

    elif args.ablation == "birth":
        cfg.max_births_per_epoch = 32
        cfg.max_deaths_per_epoch = 32
        cfg.birth_threshold = float(args.value)
        cfg.death_flux_threshold = 0.01
        cfg.death_activity_threshold = 0.01

    elif args.ablation == "death":
        cfg.max_births_per_epoch = 32
        cfg.max_deaths_per_epoch = 32
        cfg.birth_threshold = 0.25
        cfg.death_flux_max = float(args.value)

    elif args.ablation == "rate":
        value = int(args.value)
        if value not in [2, 4, 8, 16, 32, 64, 128]:
            parser.error("--value for rate must be one of: 2, 4, 8, 16, 32, 64, 128")
        cfg.max_births_per_epoch = value
        cfg.max_deaths_per_epoch = value

    elif args.ablation == "regime":
        variant = args.value
        if variant not in ["full", "no_birth", "no_death", "static"]:
            parser.error(
                "--value for regime must be one of: "
                "full, no_birth, no_death, static"
            )

        cfg.warmup_epochs = 5

        if variant == "full":
            cfg.max_births_per_epoch = 32
            cfg.max_deaths_per_epoch = 32
        elif variant == "no_birth":
            cfg.max_births_per_epoch = 0
            cfg.max_deaths_per_epoch = 32
        elif variant == "no_death":
            cfg.max_births_per_epoch = 32
            cfg.max_deaths_per_epoch = 0
        elif variant == "static":
            cfg.max_births_per_epoch = 0
            cfg.max_deaths_per_epoch = 0

    elif args.ablation == "warmup":
        value = int(args.value)
        if value < 0:
            parser.error("--value for warmup must be >= 0")
        cfg.max_births_per_epoch = 16
        cfg.max_deaths_per_epoch = 16
        cfg.structure_warmup_epochs = value

    set_seed(cfg.seed)

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

    if args.ablation is None:
        relative_save_dir = (
                Path("data/so/neurogenesis")
                / cfg.dataset.lower()
                / f"neurons_{cfg.hidden_neurons}"
                / f"seed_{cfg.seed}"
        )
    else:
        safe_value = str(args.value).replace("/", "_")
        relative_save_dir = (
                Path("data/so/ablations")
                / cfg.dataset.lower()
                / args.ablation
                / f"value_{safe_value}"
                / f"neurons_{cfg.hidden_neurons}"
                / f"seed_{cfg.seed}"
        )

    if args.output_dir is None:
        save_dir = relative_save_dir
    else:
        save_dir = (
                Path(cfg.output_dir)
                / args.output_dir
                / f"neurons_{cfg.hidden_neurons}"
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
    # Learning curves
    # ============================================================

    learning_curves = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "num_neurons": [],
        "total_born": [],
        "total_pruned": [],
    }

    # ============================================================
    # Training
    # ============================================================

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

        # --------------------------------------------------------
        # Snapshot current evaluated architecture
        # --------------------------------------------------------

        eval_neurons = model.num_neurons
        eval_total_born = len(model.birth_history)
        eval_total_pruned = len(model.death_history)

        learning_curves["epoch"].append(epoch + 1)
        learning_curves["train_loss"].append(train_loss)
        learning_curves["train_accuracy"].append(train_acc)
        learning_curves["test_loss"].append(test_loss)
        learning_curves["test_accuracy"].append(test_acc)
        learning_curves["num_neurons"].append(eval_neurons)
        learning_curves["total_born"].append(eval_total_born)
        learning_curves["total_pruned"].append(eval_total_pruned)

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
            f"Birth Cand {stats['birth_candidates']} | "
            f"Death Cand Flux {stats['death_candidates_flux']} | "
            f"Death Cand Activity {stats['death_candidates_activity']} | "
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ============================================================
    # Save learning curves
    # ============================================================

    torch.save(
        learning_curves,
        save_dir / "learning_curves.pt",
        )

    # ============================================================
    # Save final developmental statistics
    # ============================================================

    model.save_statistics(
        save_dir / "statistics.pt"
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