"""FlowGenesis ablation runner using the current model_neurogenesis_new and config_fm.

Supports beta, birth, death, rate, regime, warmup ablations,
and saves learning curves after every epoch for live visualization.
"""
import gc
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
from .model_neurogenesis_new import NeuralContainer
import argparse
from pathlib import Path



# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
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

def train_epoch(model, loader, optimizer, criterion, device, cfg, on_structure=None):

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

        if on_structure is not None and model.window_examples >= cfg.structure_check_every_examples:
            optimizer = on_structure(optimizer)

        total_loss += loss.item()

        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        # Release batch tensors as soon as possible
        del images, labels, logits, loss, pred

    return total_loss / len(loader), 100.0 * correct / total, optimizer


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
        choices=["fashionmnist", "cifar10", "mnist"],
        required=True,
    )

    parser.add_argument(
        "--hidden_neurons",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output root; defaults to data/neurogenesis/ablations")
    parser.add_argument("--reference_root", type=str, default=None,
                        help="Root containing high_flux reference summaries (required for random_all_matched)")
    parser.add_argument("--ablation", choices=["beta", "birth", "death", "rate", "regime", "warmup", "selection"],
                        required=True)
    parser.add_argument("--value", required=True,
                        help="beta/birth/death=float; rate=float (1, 2.5, 5, 10); warmup=int; regime=full/no_birth/no_death/static")

    args = parser.parse_args()

    # ============================================================
    # Configuration
    # ============================================================

    from .config_fm import Config

    cfg = Config()

    cfg.dataset = args.dataset
    cfg.hidden_neurons = args.hidden_neurons
    cfg.seed = args.seed
    cfg.input_dim = 3 * 32 * 32 if args.dataset == "cifar10" else 28 * 28


    # Preserve the original ablation definitions, adapted to current config.
    try:
        if args.ablation == "beta":
            value = float(args.value)
            if not 0 <= value <= 1:
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
            value = float(args.value)
            if value not in (1.0, 2.5, 5.0, 10.0):
                parser.error("--value for rate must be 1, 2.5, 5, or 10")
            # Integer per-check limits; fractional 2.5 alternates 2, 3.
            cfg.max_births_per_epoch = int(value)
            cfg.max_deaths_per_epoch = int(value)
        elif args.ablation == "regime":
            variant = args.value
            if variant not in ("full", "no_birth", "no_death", "static"):
                parser.error("--value for regime must be full/no_birth/no_death/static")
            cfg.structure_warmup_epochs = 5
            cfg.structure_warmup_examples = 5 * 60000 if args.dataset in ("mnist", "fashionmnist") else 5 * 50000
            cfg.max_births_per_epoch = 0 if variant in ("no_birth", "static") else 32
            cfg.max_deaths_per_epoch = 0 if variant in ("no_death", "static") else 32
            cfg.use_neurogenesis = variant not in ("no_birth", "static")
            cfg.use_pruning = variant not in ("no_death", "static")
        elif args.ablation == "selection":
            if args.value not in ("high_flux", "reverse_flux", "random_all", "random_all_matched"):
                parser.error("--value must be high_flux, reverse_flux, random_all, or random_all_matched")
            cfg.birth_selection = args.value
        elif args.ablation == "warmup":
            value = int(args.value)
            if value < 0:
                parser.error("--value for warmup must be >= 0")
            cfg.max_births_per_epoch = 16
            cfg.max_deaths_per_epoch = 16
            cfg.structure_warmup_examples = value
    except ValueError:
        parser.error(f"Invalid --value {args.value!r} for --ablation {args.ablation}")

    set_seed(cfg.seed)

    # CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    cfg.device = str(device)

    print(
        f"Dataset={cfg.dataset} | "
        f"N0={cfg.hidden_neurons} | "
        f"Rate={cfg.max_births_per_epoch} | "
        f"Seed={cfg.seed} | "
        f"Device={device}"
    )

    device = torch.device(cfg.device)

    # ============================================================
    # Data
    # ============================================================

    train_loader, test_loader = get_dataloaders(cfg)
    # Preserve epoch-equivalent structural timing, measured in examples.


    #cfg.configure_for_training_samples(len(train_loader.dataset))
    print(f"Structural timing (examples): birth={cfg.birth_persistence}, "
          f"death={cfg.death_persistence}, warmup={cfg.structure_warmup_examples}, "
          f"cooldown={cfg.structure_cooldown_examples}, "
          f"check={cfg.structure_check_every_examples}", flush=True)

    # ============================================================
    # Fixed probe set (retained for compatibility with original trainer).
    # ============================================================

    probe_size = min(512, len(train_loader.dataset))

    probe_dataset = Subset(
        train_loader.dataset,
        list(range(probe_size)),
    )

    probe_loader = DataLoader(
        probe_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    # ============================================================
    # Model
    # ============================================================

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

    safe_value = args.value.replace("/", "_").replace("\\", "_")
    output_root = Path(args.output_dir) if args.output_dir else Path(cfg.output_dir) / "ablations"
    save_dir = (output_root / cfg.dataset.lower() / args.ablation
                / f"value_{safe_value}" / f"neurons_{cfg.hidden_neurons}" / f"seed_{cfg.seed}")

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
    structure_events = []
    examples_history = []
    neuron_history = []

    def save_learning_curves():
        # Preserve the original filename and fields; refresh after each epoch.
        payload = {
            "epoch": list(range(1, len(test_acc_history) + 1)),
            "examples_seen": examples_history,
            "structure_events": structure_events,
            "train_loss": train_loss_history,
            "train_accuracy": train_acc_history,
            "test_loss": test_loss_history,
            "test_accuracy": test_acc_history,
            "num_neurons": neuron_history,
            "initial_neurons": cfg.hidden_neurons,
            "rate": float(args.value) if args.ablation == "rate" else cfg.max_births_per_epoch,
            "rate_unit": "per_structural_check",
            "seed": cfg.seed,
            "ablation": args.ablation,
            "ablation_value": args.value,
            "birth_selection": args.value if args.ablation == "selection" else "original",
        }
        destination = save_dir / "learning_curves.pt"
        temporary = save_dir / "learning_curves.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(destination)

    # Read the actual birth count at each structural check of the paired
    # high_flux run. This is a replay of a fixed reference schedule, not
    # a counterfactual trajectory with identical pruning or neuron counts.
    reference_events = None
    if args.ablation == "selection" and args.value == "random_all_matched":
        if not args.reference_root:
            parser.error("random_all_matched requires --reference_root")
        reference_path = (Path(args.reference_root) / cfg.dataset.lower()
                          / "selection" / "value_high_flux"
                          / f"neurons_{cfg.hidden_neurons}" / f"seed_{cfg.seed}"
                          / "summary.pt")
        if not reference_path.is_file():
            parser.error(f"Missing reference: {reference_path}; run high_flux first")
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        if (reference.get("dataset") != cfg.dataset
                or reference.get("seed") != cfg.seed
                or reference.get("initial_neurons") != cfg.hidden_neurons):
            parser.error(f"Reference metadata mismatch: {reference_path}")
        reference_events = reference["structure_events"]
        print(f"Matching {len(reference_events)} high_flux structural checks: {reference_path}", flush=True)

    rate_check_index = 0

    def structural_check(current_optimizer):
        nonlocal rate_check_index
        if args.ablation == "rate":
            requested_rate = float(args.value)
            base = int(requested_rate)
            # 2.5 -> 2, 3, 2, 3 ... over consecutive checks.
            per_check_limit = base + (rate_check_index % 2 if requested_rate == 2.5 else 0)
            cfg.max_births_per_epoch = per_check_limit
            cfg.max_deaths_per_epoch = per_check_limit
            rate_check_index += 1
        stats = model.end_epoch()  # aggregates the latest example window
        matched_target = None
        if reference_events is not None:
            index = len(structure_events)
            if index >= len(reference_events):
                raise RuntimeError("Matched run has more checks than reference")
            ref_event = reference_events[index]
            if int(ref_event["examples_seen"]) != model.examples_seen:
                raise RuntimeError(f"Structural check timing differs at check {index}: "
                                   f"reference={ref_event['examples_seen']}, current={model.examples_seen}")
            matched_target = int(ref_event["births"])
            model.max_births_per_epoch = matched_target
        update = model.apply_structural_updates()
        if matched_target is not None and update.births != matched_target:
            raise RuntimeError(
                f"Cannot match high_flux births at check {len(structure_events)} "
                f"(examples={model.examples_seen}): requested {matched_target}, "
                f"obtained {update.births}. Architecture/cooldown diverged. "
                f"Do not treat this as a matched comparison."
            )
        structure_events.append({
            "examples_seen": model.examples_seen,
            "flux": stats["network_mean_flux"],
            "births": update.births,
            "deaths": update.deaths,
            "neurons": update.after,
        })
        if update.changed:
            del current_optimizer
            new_optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
            return new_optimizer

        return current_optimizer

    for epoch in range(cfg.epochs):
        model.current_epoch = epoch + 1

        # --------------------------------------------------------
        # Train
        # --------------------------------------------------------

        train_loss, train_acc, optimizer = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            cfg,
            structural_check,
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
                    "examples_seen": model.examples_seen,

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

        # --------------------------------------------------------
        # Fixed-probe representation Flux
        # --------------------------------------------------------


        # --------------------------------------------------------
        # End-of-epoch statistics
        # --------------------------------------------------------

        # Structural checks are performed during train_epoch, not here.
        # Keep the residual window for the next epoch.
        update_events = [e for e in structure_events if e["examples_seen"] >
                         (examples_history[-1] if examples_history else 0)]
        births_this_epoch = sum(e["births"] for e in update_events)
        deaths_this_epoch = sum(e["deaths"] for e in update_events)
        examples_history.append(model.examples_seen)
        neuron_history.append(model.num_neurons)

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
            f"Examples {model.examples_seen} | "
            f"Birth {births_this_epoch} | "
            f"Pruned {deaths_this_epoch} | "
            f"Alive {model.num_neurons} | "
            f"Total Born {len(model.birth_history)} | "
            f"Total Pruned {len(model.death_history)}",
            flush=True,
        )
        # Make completed epochs available to visualization immediately.
        save_learning_curves()

        # --------------------------------------------------------
        # Memory cleanup
        # --------------------------------------------------------

        del train_loss, train_acc
        del test_loss, test_acc

        cleanup_memory(
            device,
            clear_cache=(epoch + 1) % 10 == 0,
        )

    if reference_events is not None and len(structure_events) != len(reference_events):
        raise RuntimeError(f"Matched run has {len(structure_events)} checks, "
                           f"reference has {len(reference_events)}")

    # ============================================================
    # Save final developmental statistics
    # ============================================================

    model.save_statistics(
        save_dir / "statistics.pt"
    )
    save_learning_curves()

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
            "examples_seen": model.examples_seen,
        },
        save_dir / "final_model.pt",
        )

    # ============================================================
    # Save compact experiment summary
    # ============================================================

    summary = {

        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "ablation": args.ablation,
        "ablation_value": args.value,
            "birth_selection": args.value if args.ablation == "selection" else "original",
        "initial_neurons": cfg.hidden_neurons,
        "max_births_per_epoch": cfg.max_births_per_epoch,
        "max_deaths_per_epoch": cfg.max_deaths_per_epoch,
        "requested_rate_per_check": float(args.value) if args.ablation == "rate" else None,
        "rate_unit": "per_structural_check",
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
        "final_examples_seen": model.examples_seen,
        "structure_events": structure_events,

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

def cleanup_memory(device, clear_cache=False):
    gc.collect()

    if clear_cache:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
if __name__ == "__main__":
    main()