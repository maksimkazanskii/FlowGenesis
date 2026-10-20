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
from torch.utils.data import DataLoader, Subset
from src_self_organization.model_neurogenesis_new import NeuralContainer
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
    """Load AG News and represent text with train-fitted sparse TF-IDF."""
    from functools import partial
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from torch.utils.data import Dataset
    from datasets import load_dataset
    import joblib

    class SparseTextDataset(Dataset):
        def __init__(self, matrix, labels):
            self.matrix = matrix.tocsr()
            self.labels = np.asarray(labels, dtype=np.int64)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return int(idx), int(self.labels[idx])

    def collate_sparse(batch, *, dataset):
        indices, labels = zip(*batch)
        # Densify only the current mini-batch, never the entire dataset.
        features = dataset.matrix[list(indices)].toarray().astype(np.float32)
        return torch.from_numpy(features), torch.tensor(labels, dtype=torch.long)

    cache_dir = Path(cfg.text_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"tfidf_{cfg.tfidf_max_features}_ngram{cfg.tfidf_ngram_max}"
    train_path = cache_dir / f"{cache_key}_train.npz"
    test_path = cache_dir / f"{cache_key}_test.npz"
    labels_path = cache_dir / f"{cache_key}_labels.npz"
    vectorizer_path = cache_dir / f"{cache_key}_vectorizer.joblib"

    if all(p.exists() for p in (train_path, test_path, labels_path, vectorizer_path)):
        print(f"Loading cached AG News TF-IDF from {cache_dir}", flush=True)
        X_train = sparse.load_npz(train_path)
        X_test = sparse.load_npz(test_path)
        with np.load(labels_path) as saved:
            y_train = saved["train"]
            y_test = saved["test"]
    else:
        print("Downloading AG News and fitting training-only TF-IDF...", flush=True)
        raw = load_dataset("ag_news")
        vectorizer = TfidfVectorizer(
            max_features=cfg.tfidf_max_features,
            ngram_range=(1, cfg.tfidf_ngram_max),
            dtype=np.float32,
            sublinear_tf=True,
            lowercase=True,
        )
        X_train = vectorizer.fit_transform(raw["train"]["text"]).tocsr()
        X_test = vectorizer.transform(raw["test"]["text"]).tocsr()
        y_train = np.asarray(raw["train"]["label"], dtype=np.int64)
        y_test = np.asarray(raw["test"]["label"], dtype=np.int64)
        sparse.save_npz(train_path, X_train)
        sparse.save_npz(test_path, X_test)
        np.savez(labels_path, train=y_train, test=y_test)
        joblib.dump(vectorizer, vectorizer_path)

    if X_train.shape[1] != cfg.input_dim:
        raise ValueError(
            f"TF-IDF has {X_train.shape[1]} features but config input_dim={cfg.input_dim}"
        )
    train_dataset = SparseTextDataset(X_train, y_train)
    test_dataset = SparseTextDataset(X_test, y_test)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=partial(collate_sparse, dataset=train_dataset),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=partial(collate_sparse, dataset=test_dataset),
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
        choices=["agnews"],
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

    parser.add_argument('--ablation', choices=['regime'], default='regime')
    parser.add_argument('--value', choices=['static'], default='static')
    parser.add_argument('--output_dir', default='data/neurogenesis/static_sweep')
    args = parser.parse_args()

    # ============================================================
    # Configuration
    # ============================================================

    if args.dataset == "agnews":
        from src_self_organization.config_agnews import Config

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    cfg = Config()

    cfg.dataset = args.dataset
    cfg.hidden_neurons = args.hidden_neurons
    cfg.seed = args.seed
    # Static baseline: no structural changes, same recurrent model and AG News config.
    cfg.use_neurogenesis = False
    cfg.use_pruning = False
    cfg.max_births_per_epoch = 0
    cfg.max_deaths_per_epoch = 0


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
        f"Rate=16 | "
        f"Seed={cfg.seed} | "
        f"Device={device}"
    )

    device = torch.device(cfg.device)

    # ============================================================
    # Data
    # ============================================================

    train_loader, test_loader = get_dataloaders(cfg)

    # ============================================================
    # Fixed probe set for structural Flux
    # ============================================================

    probe_size = min(512, len(train_loader.dataset))

    # Probe loader unused in static mode.

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

    save_dir = (
            Path(args.output_dir) / "agnews" / "regime" / "value_static"
            / cfg.dataset.lower()
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
    # Training
    # ============================================================
    train_loss_history = []
    train_acc_history = []

    test_loss_history = []
    test_acc_history = []
    structure_events = []
    examples_history = []
    neuron_history = []

    def structural_check(current_optimizer):
        stats = model.end_epoch()  # aggregates the latest example window
        update = model.apply_structural_updates()
        structure_events.append({
            "examples_seen": model.examples_seen,
            "flux": stats["network_mean_flux"],
            "births": update.births,
            "deaths": update.deaths,
            "neurons": update.after,
        })
        if update.changed:
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
            None,
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
        neuron_history.append(int(model.num_neurons))

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
            f"Total Pruned {len(model.death_history)}"
        )
        # --------------------------------------------------------
        # Memory cleanup
        # --------------------------------------------------------

        del train_loss, train_acc
        del test_loss, test_acc

        cleanup_memory(
            device,
            clear_cache=(epoch + 1) % 10 == 0,
        )

    # ============================================================
    # Save final developmental statistics
    # ============================================================

    model.save_statistics(
        save_dir / "statistics.pt"
    )
    torch.save(
        {
            "epoch": list(range(1, cfg.epochs + 1)),
            "examples_seen": examples_history,
            "structure_events": structure_events,
            "num_neurons": neuron_history,
            "train_loss": train_loss_history,
            "train_accuracy": train_acc_history,
            "test_loss": test_loss_history,
            "test_accuracy": test_acc_history,

            "initial_neurons": cfg.hidden_neurons,
            "rate": cfg.max_births_per_epoch,
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