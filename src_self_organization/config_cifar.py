from dataclasses import dataclass
import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:

    # ============================================================
    # General
    # ============================================================

    seed: int = 42
    deterministic: bool = True

    device: str = get_device()

    dataset: str = "CIFAR10"
    output_dir: str = "data/so"

    # ============================================================
    # Dataset
    # ============================================================

    input_dim: int = 3 * 32 * 32
    output_dim: int = 10

    batch_size: int = 32
    num_workers: int = 0

    # ============================================================
    # Training
    # ============================================================

    epochs: int = 250

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # ============================================================
    # Neural Container
    # ============================================================

    hidden_neurons: int = 50

    recurrent_connection_probability: float = 0.20
    self_connections: bool = False

    relaxation_steps: int = 10

    # ============================================================
    # Self-Organization
    # ============================================================

    use_flux: bool = True
    use_neurogenesis: bool = True
    use_pruning: bool = True

    # ------------------------------------------------------------
    # Flux
    # ------------------------------------------------------------

    flux_decay: float = 0.90
    activity_decay: float = 0.90
    capacity_decay: float = 0.99

    # ------------------------------------------------------------
    # Neurogenesis
    # ------------------------------------------------------------

    birth_threshold: float = 0.80
    birth_persistence: int = 5
    max_births_per_epoch: int = 16

    # ------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------

    death_activity_fraction: float = 0.1
    death_flux_fraction: float = 0.1

    death_persistence: int = 5

    # Fixed for CIFAR N0 ablation
    max_deaths_per_epoch: int = 16

    # ----------------x--------------------------------------------
    # Structural limits
    # ------------------------------------------------------------

    min_neurons: int = 20
    max_neurons: int = 1000

    structure_warmup_epochs: int = 3
    structure_cooldown_epochs: int = 1

    split_noise_std: float = 0.01

    birth_fraction = 0.2
    death_fraction = 0.2