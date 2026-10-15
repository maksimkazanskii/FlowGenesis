from dataclasses import dataclass
import torch


@dataclass
class Config:

    # ============================================================
    # General
    # ============================================================

    seed: int = 42
    deterministic: bool = True

    device: str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    dataset: str = "FashionMNIST"
    output_dir: str = "data/flow"

    # ============================================================
    # Dataset
    # ============================================================

    input_dim: int = 28 * 28
    output_dim: int = 10

    batch_size: int = 128
    num_workers: int = 4

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

    # Baseline used for the N0 ablation.
    # Individual birth-threshold experiments can override this.
    birth_threshold: float = 0.25

    birth_persistence: int = 3

    # August-3 baseline: maximum structural births per epoch.
    max_births_per_epoch: int = 32

    # ------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------

    # Relative pruning parameters used by the recovered model.
    death_activity_fraction: float = 0.10
    death_flux_fraction: float = 0.20
    death_flux_max: float = 0.13

    death_persistence: int = 5

    # August-3 baseline: maximum structural deaths per epoch.
    max_deaths_per_epoch: int = 32

    # ------------------------------------------------------------
    # Structural limits
    # ------------------------------------------------------------

    min_neurons: int = 20
    max_neurons: int = 300

    structure_warmup_epochs: int = 3
    structure_cooldown_epochs: int = 1

    split_noise_std: float = 0.01

    # ============================================================
    # Legacy / compatibility
    # ============================================================

    birth_fraction: float = 0.20
    death_fraction: float = 0.20