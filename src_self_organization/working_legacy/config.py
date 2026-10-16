from dataclasses import dataclass
import torch
import gc

@dataclass
class Config:

    # ============================================================
    # General
    # ============================================================

    seed: int = 42

    deterministic: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    dataset: str = "FashionMNIST"

    output_dir: str = "data/neurogenesis"

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

    epochs: int = 100

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

    birth_threshold: float = 0.25

    birth_persistence: int = 3

    max_births_per_epoch: int = 5

    # ------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------

    death_flux_threshold: float = 0.01

    death_activity_threshold: float = 0.01

    death_persistence: int = 5

    max_deaths_per_epoch: int = 5

    # ------------------------------------------------------------
    # Structural limits
    # ------------------------------------------------------------

    min_neurons: int = 20

    max_neurons: int = 300

    structure_warmup_epochs: int = 3

    structure_cooldown_epochs: int = 1

    split_noise_std: float = 0.01

    birth_fraction = 0.2
    death_fraction = 0.2