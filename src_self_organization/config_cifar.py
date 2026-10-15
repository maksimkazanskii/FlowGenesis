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
    dataset: str = "CIFAR10"
    output_dir: str = "data/neurogenesis/cifar10"
    # ============================================================
    # Dataset
    # ============================================================
    input_dim: int = 3 * 32 * 32
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

    birth_threshold: float = 0.2

    birth_persistence: int = 5000

    max_births_per_epoch: int = 32

    # ------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------

    death_flux_threshold: float = 0.005

    death_activity_threshold: float = 0.01

    death_persistence: int = 10000


    max_deaths_per_epoch: int = 32

    # ------------------------------------------------------------
    # Structural limits
    # ------------------------------------------------------------

    min_neurons: int = 50

    max_neurons: int = 900

    structure_warmup_examples: int = 20000

    structure_cooldown_examples: int = 10000

    structure_check_every_examples: int = 10000

    split_noise_std: float = 0.01

    birth_fraction = 0.2
    death_fraction = 0.2