"""
config.py

Experiment 0
==========================
Random Neural Container

Goal:
Determine whether an initially random recurrent neural graph
can learn MNIST before introducing any developmental dynamics.

Experiment 0 intentionally contains:

    ✓ no Flux
    ✓ no Capacity
    ✓ no Neurogenesis

Only SGD is used.
"""

from dataclasses import dataclass
import torch


@dataclass
class Config:

    # ============================================================
    # General
    # ============================================================

    seed: int = 42

    deterministic: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir: str = "results/container_mnist"

    experiment_name: str = "experiment0"

    # ============================================================
    # Dataset
    # ============================================================

    dataset: str = "FashionMNIST"

    input_dim: int = 28 * 28

    output_dim: int = 10

    batch_size: int = 128

    num_workers: int = 4

    # ============================================================
    # Training
    # ============================================================

    epochs: int = 20

    optimizer: str = "adam"

    learning_rate: float = 1e-3

    weight_decay: float = 1e-4

    gradient_clip: float = 5.0

    # ============================================================
    # Neural Container
    # ============================================================

    hidden_neurons: int = 100

    # probability of one directed recurrent edge
    recurrent_connection_probability: float = 0.20

    self_connections: bool = False

    learn_adjacency: bool = False

    # ============================================================
    # Relaxation Dynamics
    # ============================================================

    relaxation_steps: int = 10

    activation: str = "relu"

    # ============================================================
    # Initialization
    # ============================================================

    weight_init_std: float = 0.05

    bias_init: float = 0.0

    # ============================================================
    # Logging
    # ============================================================

    print_every: int = 100

    save_every_epoch: bool = True

    save_best_model: bool = True

    # ============================================================
    # Future SNC (disabled)
    # ============================================================

    use_flux: bool = False

    use_capacity: bool = True

    use_neurogenesis: bool = False

    use_pruning: bool = False

    flux_decay: float = 0.99

    capacity_alpha: float = 0.01

    capacity_beta: float = 0.001

    capacity_lambda: float = 0.5
    birth_threshold: float = 1.0


    death_threshold: float = 0.05