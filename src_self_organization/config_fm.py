from dataclasses import dataclass
import torch

@dataclass
class Config:

    # ============================================================
    # General
    # ============================================================

    seed: int = 42

    deterministic: bool = True

    device: str = "auto"

    dataset: str = "FashionMNIST"

    output_dir: str = "data/neurogenesis"

    # ============================================================
    # Dataset
    # ============================================================

    input_dim: int = 28 * 28

    output_dim: int = 10

    batch_size: int = 128

    num_workers: int = 0

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

    birth_persistence_epochs: int = 3

    max_births_per_epoch: int = 5

    # ------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------

    death_flux_threshold: float = 0.01

    death_activity_threshold: float = 0.01

    death_persistence_epochs: int = 5

    max_deaths_per_epoch: int = 5

    # ------------------------------------------------------------
    # Structural limits
    # ------------------------------------------------------------

    min_neurons: int = 10

    max_neurons: int = 300

    structure_warmup_epochs: int = 3

    structure_cooldown_epochs: int = 1

    split_noise_std: float = 0.01

    birth_fraction: float = 0.2
    death_fraction: float = 0.2

    # New model requires relative death criteria, unlike the original model.
    death_activity_fraction: float = 0.10
    death_flux_fraction: float = 0.20
    death_flux_max: float = 0.13

    # Derived from actual training-set size before constructing the model.
    birth_persistence: int = 10000
    death_persistence: int = 10000
    structure_warmup_examples: int = 100000
    structure_cooldown_examples: int = 10000
    structure_check_every_examples: int = 10000

    cache_clear_every_epochs: int = 10
    checkpoint_every_epochs: int = 10

    def configure_for_training_samples(self, samples_per_epoch: int):
        self.birth_persistence = (
                self.birth_persistence_epochs * samples_per_epoch
        )
        self.death_persistence = (
                self.death_persistence_epochs * samples_per_epoch
        )
        self.structure_warmup_examples = (
                self.structure_warmup_epochs * samples_per_epoch
        )
        self.structure_cooldown_examples = (
                self.structure_cooldown_epochs * samples_per_epoch
        )
        self.structure_check_every_examples = samples_per_epoch