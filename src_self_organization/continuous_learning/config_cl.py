"""Split CIFAR-10 continual-learning configuration for FlowGenesis."""
from dataclasses import dataclass
from ..config_cifar import Config as CIFARConfig

MODELS = {
    "static20": (20, False, False),
    "static80": (80, False, False),
    "full": (20, True, True),
    "no_birth": (20, False, True),
    "no_death": (20, True, False),
}
SEEDS = (0, 1, 2, 3, 4)


def make_config(model_name: str, seed: int, epochs_per_task: int = 20):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name}; choose {list(MODELS)}")
    cfg = CIFARConfig()
    n0, births, deaths = MODELS[model_name]
    cfg.dataset = "cifar10"
    cfg.input_dim = 3 * 32 * 32
    cfg.output_dim = 10
    cfg.hidden_neurons = n0
    cfg.seed = seed
    cfg.use_neurogenesis = births
    cfg.use_pruning = deaths
    cfg.max_births_per_epoch = 32 if births else 0
    cfg.max_deaths_per_epoch = 32 if deaths else 0
    cfg.epochs_per_task = epochs_per_task
    cfg.num_tasks = 5
    # Preserve the original CIFAR hyperparameters. These relative criteria
    # are used by the uploaded model but absent from config_cifar.py.
    cfg.death_activity_fraction = getattr(cfg, "death_activity_fraction", 0.10)
    cfg.death_flux_fraction = getattr(cfg, "death_flux_fraction", 0.20)
    cfg.death_flux_max = getattr(cfg, "death_flux_max", 0.13)
    return cfg
