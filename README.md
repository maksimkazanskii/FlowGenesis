# FlowGenesis

**FlowGenesis** is an experimental framework for studying adaptive neural networks whose structure can change during training. The project investigates neuron birth and removal, activity-dependent structural adaptation, connectivity, and the behavior of evolving representations across classification and continual-learning tasks.

The repository contains training code, controlled ablations, generalization comparisons, progressive-complexity experiments, continual-learning experiments, statistical aggregation, and visualization utilities.

> **Scope:** This README describes the active project structure shown below. Historical code and files with `legacy` in their names are intentionally omitted. Exact model equations, CLI defaults, and experimental findings should be taken from the corresponding source code and paper; they are not inferred from filenames.

## Research questions

The experimental code is organized around several questions:

- **Structural adaptation:** How do neuron birth, death, activity, and connectivity settings affect network size and predictive performance?
- **Selection mechanisms:** How does the choice of neurons for structural interventions affect performance and stability?
- **Capacity and generalization:** How do adaptive models compare across initial widths, datasets, and static-network configurations?
- **Changing task demands:** Can network complexity adapt when task complexity changes?
- **Continual learning:** How do structurally adaptive models behave over sequences of tasks?
- **Learned representations:** What activation patterns and functional structure emerge in the trained networks?

## Datasets and experiments

The codebase includes experiments for **FashionMNIST**, **CIFAR-10**, **AG News**, and continual-learning configurations using CIFAR data. Additional scripts cover rotating-MNIST and structural-recovery experiments. Dataset preprocessing, splits, and task definitions are specified by their respective training scripts and configuration modules.

| Experiment family | Main location | Purpose |
| --- | --- | --- |
| Main training and ablations | Repository root | Train adaptive models and vary structural hyperparameters |
| Causal selection experiments | `causal_experiments/` | Compare neuron-selection interventions and analyze their outcomes |
| CIFAR generalization | `generalization_cifar/` | Study initial-width and generalization settings |
| AG News generalization | `generalization_news/` | Compare adaptive and static settings on text classification |
| Progressive complexity | `progressive_complexity/` | Evaluate adaptation under changing complexity |
| Progressive task sequences | `progressive_flowgenesis_20tasks/`, `progressive_flowgenesis_50tasks/` | Study longer sequences of tasks |
| Continual learning | `continuous_learning/` | Train and evaluate self-organizing models in continual-learning settings |
| Statistics | `stats/` | Aggregate results and compare settings |
| Visualization | `vis/` | Plot experimental outcomes and inspect learned representations |

## Repository structure

```text
src_self_organization/
├── config_fm.py
├── config_cifar.py
├── config_agnews.py
├── model_neurogenesis_new.py
├── train_neurogenesis.py
├── train_neurogenesis_ablation.py
├── train_flowgenesis_causal.py
├── train_rotating_mnist.py
├── train_structural_recovery.py
├── _train_activity_fraction_ablation.py
├── _train_death_fraction_ablation.py
├── run_*.sh
├── run_static_sweep.py
├── causal_experiments/
│   ├── config_fm.py
│   ├── model_neurogenesis_new.py
│   ├── train_flowgenesis_causal.py
│   ├── stats.py
│   └── vis_causal.py
├── continuous_learning/
│   ├── model_neurogenesis_cl_selforg.py
│   ├── train_cl_selforg.py
│   └── run_cl_selforg.sh
├── generalization_cifar/
│   └── train_neurogenesis_n0_ablation.py
├── generalization_news/
│   ├── train_neurogenesis_n0_ablation.py
│   ├── train_static_agnews.py
│   └── run_static_sweep_agnews.py
├── progressive_complexity/
│   ├── config_cifar.py
│   ├── model_neurogenesis_new.py
│   └── train_complexity_cycle.py
├── progressive_flowgenesis_20tasks/
│   ├── config_cifar.py
│   ├── model_neurogenesis_new.py
│   ├── train.py
│   └── vis_progressive.py
├── progressive_flowgenesis_50tasks/
│   ├── config_cifar.py
│   ├── model_neurogenesis_new.py
│   └── train.py
├── stats/
│   └── [aggregation and comparison scripts]
└── vis/
    └── [experiment and architecture visualization scripts]
```

Generated `__pycache__/` directories are excluded from this overview.

## Core components

### Model and configuration

- `model_neurogenesis_new.py` — main model implementation for adaptive-network experiments.
- `config_fm.py`, `config_cifar.py`, `config_agnews.py` — dataset-specific configuration modules.
- `train_neurogenesis.py` — main training entry point.
- `train_neurogenesis_ablation.py` — structural hyperparameter ablations.

Some experiment directories contain local copies of model and configuration modules. These should be treated as experiment-specific implementations unless equivalence with the root modules has been verified.

### Structural ablations

The repository provides shell launchers for the following experimental dimensions:

| Launcher | Experimental dimension |
| --- | --- |
| `run_birth_rate.sh` | Neuron birth settings |
| `run_death.sh` | Neuron death settings |
| `run_activity_fraction.sh` | Activity-fraction settings |
| `run_connectivity.sh` | Connectivity settings |
| `run_n0.sh` | Initial neuron count |
| `run_rate.sh` | Structural update rate |
| `run_warmup.sh` | Warmup settings |
| `run_static.sh` | Static-model comparison |
| `run_agnews.sh` | AG News experiments |

The corresponding activity and death experiments also have dedicated Python entry points: `_train_activity_fraction_ablation.py` and `_train_death_fraction_ablation.py`.

### Causal selection experiments

`causal_experiments/` groups the training, statistical analysis, and visualization code for controlled neuron-selection comparisons. The experimental program includes high-flux, reverse-flux, and random selection conditions. Consult `causal_experiments/train_flowgenesis_causal.py` for the implemented intervention logic and available CLI arguments.

### Generalization and static baselines

`generalization_cifar/` and `generalization_news/` contain dataset-specific experiments. The AG News directory additionally includes `train_static_agnews.py` and `run_static_sweep_agnews.py` for static-model comparisons. The root `run_static_sweep.py` supports the broader static-sweep workflow.

### Progressive complexity and continual learning

`progressive_complexity/` contains the complexity-cycle training script. The `progressive_flowgenesis_20tasks/` and `progressive_flowgenesis_50tasks/` directories contain task-sequence training implementations, with a visualization script in the 20-task directory.

`continuous_learning/` contains a self-organizing continual-learning model, training script, and shell launcher. These experiments should not be conflated with standard single-task generalization benchmarks.

## Running experiments

Run module-based experiments from the **parent directory** of `src_self_organization` so that Python can resolve the package paths.

For example, inspect the available options for the main trainer:

```bash
python -m src_self_organization.train_neurogenesis --help
```

To inspect the ablation trainer:

```bash
python -m src_self_organization.train_neurogenesis_ablation --help
```

A selection-intervention example used in the FashionMNIST experiment workflow is:

```bash
python -u -m src_self_organization.train_flowgenesis_causal \
  --dataset fashionmnist \
  --hidden_neurons 20 \
  --ablation selection \
  --value high_flux \
  --seed 1
```

For shell launchers, review the configuration and output paths before executing them:

```bash
bash src_self_organization/run_death.sh
bash src_self_organization/run_activity_fraction.sh
```

These commands assume the repository's Python dependencies and datasets are already available. The precise installation requirements are not specified by the directory tree; refer to the project's environment or dependency specification, if provided.

## Statistical analysis

The `stats/` package contains scripts for aggregating and comparing results:

- `aggregate_birth.py`, `aggregate_death.py`, `aggregate_connectivity.py`, `aggregate_rate.py`, `aggregate_warmup.py` — experiment-specific aggregation.
- `aggregate_stats.py`, `stats.py`, `independent_statistics.py` — general and independent-setting statistics.
- `comparative_statistics.py`, `comparative_statistics_agnews.py`, `comparative_statistics_agnews_auto.py` — cross-model and dataset-specific comparisons.
- `cross_setting_statistics.py` — comparisons across experiment settings.
- `stats_flowgenesis_causal.py` — causal-experiment statistics.

The separate `causal_experiments/stats.py` analyzes the selection experiments within that experiment family.

## Visualization

The `vis/` directory contains:

- **Ablation plots:** `vis_birth_rate.py`, `vis_death_rate.py`, `vis_connectivity.py`, `vis_rate.py`, `vis_warmup.py`.
- **Width and dataset comparisons:** `vis_n0.py`, `vis_n0_cifar.py`, `vis_n0_agnews.py`.
- **Activation and selectivity analysis:** `vis_fashionmnist_activation.py`, `vis_cifar_activation.py`, `vis_cifar_selectivity.py`.
- **Spatial and transformation sensitivity:** `vis_cifar_spatial_sensitivity.py`, `vis_cifar_translation_sensitivity.py`.
- **Learned architecture:** `visualize_architecture.py`, `visualize_architecture_fixed_mnist.py`, `visualize_architecture_fixed_cifar.py`, `visualize_architecture_fixed_agnews.py`.

Additional visualization scripts include `causal_experiments/vis_causal.py` and `progressive_flowgenesis_20tasks/vis_progressive.py`.

## Reproducibility notes

For interpretable comparisons, record the dataset, configuration, initial width, intervention setting, random seed, checkpoint-selection procedure, and software environment for every run. When comparing settings, use matched evaluation protocols and report variability across seeds rather than only a single run.

The training scripts and their associated configuration modules are the authoritative sources for parameter defaults and output locations. The README intentionally does not claim specific performance results, dependency versions, or checkpoint formats that cannot be verified from the supplied repository tree.

## Project status

FlowGenesis is an active research codebase. Its experiment directories include both single-task structural adaptation studies and extensions to generalization, changing task complexity, and continual learning. The experimental results and theoretical claims should be interpreted alongside the current manuscript and reproducibility artifacts.
