# SpikingResNet — Energy-Efficient OoD Detection with Spike-Like Models

This repository contains the code accompanying the paper:

> **Out-of-Distribution Detection with Spike-Like Networks: Population Coding, Training Depth, and Single-Step Feature Representations**  
> A. Avramović, S. Gajić, V. Jovanović, V. Risojević, D. Sluga

Spike-like models replace standard ReLU activations with leaky integrate-and-fire (LIF) neurons on conventional ResNet backbones, operating at a single inference time step to achieve binary spike activations and substantially reduced arithmetic complexity, while remaining competitive with full-precision CNN baselines on OpenOOD benchmarks.

Key results on CIFAR-10 (ResNet-18, single-step inference):
- **93.91% / 96.63% AUROC** (Near-OoD / Far-OoD) with `T=1`, `E=5` — competitive with OpenOOD top-5
- **97.64% / 98.86% AUROC**, **FPR95 12.34% / 5.41%** with `T_train=2`, `E=50` — outperforms all listed SOTA baselines

On CIFAR-100 (ResNet-18): **88.89% / 93.39% AUROC**, exceeding the top CNN entry on the OpenOOD leaderboard.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Pre-trained Weights](#pre-trained-weights)
- [Datasets](#datasets)
- [Configuration Reference](#configuration-reference)
- [Training](#training)
- [Testing & Evaluation](#testing--evaluation)
- [Experiment Modes](#experiment-modes)
- [Running on a SLURM Cluster](#running-on-a-slurm-cluster)
- [Outputs](#outputs)
- [Citation](#citation)

---

## Project Structure

```
spikingResNet/
├── main.py                  # Entry point for all modes
├── train.py                 # Training loop
├── test.py                  # Accuracy evaluation
├── feature.py               # Feature extraction pipeline
├── config.py                # Config dataclass and YAML loader
├── models/                  # Network architectures
│   ├── my_spikeresnet.py    # Spike-like ResNet (LIF activations)
│   ├── resnet.py            # Standard ResNet backbone
│   ├── plain.py             # ConvNet backbone
│   └── resnet9.py           # ResNet-9 variant
├── metrics/
│   └── Metrics.py           # AUROC/FPR95 computation and OoD statistics
├── utils/
│   └── Utils.py             # I/O utilities
├── experiments/
│   ├── train/               # Training config files
│   └── test/                # Test/evaluation config files
├── data/                    # Datasets (auto-downloaded or pre-placed)
├── weights/                 # Saved model weights (auto-created)
│   ├── spike/
│   └── conv/
├── features/                # Extracted feature vectors (auto-created)
│   ├── spike/
│   └── conv/
├── results/                 # Output metrics and JSON results (auto-created)
├── tools/                   # Post-processing and plotting scripts
├── runTrain.sh              # SLURM training submission script
└── runTest.sh               # SLURM test submission script
```

---

## Requirements

Install dependencies via pip:

```bash
pip install torch torchvision snntorch pyyaml numpy scikit-learn
```

A CUDA-capable GPU is strongly recommended. The code falls back to CPU automatically if no GPU is available.

---

## Pre-trained Weights

Pre-trained model weights are available on Hugging Face:

> **[https://huggingface.co/dsluga/spikingresnet](https://huggingface.co/dsluga/spikingresnet)**

Download the desired `.pth` files and place them under `weights/spike/` or `weights/conv/` according to the model type before running evaluation.

---

## Datasets

Most datasets are downloaded automatically by torchvision on first use. Place or symlink them under `data/` with the following structure:

```
data/
├── cifar10/
├── cifar100/
├── mnist/
├── fmnist/        # FashionMNIST
├── kmnist/
├── svhn/
├── textures/      # DTD — download from https://www.robots.ox.ac.uk/~vgg/data/dtd/
├── places/        # Places365 — download from http://places2.csail.mit.edu/
└── tImage200/     # Tiny ImageNet-200 — download from http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

---

## Configuration Reference

All runs are controlled by YAML files under `experiments/`. Key parameters:

| Parameter | Description | Options |
|---|---|---|
| `dataset.id` | In-distribution dataset | `CIFAR10`, `CIFAR100`, `SVHN`, `MNIST`, ... |
| `dataset.features` | Datasets for feature extraction | list of dataset names |
| `model.type` | Network type | `spike` (LIF), `conv` (standard CNN) |
| `model.resnet_model` | Backbone depth | `4` (ConvNet), `10` (ResNet-10), `18` (ResNet-18) |
| `model.expansion` | Population coding expansion factor | `1` = no expansion |
| `model.num_time_steps_train` | Training time steps T | `1`, `2`, `4`, `8` |
| `model.num_time_steps_extract` | Inference time steps | `1` for single-step inference |
| `training.loss` | Loss function | `mse_count_loss`, `count_loss`, `rate_loss`, `cross_entropy` |
| `training.ce_source` | Tensor supervised by `cross_entropy` | `logits`, `membranes`, `spikes` |
| `training.ce_mode` | CE temporal aggregation | `temporal_mean`, `per_timestep`, `final_timestep`, `spike_rate`, `spike_count` |
| `training.population_reduction` | Class-population aggregation | `mean`, `sum` |
| `training.auto_aug` | AutoAugment | `true` / `false` |
| `training.epochs` | Training epochs | 400 (with aug), 200 (without) |
| `optimizer.name` | Optimiser | `adam`, `sgd` |
| `optimizer.learning_rate` | Learning rate | default `2e-4` |
| `mode` | Run mode | `train`, `test` |
| `test_type` | Evaluation type | `standard`, `accuracy`, `experiment_1`, `experiment_2`, `experiment_3` |
| `experiment.seeds` | Random seeds for multi-run averaging | e.g. `[42, 1987, 1991, 2020, 2024]` |
| `experiment.expansions` | Population coding expansion sweep | e.g. `[1, 5, 10, 25, 50]` |
| `experiment.resnet_models` | Backbone sweep | e.g. `[4, 10, 18]` |

---

## Training

### Single model

Set `mode: train` in your config file and run:

```bash
python main.py --config experiments/config_train.yaml
```

Ready-made training configs are provided under `experiments/train/`. For example, to train a spike-ResNet18 on CIFAR-10 with MSE loss and AutoAugment:

```bash
python main.py --config experiments/train/config_exp9_train_spike_mse_aug_cifar10.yaml
```

### Multi-seed / multi-architecture sweep

Set `experiment.seeds`, `experiment.expansions`, and `experiment.resnet_models` in the YAML to sweep multiple configurations in a single run:

```yaml
experiment:
  seeds: [42, 1987, 1991, 2020, 2024]
  expansions: [1, 5, 10, 25, 50]
  resnet_models: [4, 10, 18]
```

Trained weights are saved automatically under `weights/spike/` or `weights/conv/`.

---

## Testing & Evaluation

All evaluation modes use the same entry point:

```bash
python main.py --config experiments/<config_file>.yaml
```

### ID classification accuracy

Tests accuracy across a sweep of seeds, expansions, and architectures:

```bash
python main.py --config experiments/config_test_accuracy.yaml
```

### Experiment 1 — kNN on penultimate-layer features

Evaluates kNN-based OoD detection on membrane-potential features from the penultimate layer. Corresponds to **Table I** (CIFAR-10) in the paper.

```bash
python main.py --config experiments/config_test_experiment_1.yaml
```

Example sweep config:

```yaml
mode: test
test_type: experiment_1
experiment:
  methods: ['KNN']
  seeds: [42, 1987, 1991, 2020, 2024]
  expansions: [1, 5, 10, 25, 50]
  resnet_models: [4, 10, 18]
```

### Experiment 2 — Scoring-based post-hoc methods

Evaluates post-hoc OoD detectors (MSP, Energy, ODIN, MLS, ASH, ViM) on final-layer features. Corresponds to **Table II** (CIFAR-10) in the paper.

```bash
python main.py --config experiments/config_test_experiment_2.yaml
```

Example sweep config:

```yaml
mode: test
test_type: experiment_2
experiment:
  methods: ['ASH', 'MSP', 'ODIN', 'ENGY', 'MLS', 'VIM']
  seeds: [42]
  expansions: [1, 5, 10, 25, 50]
  resnet_models: [4, 10, 18]
```

### Experiment 3 — Multi-step training with single-step inference

Evaluates kNN and ViM across training time steps T ∈ {1, 2, 4, 8} while keeping inference at T=1. Corresponds to **Table III** (CIFAR-10) and **Table VII** (CIFAR-100) in the paper.

```bash
python main.py --config experiments/config_test_experiment_3.yaml
```

Example sweep config:

```yaml
mode: test
test_type: experiment_3
model:
  num_time_steps_train: 2   # Change to 1, 2, 4, or 8
  num_time_steps_extract: 1
experiment:
  methods_1: ['KNN']
  methods_2: ['VIM']
  seeds: [42]
  expansions: [1, 5, 10, 25, 50]
  resnet_models: [4, 10, 18]
```

---

## Experiment Modes

| `test_type` | Description |
|---|---|
| `standard` | Train → test accuracy → extract features → compute statistics (single config) |
| `accuracy` | Sweep seeds/expansions/models, report ID accuracy only |
| `experiment_1` | kNN OoD detection sweep on penultimate-layer membrane-potential features |
| `experiment_2` | Post-hoc scoring method sweep (ASH, MSP, ODIN, Energy, MLS, ViM) |
| `experiment_3` | Combined kNN + ViM sweep over training time steps T |

---

## Outputs

| Location | Contents |
|---|---|
| `weights/spike/` or `weights/conv/` | Saved model checkpoints (`.pth`) |
| `features/spike/` or `features/conv/` | Extracted feature vectors (`.npz`) |
| `results/` | JSON files with AUROC / FPR95 per method, backbone, expansion, and seed |

---

## Testing

Run the deterministic CPU unit tests with:

```bash
python -m unittest discover -s tests -v
```

The suite covers CE aggregation and validation, YAML configuration loading,
model output compatibility, explicit logits, and classifier gradient flow.

### Loss Accuracy Sweep

Compare the four named training losses using a base experiment configuration:

```bash
python tools/loss_accuracy_sweep.py \
  --config experiments/loss_accuracy/cifar10.yaml
```

For each loss, the script checks the corresponding file under
`weights/spike/loss/`. Missing weights trigger training; existing weights are
reused. The `experiment.resnet_models`, `experiment.expansions`, and
`experiment.seeds` YAML arrays define a Cartesian product, and every loss is
run at every point. Spike and membrane test accuracies are written after each
individual seed run to a nested JSON data file under `results/`. A TXT report
with one row per loss/model/expansion reports the mean and population standard
deviation across completed seeds. Both files share the same base filename. The
JSON is updated after each individual test; restarting with the same output and
experiment configuration skips entries already marked `completed`.

For CE runs, the report also includes configured accuracy using the same output
source and temporal reduction as training. Historical `rate_loss` and
`count_loss` inputs are stored with explicit names such as
`ce_spikes_per_timestep_mean` and `ce_spikes_spike_count_sum`; equivalent
explicit CE variants are run only once.

To additionally test every valid CE source and temporal mode:

```bash
python tools/loss_accuracy_sweep.py \
  --config experiments/loss_accuracy/cifar10.yaml \
  --all-ce \
  --population-reduction mean
```

Explicit CE variants use distinct weight tags such as
`ce_logits_temporal_mean_mean`, preventing checkpoints trained with different
CE semantics from sharing a filename. Use `--output results/name.json` to set
the JSON path, `--report results/name.txt` to set the report path, or
`--fail-fast` to stop after the first failed variant.

---

## Citation

The paper is currently under revision. Citation details will be updated upon publication. In the meantime, if you use this code, please credit the authors:

> A. Avramović, S. Gajić, V. Jovanović, V. Risojević, D. Sluga — *Out-of-Distribution Detection with Spike-Like Networks: Population Coding, Training Depth, and Single-Step Feature Representations*

