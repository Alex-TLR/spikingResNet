# Test Suite

This directory contains deterministic unit tests for the repository's core
training configuration, SNN loss adaptation, model output contracts, and loss
accuracy sweep orchestration. The tests run on CPU and do not download datasets
or perform full model training.

## Running All Tests

From the repository root, run:

```bash
python -m unittest discover -s tests -v
```

To use the repository's Conda environment explicitly:

```bash
/home/alex/anaconda3/envs/Alex/bin/python -m unittest discover -s tests -v
```

## Test Modules

| Module | Coverage |
|---|---|
| `test_config.py` | YAML loading, defaults, legacy CE mapping, explicit CE options, and invalid files/options |
| `test_data_normalization.py` | Fit-controlled CIFAR normalization for membrane and spike training paths |
| `test_snn_loss.py` | Temporal CE modes, population aggregation, validation, score calculation, and gradient propagation |
| `test_spikeresnet_logits.py` | Four-output compatibility, explicit pre-LIF logits, output shape, and classifier gradients |
| `test_loss_accuracy_sweep.py` | Weight naming, CE variants, Cartesian model/expansion/seed execution, weight reuse, nested JSON persistence, and seed aggregation |

## Design Rules

- Keep unit tests deterministic and independent of CUDA.
- Do not download or load full datasets in unit tests.
- Use temporary directories for files, weights, and JSON output.
- Mock training and evaluation when testing orchestration.
- Add a regression test for every corrected loss, configuration, or model-output bug.