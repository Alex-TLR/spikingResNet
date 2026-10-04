# Loss Accuracy Configurations

These YAML files define the common dataset, model, optimizer, and runtime settings
used by `tools/loss_accuracy_sweep.py`. The script overrides only loss-related
fields for each requested variant. Final weights and resumable checkpoints are
stored in `weights/spike/loss/`. Sweep data is stored as JSON under `results/`,
with a human-readable TXT report generated beside it using the same filename.

The arrays under `experiment` define the complete sweep grid:

```yaml
experiment:
  seeds: [42, 1984, 2026]
  expansions: [1, 5, 100]
  resnet_models: [4, 10, 18]
```

Every loss runs for every model, expansion, and seed combination. The JSON
retains each individual seed result under
`results/<loss>/<model>/<expansion>/<seed>`. Its `summary` section and the TXT
report contain the mean and population standard deviation across completed
seeds for each loss/model/expansion group. Both outputs are rewritten after
every individual test run. On restart, the script loads the existing JSON and
skips completed loss/model/expansion/seed entries, so execution continues from
the unfinished combinations.

Historical CLI names are accepted but receive explicit artifact names:

- `rate_loss` becomes `ce_spikes_per_timestep_<population-reduction>`.
- `count_loss` becomes `ce_spikes_spike_count_sum`, matching snnTorch's
  population-count semantics.
- `cross_entropy` becomes `ce_<source>_<mode>_<population-reduction>`.
- `mse_count_loss` remains unchanged because it is not cross-entropy.

Equivalent historical and `--all-ce` variants are deduplicated. The JSON and
TXT report include configured accuracy, computed with the CE source and mode
used for training, alongside common spike-count and temporal-mean membrane
accuracies.

The `training.fit` field controls both the backpropagation path and CIFAR input
normalization. Generated logits and membrane CE variants set `fit: membrane`
and use standard CIFAR statistics; spike-source CE and MSE count variants set
`fit: spike` and preserve the existing identity normalization.
Corrected logits-CE weights include `_cifar_norm` in their artifact tag so
weights trained earlier with identity normalization are not reused.

Available base configurations:

- `cifar10.yaml`: CIFAR-10, spike-ResNet18, four timesteps, expansion 1.
- `cifar100.yaml`: CIFAR-100, spike-ResNet18, four timesteps, expansion 1.

Run the historical loss sweep:

```bash
python tools/loss_accuracy_sweep.py \
  --config experiments/loss_accuracy/cifar10.yaml
```

Run every valid explicit CE source and temporal mode:

```bash
python tools/loss_accuracy_sweep.py \
  --config experiments/loss_accuracy/cifar10.yaml \
  --all-ce \
  --population-reduction mean
```

Change the model, expansion, and seed arrays or other experiment settings in a
new YAML file rather than editing a completed experiment configuration. Use a
new `experiment.case` value so generated weights remain isolated.
