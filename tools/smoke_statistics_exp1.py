import os
import numpy as np

# Monkeypatch the heavy operations inside the metrics module and run a fast dry-run of statistics_exp_1

import sys
import types

# Provide a minimal fake torch module so the metrics module can be imported in
# environments without PyTorch (we only need functions used at import/runtime
# for this smoke test; real runs should use a real PyTorch install).
fake_torch = types.SimpleNamespace()
fake_torch.cuda = types.SimpleNamespace()
fake_torch.cuda.empty_cache = lambda: None
fake_torch.no_grad = lambda : (lambda x: x)
fake_torch.Tensor = object
sys.modules['torch'] = fake_torch

# Provide a fake 'feature' module to avoid importing heavy dataset libraries (torchvision) during import
fake_feature_mod = types.SimpleNamespace()
def fake_feature_extraction(config):
    print(f"[smoke] fake feature_extraction_spike called (module-level) for dataset_feat={getattr(config, 'dataset_feat', None)} expansion={config.expansion} seed={config.seed}")
    return
fake_feature_mod.feature_extraction_spike = fake_feature_extraction
sys.modules['feature'] = fake_feature_mod

# Provide a minimal fake utils.Utils module to avoid importing torchvision and other deps
fake_utils = types.ModuleType('utils.Utils')
class FakeUtils:
    @staticmethod
    def get_device():
        return 'cpu'

    @staticmethod
    def find_threshold(labels, distances, param, drop=False):
        # Return dummy tuple (unused, threshold)
        try:
            th = float(np.median(distances)) if len(distances) > 0 else 0.0
        except Exception:
            th = 0.0
        return None, th

def distances_from_average_clusters(*args, **kwargs):
    return None

fake_utils.Utils = FakeUtils
fake_utils.distances_from_average_clusters = distances_from_average_clusters
sys.modules['utils.Utils'] = fake_utils
sys.modules['utils'] = types.ModuleType('utils')

from config import ExperimentConfig
# Provide a fake 'train' module to avoid importing real training dependencies
fake_train = types.ModuleType('train')
def fake_training_population(config):
    print(f"[smoke] fake training_population called for resnet_model={getattr(config, 'resnet_model', None)} expansion={config.expansion} seed={config.seed}")
    return
fake_train.training_population = fake_training_population
sys.modules['train'] = fake_train

# Provide fake model modules to avoid importing torch.nn etc.
fake_models_spikeresnet = types.ModuleType('models.spikeresnet')
class DummyModel:
    def __init__(self, *args, **kwargs):
        pass

for name in ['spikeConvNN1', 'spikeConvNN2', 'spikeConvNN4', 'SpikeResNet9Model', 'SpikeResNet10Model', 'SpikeResNet18Model', 'SpikeResNet20Model']:
    setattr(fake_models_spikeresnet, name, DummyModel)

sys.modules['models.spikeresnet'] = fake_models_spikeresnet

fake_models_plain = types.ModuleType('models.plain')
setattr(fake_models_plain, 'spikeLinearNet1', DummyModel)
sys.modules['models.plain'] = fake_models_plain

import metrics.Metrics as metrics_mod

# Replace heavy feature extraction with a no-op that just logs
def fake_feature_extraction(config):
    print(f"[smoke] fake feature_extraction_spike called for dataset_feat={getattr(config, 'dataset_feat', None)} expansion={config.expansion} seed={config.seed}")
    # do nothing; pretend features exist
    return

# Replace test_metrics with a lightweight fake that returns plausible stats
def fake_test_metrics(config, case, nameID, methods, features='spikes'):
    namesOOD = [d for d in getattr(config, 'dataset_feat', []) if d != config.dataset_ID]
    num_methods = len(methods)
    # return shape (num_ood, num_methods*3), fill AUROC columns with 0.85
    stats = np.zeros((len(namesOOD), num_methods * 3), dtype=np.float64)
    if len(namesOOD) > 0 and num_methods > 0:
        stats[:, :num_methods] = 0.85
        stats[:, num_methods:2*num_methods] = 0.6  # aupr
        stats[:, 2*num_methods:3*num_methods] = 0.1  # fpr95
    print(f"[smoke] fake_test_metrics returning stats for namesOOD={namesOOD}, features={features}")
    return stats

# Apply monkeypatches
metrics_mod.feature_extraction_spike = fake_feature_extraction
metrics_mod.test_metrics = fake_test_metrics

# Build a minimal config
cfg = ExperimentConfig()
cfg.dataset_ID = 'CIFAR10'
# keep dataset_feat consistent with default config
cfg.dataset_feat = ['CIFAR10', 'CIFAR100', 'tImage200']
# define near and far as subsets
cfg.near_ood = ['CIFAR100']
cfg.far_ood = ['tImage200']
cfg.case = 'smoke'
cfg.methods = ['KNN']

seeds = [0]
expansions = [1, 5]
resnet_models = [4]

# Ensure results dir is present
os.makedirs(os.path.join('results', 'ex_1'), exist_ok=True)

print('[smoke] Starting statistics_exp_1 smoke run')
metrics_mod.statistics_exp_1(cfg, seeds, expansions, resnet_models)
print('[smoke] Completed')
