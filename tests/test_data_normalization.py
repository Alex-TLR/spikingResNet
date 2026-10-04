"""Unit tests for fit-controlled CIFAR input normalization."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config import ExperimentConfig
from utils.Utils import Utils


class CifarNormalizationTest(unittest.TestCase):
    """Verify that normalization follows the configured fitting method."""

    def make_config(self, loss, fit, ce_source=None):
        """Create a minimal configuration for pipeline selection."""
        config = ExperimentConfig()
        config.loss = loss
        config.fit = fit
        config.ce_source = ce_source
        return config

    def load_cifar_transforms(self, config, auto_aug=True):
        """Capture train and test transforms without downloading CIFAR-10."""
        datasets = []

        def fake_cifar10(**kwargs):
            dataset = SimpleNamespace(**kwargs)
            datasets.append(dataset)
            return dataset

        with patch("utils.Utils.CIFAR10", side_effect=fake_cifar10):
            Utils.load_data("CIFAR10", config, auto_aug=auto_aug)
        return datasets[0].transform, datasets[1].transform

    def test_membrane_fit_returns_standard_train_and_test_transforms(self):
        """Membrane fitting always normalizes with CIFAR statistics."""
        for source in ("logits", "membranes"):
            with self.subTest(source=source):
                config = self.make_config("cross_entropy", "membrane", source)
                train_transform, test_transform = self.load_cifar_transforms(config)

                self.assertEqual(
                    train_transform.transforms[-1].mean,
                    (0.4914, 0.4822, 0.4465),
                )
                self.assertEqual(
                    test_transform.transforms[-1].mean,
                    (0.4914, 0.4822, 0.4465),
                )
                self.assertEqual(
                    train_transform.transforms[-1].std,
                    (0.2023, 0.1994, 0.201),
                )
                self.assertEqual(
                    test_transform.transforms[-1].std,
                    (0.2023, 0.1994, 0.201),
                )

    def test_spike_fit_returns_identity_train_and_test_transforms(self):
        """Spike fitting always preserves raw tensor intensity scaling."""
        for loss, source in (("cross_entropy", "spikes"), ("mse_count_loss", None)):
            with self.subTest(loss=loss):
                config = self.make_config(loss, "spike", source)
                train_transform, test_transform = self.load_cifar_transforms(config)

                self.assertEqual(train_transform.transforms[-1].mean, (0, 0, 0))
                self.assertEqual(test_transform.transforms[-1].mean, (0, 0, 0))
                self.assertEqual(train_transform.transforms[-1].std, (1, 1, 1))
                self.assertEqual(test_transform.transforms[-1].std, (1, 1, 1))


if __name__ == "__main__":
    unittest.main()