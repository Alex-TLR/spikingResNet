"""Unit tests for YAML loading and experiment configuration defaults."""

import tempfile
import unittest
from pathlib import Path

from config import ExperimentConfig, load_config_from_yaml


class ExperimentConfigTest(unittest.TestCase):
    """Verify default, explicit, legacy, and invalid configurations."""

    def write_config(self, directory, content):
        """Write YAML content to a temporary configuration file."""
        path = Path(directory) / "config.yaml"
        path.write_text(content, encoding="utf-8")
        return path

    def test_default_configuration(self):
        """The in-code configuration exposes the expected training defaults."""
        config = ExperimentConfig()
        self.assertEqual(config.dataset_ID, "CIFAR10")
        self.assertEqual(config.loss, "count_loss")
        self.assertEqual(config.fit, "spike")

    def test_legacy_spike_ce_defaults(self):
        """Legacy spike fitting resolves to spike-rate cross-entropy."""
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(
                directory,
                "training:\n  loss: cross_entropy\n  fit: spike\n",
            )
            config = load_config_from_yaml(path)

        self.assertEqual(config.ce_source, "spikes")
        self.assertEqual(config.ce_mode, "spike_rate")
        self.assertEqual(config.population_reduction, "sum")

    def test_legacy_membrane_ce_defaults(self):
        """Legacy membrane fitting resolves to temporal-mean cross-entropy."""
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(
                directory,
                "training:\n  loss: cross_entropy\n  fit: membrane\n",
            )
            config = load_config_from_yaml(path)

        self.assertEqual(config.ce_source, "membranes")
        self.assertEqual(config.ce_mode, "temporal_mean")
        self.assertEqual(config.population_reduction, "sum")

    def test_explicit_ce_options(self):
        """Explicit CE source, mode, and population reduction are preserved."""
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(
                directory,
                "training:\n"
                "  loss: cross_entropy\n"
                "  fit: membrane\n"
                "  ce_source: logits\n"
                "  ce_mode: final_timestep\n"
                "  population_reduction: mean\n",
            )
            config = load_config_from_yaml(path)

        self.assertEqual(config.ce_source, "logits")
        self.assertEqual(config.ce_mode, "final_timestep")
        self.assertEqual(config.population_reduction, "mean")

    def test_invalid_ce_combination_is_rejected(self):
        """A spike-only CE mode cannot be paired with classifier logits."""
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(
                directory,
                "training:\n"
                "  ce_source: logits\n"
                "  ce_mode: spike_count\n",
            )
            with self.assertRaisesRegex(ValueError, "not compatible"):
                load_config_from_yaml(path)

    def test_empty_configuration_is_rejected(self):
        """An empty YAML document raises a descriptive error."""
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_config(directory, "")
            with self.assertRaisesRegex(ValueError, "empty"):
                load_config_from_yaml(path)

    def test_missing_configuration_is_rejected(self):
        """A missing YAML path raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.yaml"
            with self.assertRaises(FileNotFoundError):
                load_config_from_yaml(path)


if __name__ == "__main__":
    unittest.main()