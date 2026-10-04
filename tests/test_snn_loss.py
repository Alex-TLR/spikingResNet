"""Unit tests for temporal and population-coded SNN cross-entropy."""

import unittest

import torch
import torch.nn.functional as F

from utils.snn_loss import (
    aggregate_population,
    cross_entropy_scores,
    resolve_ce_options,
    temporal_cross_entropy,
    validate_ce_options,
)


class SnnCrossEntropyTest(unittest.TestCase):
    """Verify CE aggregation formulas, validation, and autograd behavior."""

    def setUp(self):
        """Create a deterministic two-step population-coded trace."""
        self.trace = torch.tensor(
            [
                [[1.0, 3.0, 2.0, 4.0], [4.0, 2.0, 3.0, 1.0]],
                [[2.0, 4.0, 1.0, 5.0], [3.0, 1.0, 4.0, 2.0]],
            ]
        )
        self.labels = torch.tensor([0, 1])

    def test_population_sum_and_mean(self):
        """Population groups support sum and mean reduction."""
        expected_sum = torch.tensor(
            [
                [[4.0, 6.0], [6.0, 4.0]],
                [[6.0, 6.0], [4.0, 6.0]],
            ]
        )
        self.assertTrue(torch.equal(aggregate_population(self.trace, 2, "sum"), expected_sum))
        self.assertTrue(
            torch.equal(aggregate_population(self.trace, 2, "mean"), expected_sum / 2)
        )

    def test_temporal_mean(self):
        """Temporal-mean CE matches a direct PyTorch calculation."""
        grouped = aggregate_population(self.trace, 2, "mean")
        expected = F.cross_entropy(grouped.mean(dim=0), self.labels)
        actual = temporal_cross_entropy(self.trace, self.labels, "temporal_mean", 2, "mean")
        self.assertTrue(torch.allclose(actual, expected))

    def test_per_timestep(self):
        """Per-timestep CE equals the mean of independent timestep losses."""
        grouped = aggregate_population(self.trace, 2, "sum")
        expected = torch.stack(
            [F.cross_entropy(step, self.labels) for step in grouped]
        ).mean()
        actual = temporal_cross_entropy(self.trace, self.labels, "per_timestep", 2, "sum")
        self.assertTrue(torch.allclose(actual, expected))

    def test_final_timestep(self):
        """Final-timestep CE uses only the final population trace."""
        grouped = aggregate_population(self.trace, 2, "mean")
        expected = F.cross_entropy(grouped[-1], self.labels)
        actual = temporal_cross_entropy(self.trace, self.labels, "final_timestep", 2, "mean")
        self.assertTrue(torch.allclose(actual, expected))

    def test_spike_rate_and_count(self):
        """Spike-rate and spike-count CE use temporal mean and sum."""
        grouped = aggregate_population(self.trace, 2, "sum")
        rate = temporal_cross_entropy(self.trace, self.labels, "spike_rate", 2, "sum")
        count = temporal_cross_entropy(self.trace, self.labels, "spike_count", 2, "sum")
        self.assertTrue(torch.allclose(rate, F.cross_entropy(grouped.mean(0), self.labels)))
        self.assertTrue(torch.allclose(count, F.cross_entropy(grouped.sum(0), self.labels)))

    def test_scores_follow_mode(self):
        """Accuracy scores follow the selected temporal mode."""
        grouped = aggregate_population(self.trace, 2, "mean")
        self.assertTrue(
            torch.equal(
                cross_entropy_scores(self.trace, "final_timestep", 2, "mean"),
                grouped[-1],
            )
        )

    def test_legacy_defaults(self):
        """Legacy fit names resolve to backward-compatible CE settings."""
        self.assertEqual(
            resolve_ce_options(None, None, None, "membrane"),
            ("membranes", "temporal_mean", "sum"),
        )
        self.assertEqual(
            resolve_ce_options(None, None, None, "spike"),
            ("spikes", "spike_rate", "sum"),
        )

    def test_invalid_source_mode_combination(self):
        """Incompatible CE source and mode combinations are rejected."""
        with self.assertRaisesRegex(ValueError, "not compatible"):
            validate_ce_options("logits", "spike_count", "mean")

    def test_invalid_options(self):
        """Unknown source, mode, and population reduction values fail."""
        with self.assertRaisesRegex(ValueError, "Unsupported CE source"):
            validate_ce_options("voltages", "temporal_mean", "mean")
        with self.assertRaisesRegex(ValueError, "Unsupported CE mode"):
            validate_ce_options("spikes", "maximum", "mean")
        with self.assertRaisesRegex(ValueError, "Unsupported population reduction"):
            validate_ce_options("spikes", "spike_rate", "maximum")

    def test_population_input_validation(self):
        """Population aggregation validates shape, classes, and divisibility."""
        with self.assertRaisesRegex(ValueError, "time-major trace"):
            aggregate_population(torch.zeros(2, 4), 2)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            aggregate_population(self.trace, 0)
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            aggregate_population(torch.zeros(2, 2, 5), 2)

    def test_population_identity_preserves_tensor(self):
        """A one-neuron-per-class trace is returned without copying."""
        trace = torch.randn(2, 3, 4, requires_grad=True)
        self.assertIs(aggregate_population(trace, 4), trace)

    def test_temporal_cross_entropy_propagates_gradients(self):
        """Temporal CE produces finite gradients for the source trace."""
        trace = self.trace.clone().requires_grad_()
        loss = temporal_cross_entropy(trace, self.labels, "per_timestep", 2, "mean")
        loss.backward()
        self.assertIsNotNone(trace.grad)
        self.assertTrue(torch.isfinite(trace.grad).all())


if __name__ == "__main__":
    unittest.main()
