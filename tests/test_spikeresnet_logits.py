"""Unit tests for explicit classifier-logit traces from spiking models."""

import unittest

import torch

from models.spikeresnet import SpikeResNet9Model, spike_resnet10


class SpikeResNetLogitsTest(unittest.TestCase):
    """Verify logits semantics, gradients, shapes, and API compatibility."""

    def test_legacy_logits_match_classifier_hook_and_receive_gradients(self):
        """Returned legacy logits match the classifier and retain gradients."""
        model = SpikeResNet9Model(1, 3, 0.95, 0.25)
        inputs = torch.randn(2, 1, 28, 28)
        captured = []
        handle = model.fc7.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)
        )

        outputs = model(inputs, 2, return_logits_trace=True)
        handle.remove()

        self.assertEqual(len(outputs), 5)
        self.assertTrue(torch.equal(outputs[4], torch.stack(captured)))
        outputs[4].mean().backward()
        self.assertIsNotNone(model.fc7.weight.grad)

    def test_generic_model_preserves_default_contract(self):
        """Generic models retain four outputs unless logits are requested."""
        model = spike_resnet10(3, 3, expansion=2)
        inputs = torch.randn(2, 3, 32, 32)

        default_outputs = model(inputs, 2)
        extended_outputs = model(inputs, 2, return_logits_trace=True)

        self.assertEqual(len(default_outputs), 4)
        self.assertEqual(len(extended_outputs), 5)
        self.assertEqual(extended_outputs[4].shape, (2, 2, 6))


if __name__ == "__main__":
    unittest.main()
