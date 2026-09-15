"""Architecture and Grad-CAM mechanics."""

import unittest

import torch

from src.config import NUM_CLASSES, Config
from src.gradcam import GradCAM
from src.model import TrafficSignNet, build_model, count_parameters


class TestArchitecture(unittest.TestCase):
    def setUp(self):
        self.model = TrafficSignNet()

    def test_output_shape(self):
        out = self.model(torch.randn(4, 3, 32, 32))
        self.assertEqual(tuple(out.shape), (4, NUM_CLASSES))

    def test_parameter_budget(self):
        """Kept near 99k: the size that trains in ~6 minutes on a CPU."""
        n = count_parameters(self.model)
        self.assertGreater(n, 50_000)
        self.assertLess(n, 150_000)

    def test_accepts_other_input_sizes(self):
        """Global average pooling makes the head independent of input size."""
        for size in (32, 48, 64):
            out = self.model(torch.randn(2, 3, size, size))
            self.assertEqual(tuple(out.shape), (2, NUM_CLASSES))

    def test_width_multiplier_changes_capacity(self):
        small = count_parameters(TrafficSignNet(width_mult=0.5))
        base = count_parameters(TrafficSignNet(width_mult=1.0))
        large = count_parameters(TrafficSignNet(width_mult=2.0))
        self.assertLess(small, base)
        self.assertLess(base, large)

    def test_conv_layers_have_no_bias(self):
        """BatchNorm follows every conv, so a conv bias would be redundant."""
        for block in self.model.features:
            self.assertIsNone(block.conv.bias)

    def test_last_conv_layer_accessor(self):
        layer = self.model.get_last_conv_layer()
        self.assertIsInstance(layer, torch.nn.Conv2d)
        self.assertEqual(layer.out_channels, 128)

    def test_gradients_reach_the_first_layer(self):
        out = self.model(torch.randn(2, 3, 32, 32))
        out.sum().backward()
        first = self.model.features[0].conv.weight
        self.assertIsNotNone(first.grad)
        self.assertGreater(float(first.grad.abs().sum()), 0.0)

    def test_build_model_honours_config(self):
        model = build_model(Config(width_mult=0.5, dropout=0.1))
        self.assertLess(count_parameters(model), count_parameters(TrafficSignNet()))


class TestGradCAM(unittest.TestCase):
    def setUp(self):
        self.model = TrafficSignNet()
        self.x = torch.randn(1, 3, 32, 32)

    def test_produces_a_normalised_map(self):
        with GradCAM(self.model, self.model.get_last_conv_layer()) as gc:
            cam, idx, prob = gc(self.x)
        # The hook sits on the conv output, which precedes that block's pool:
        # 32 -> 16 -> 8, so the map is 8x8 rather than the post-pool 4x4.
        self.assertEqual(cam.shape, (8, 8))
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertLessEqual(float(cam.max()), 1.0 + 1e-6)
        self.assertTrue(0 <= idx < NUM_CLASSES)
        self.assertTrue(0.0 <= prob <= 1.0)

    def test_explains_the_requested_class(self):
        with GradCAM(self.model, self.model.get_last_conv_layer()) as gc:
            _, idx, _ = gc(self.x, class_idx=7)
        self.assertEqual(idx, 7)

    def test_hooks_are_removed_on_exit(self):
        gc = GradCAM(self.model, self.model.get_last_conv_layer())
        gc(self.x)
        gc.remove()
        self.assertEqual(len(gc._handles), 0)

    def test_works_while_the_model_is_in_eval_mode(self):
        """Grad-CAM needs eval() for BatchNorm statistics but still requires
        gradients, so it must not run under no_grad/inference_mode."""
        self.model.eval()
        with GradCAM(self.model, self.model.get_last_conv_layer()) as gc:
            cam, _, _ = gc(self.x)
        self.assertEqual(cam.shape, (8, 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
