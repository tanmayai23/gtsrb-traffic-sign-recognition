"""Invariants of the preprocessing and augmentation pipeline."""

import unittest

import numpy as np

from src.transforms import (apply_clahe, augment, augment_affine,
                            augment_photometric, crop_roi, denormalise, resize,
                            to_tensor)


def synthetic_sign(h: int = 60, w: int = 55) -> np.ndarray:
    """A red square on grey, standing in for a sign crop."""
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    img[10:-10, 10:-10] = (200, 30, 30)
    return img


class TestCropROI(unittest.TestCase):
    def test_crop_reduces_size(self):
        img = synthetic_sign(60, 55)
        out = crop_roi(img, 10, 10, 45, 50, margin=0.0)
        self.assertEqual(out.shape[:2], (40, 35))

    def test_margin_expands_the_box(self):
        img = synthetic_sign(60, 55)
        tight = crop_roi(img, 15, 15, 40, 45, margin=0.0)
        loose = crop_roi(img, 15, 15, 40, 45, margin=0.2)
        self.assertGreater(loose.shape[0], tight.shape[0])
        self.assertGreater(loose.shape[1], tight.shape[1])

    def test_clips_to_image_bounds(self):
        img = synthetic_sign(60, 55)
        out = crop_roi(img, 0, 0, 55, 60, margin=0.5)
        self.assertLessEqual(out.shape[0], 60)
        self.assertLessEqual(out.shape[1], 55)

    def test_degenerate_roi_falls_back_to_full_image(self):
        img = synthetic_sign()
        out = crop_roi(img, 30, 30, 30, 30, margin=0.0)
        self.assertEqual(out.shape, img.shape)


class TestCLAHE(unittest.TestCase):
    def test_preserves_shape_and_dtype(self):
        img = synthetic_sign()
        out = apply_clahe(img)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_increases_contrast_on_a_flat_image(self):
        img = np.random.default_rng(0).integers(100, 140, (40, 40, 3), dtype=np.uint8)
        self.assertGreater(apply_clahe(img).std(), img.std())

    def test_keeps_red_dominant(self):
        """CLAHE runs on luminance only, so hue must survive it.

        Equalising each RGB channel separately would shift colour, and colour is
        class-discriminative for traffic signs.
        """
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[:, :, 0] = 200
        img[:, :, 1] = 40
        img[:, :, 2] = 40
        out = apply_clahe(img)
        self.assertGreater(int(out[:, :, 0].mean()), int(out[:, :, 1].mean()) + 50)


class TestResize(unittest.TestCase):
    def test_output_is_square_and_uint8(self):
        for size in (32, 48):
            out = resize(synthetic_sign(70, 50), size)
            self.assertEqual(out.shape, (size, size, 3))
            self.assertEqual(out.dtype, np.uint8)

    def test_upscales_small_inputs(self):
        self.assertEqual(resize(synthetic_sign(15, 15), 32).shape, (32, 32, 3))


class TestAugmentation(unittest.TestCase):
    def setUp(self):
        self.img = resize(synthetic_sign(), 32)
        self.rng = np.random.default_rng(42)

    def test_preserves_shape_and_dtype(self):
        out = augment(self.img, self.rng)
        self.assertEqual(out.shape, self.img.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_actually_changes_the_image(self):
        self.assertFalse(np.array_equal(augment(self.img, self.rng), self.img))

    def test_is_deterministic_for_a_given_seed(self):
        a = augment(self.img, np.random.default_rng(7))
        b = augment(self.img, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_never_mirrors_the_image(self):
        """Regression guard for the most damaging possible augmentation bug.

        Mirroring maps class 33 (turn right ahead) onto class 34 (turn left
        ahead), 19 onto 20, and scrambles every speed-limit digit. An
        asymmetric pattern must never come back closer to its own mirror than
        to itself.
        """
        asym = np.zeros((32, 32, 3), dtype=np.uint8)
        asym[:, :10] = 255          # bright stripe on the left only
        mirrored = asym[:, ::-1]

        for seed in range(30):
            out = augment_affine(asym, np.random.default_rng(seed))
            d_self = np.abs(out.astype(int) - asym.astype(int)).mean()
            d_mirror = np.abs(out.astype(int) - mirrored.astype(int)).mean()
            self.assertLess(d_self, d_mirror,
                            f"seed {seed} produced a mirrored image")

    def test_photometric_stays_in_range(self):
        for seed in range(10):
            out = augment_photometric(self.img, np.random.default_rng(seed))
            self.assertGreaterEqual(int(out.min()), 0)
            self.assertLessEqual(int(out.max()), 255)

    def test_affine_leaves_no_black_border(self):
        """BORDER_REPLICATE, not a zero fill: black corners would be a cue the
        model could use to detect augmented samples."""
        bright = np.full((32, 32, 3), 200, dtype=np.uint8)
        for seed in range(10):
            out = augment_affine(bright, np.random.default_rng(seed))
            self.assertGreater(int(out.min()), 100, "black border introduced")


class TestTensorConversion(unittest.TestCase):
    def setUp(self):
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.25, 0.25, 0.25]

    def test_shape_and_dtype(self):
        t = to_tensor(resize(synthetic_sign(), 32), self.mean, self.std)
        self.assertEqual(tuple(t.shape), (3, 32, 32))
        self.assertEqual(str(t.dtype), "torch.float32")

    def test_normalisation_round_trips(self):
        img = resize(synthetic_sign(), 32)
        back = denormalise(to_tensor(img, self.mean, self.std), self.mean, self.std)
        np.testing.assert_allclose(back.astype(int), img.astype(int), atol=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
