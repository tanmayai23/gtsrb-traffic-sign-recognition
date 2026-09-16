"""Track-aware splitting, tested on synthetic data.

These tests never touch the real dataset, so they run on a fresh clone. The
project's central correctness claim -- that no physical sign appears in both
train and validation -- is therefore verifiable without a 365 MB download.

`test_split_integrity.py` checks the same properties on the real manifests once
`prepare` has been run.
"""

import unittest

import pandas as pd

from src.config import NUM_CLASSES
from src.prepare import track_aware_split


def synthetic_annotations(n_classes: int = NUM_CLASSES, tracks_per_class: int = 10,
                          frames_per_track: int = 30) -> pd.DataFrame:
    """Mimic GTSRB's structure: every class holds tracks, every track holds frames.

    Track numbers deliberately restart at 0 in each class, exactly as they do in
    the real archive.
    """
    rows = []
    for class_id in range(n_classes):
        for track_num in range(tracks_per_class):
            for frame in range(frames_per_track):
                rows.append({
                    "path": f"/data/{class_id:05d}/{track_num:05d}_{frame:05d}.ppm",
                    "filename": f"{track_num:05d}_{frame:05d}.ppm",
                    "class_id": class_id,
                    "track_num": track_num,
                    "frame_num": frame,
                    # The composite key: bare track_num is ambiguous across classes.
                    "track_id": f"{class_id}_{track_num}",
                })
    return pd.DataFrame(rows)


class TestTrackAwareSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_annotations()

    def test_no_track_appears_in_both_splits(self):
        train, val, _ = track_aware_split(self.df, val_frac=0.2, seed=42)
        self.assertEqual(set(train["track_id"]) & set(val["track_id"]), set())

    def test_every_frame_of_a_track_stays_together(self):
        """The property that makes the split meaningful.

        A track is ~30 near-identical frames of one physical sign. If even one
        frame crosses over, the model is validated on something it has seen.
        """
        train, val, _ = track_aware_split(self.df, val_frac=0.2, seed=42)
        for split in (train, val):
            sizes = split.groupby("track_id").size()
            self.assertTrue((sizes == 30).all(),
                            "a track was split across the boundary")

    def test_no_image_is_duplicated_or_lost(self):
        train, val, _ = track_aware_split(self.df, val_frac=0.2, seed=42)
        self.assertEqual(len(train) + len(val), len(self.df))
        self.assertEqual(set(train["path"]) | set(val["path"]), set(self.df["path"]))
        self.assertEqual(set(train["path"]) & set(val["path"]), set())

    def test_every_class_appears_on_both_sides(self):
        train, val, _ = track_aware_split(self.df, val_frac=0.2, seed=42)
        for name, split in (("train", train), ("val", val)):
            self.assertEqual(set(split["class_id"]), set(range(NUM_CLASSES)),
                             f"{name} does not cover every class")

    def test_validation_fraction_is_respected(self):
        for frac in (0.1, 0.2, 0.3):
            train, val, _ = track_aware_split(self.df, val_frac=frac, seed=42)
            actual = len(val) / (len(train) + len(val))
            self.assertAlmostEqual(actual, frac, delta=0.05)

    def test_split_is_deterministic(self):
        a, _, _ = track_aware_split(self.df, val_frac=0.2, seed=42)
        b, _, _ = track_aware_split(self.df, val_frac=0.2, seed=42)
        self.assertEqual(set(a["path"]), set(b["path"]))

    def test_different_seeds_give_different_splits(self):
        a, _, _ = track_aware_split(self.df, val_frac=0.2, seed=1)
        b, _, _ = track_aware_split(self.df, val_frac=0.2, seed=2)
        self.assertNotEqual(set(a["path"]), set(b["path"]))

    def test_classes_with_few_tracks_keep_a_training_side(self):
        """Rare classes must not lose every track to validation.

        GTSRB's smallest classes hold only a handful of tracks, and an
        aggressive val_frac could otherwise claim all of them.
        """
        sparse = synthetic_annotations(tracks_per_class=2)
        train, val, _ = track_aware_split(sparse, val_frac=0.5, seed=42)
        for class_id in range(NUM_CLASSES):
            self.assertGreater(len(train[train["class_id"] == class_id]), 0,
                               f"class {class_id} has no training data")
            self.assertGreater(len(val[val["class_id"] == class_id]), 0,
                               f"class {class_id} has no validation data")

    def test_leaky_random_split_is_what_this_avoids(self):
        """Demonstrates the bug being prevented.

        A random per-image split scatters frames of the same sign across both
        sides. Asserting that it leaks documents precisely what the track-aware
        split is for.
        """
        shuffled = self.df.sample(frac=1.0, random_state=0).reset_index(drop=True)
        cut = int(0.8 * len(shuffled))
        naive_train, naive_val = shuffled[:cut], shuffled[cut:]

        leaked = set(naive_train["track_id"]) & set(naive_val["track_id"])
        self.assertGreater(
            len(leaked), 0,
            "a random split should leak tracks -- that is the motivation here",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
