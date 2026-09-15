"""The central correctness check: no track may appear in both train and val.

GTSRB stores ~30 consecutive video frames of each physical sign. If those
frames are scattered across train and val, the model is validated on images it
has effectively already seen, and validation accuracy becomes meaningless
(typically ~99.9%). These tests fail if that regression is ever introduced.
"""

import unittest
from pathlib import Path

import pandas as pd

from src.config import NUM_CLASSES

MANIFEST_DIR = Path(__file__).resolve().parent.parent / "data" / "manifests"


def manifests_exist() -> bool:
    return all((MANIFEST_DIR / f"{n}.csv").exists() for n in ("train", "val", "test"))


@unittest.skipUnless(manifests_exist(), "run `python -m src.cli prepare` first")
class TestSplitIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(MANIFEST_DIR / "train.csv")
        cls.val = pd.read_csv(MANIFEST_DIR / "val.csv")
        cls.test = pd.read_csv(MANIFEST_DIR / "test.csv")

    def test_no_track_overlap(self):
        overlap = set(self.train["track_id"]) & set(self.val["track_id"])
        self.assertEqual(
            len(overlap), 0,
            f"{len(overlap)} tracks leak across the split, e.g. {list(overlap)[:5]}",
        )

    def test_no_image_overlap(self):
        overlap = set(self.train["path"]) & set(self.val["path"])
        self.assertEqual(len(overlap), 0, "the same image file appears in both splits")

    def test_all_classes_in_both_splits(self):
        for name, df in (("train", self.train), ("val", self.val)):
            missing = set(range(NUM_CLASSES)) - set(df["class_id"])
            self.assertEqual(missing, set(), f"{name} is missing classes {sorted(missing)}")

    def test_track_id_is_class_scoped(self):
        """A track id must identify exactly one class.

        Track numbers restart from 0 in every class directory, so a bare
        "00000" prefix is ambiguous. The identifier has to combine class and
        track, or unrelated tracks from different classes get merged.
        """
        combined = pd.concat([self.train, self.val])
        per_track_classes = combined.groupby("track_id")["class_id"].nunique()
        bad = per_track_classes[per_track_classes > 1]
        self.assertEqual(len(bad), 0,
                         f"track ids spanning multiple classes: {list(bad.index[:5])}")

    def test_split_totals(self):
        self.assertEqual(len(self.train) + len(self.val), 39209,
                         "train + val should cover the full GTSRB training set")
        self.assertEqual(len(self.test), 12630, "official test set is 12,630 images")

    def test_validation_is_a_reasonable_fraction(self):
        frac = len(self.val) / (len(self.train) + len(self.val))
        self.assertGreater(frac, 0.10)
        self.assertLess(frac, 0.35)

    def test_frames_of_a_track_stay_together(self):
        """Every image of a given track must land on the same side."""
        train_tracks = set(self.train["track_id"])
        for track_id, group in self.val.groupby("track_id"):
            self.assertNotIn(track_id, train_tracks,
                             f"track {track_id} is split across train and val")


if __name__ == "__main__":
    unittest.main(verbosity=2)
