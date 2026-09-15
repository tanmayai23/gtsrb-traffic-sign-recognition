"""Dataset and dataloader construction.

torchvision is not installed (and not needed), so this is a plain Dataset over
the manifest CSVs written by ``prepare``.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from .config import NUM_CLASSES
from .transforms import augment, load_image, preprocess_deterministic, to_tensor


class GTSRBDataset(Dataset):
    """GTSRB images listed by a manifest CSV.

    Decoding a .ppm and running CLAHE costs about 1 ms per image. Across 31k
    training images that is ~30 s of work *per epoch* if done lazily -- more
    than the epoch's actual compute. Since those steps are deterministic, they
    run once up front and the results live in RAM as uint8.

    31k x 32 x 32 x 3 bytes is about 96 MB, which is a good trade.
    """

    def __init__(self, manifest: Path | pd.DataFrame, img_size: int = 32,
                 train: bool = False, use_clahe: bool = True,
                 norm_mean=None, norm_std=None, limit: int | None = None,
                 seed: int = 42, precache: bool = True, quiet: bool = False):
        self.df = (pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame)
                   else manifest.reset_index(drop=True))

        if limit is not None and limit < len(self.df):
            # Stratified subsample so --quick still sees every class.
            frac = limit / len(self.df)
            self.df = (
                self.df.groupby("class_id", group_keys=False)[self.df.columns.tolist()]
                .apply(lambda g: g.sample(n=max(1, int(round(frac * len(g)))),
                                          random_state=seed))
                .reset_index(drop=True)
            )

        self.img_size = img_size
        self.train = train
        self.use_clahe = use_clahe
        self.mean = norm_mean if norm_mean is not None else [0.5, 0.5, 0.5]
        self.std = norm_std if norm_std is not None else [0.5, 0.5, 0.5]
        self.seed = seed

        self.labels = self.df["class_id"].to_numpy(dtype=np.int64)
        self.paths = self.df["path"].tolist()
        self.has_roi = all(c in self.df.columns for c in ("x1", "y1", "x2", "y2"))
        self.rois = (self.df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.int64)
                     if self.has_roi else None)

        self.cache = None
        if precache:
            self._build_cache(quiet)

    def _load_one(self, i: int) -> np.ndarray:
        roi = tuple(self.rois[i]) if self.rois is not None else None
        return preprocess_deterministic(
            load_image(self.paths[i]), roi, self.img_size, self.use_clahe
        )

    def _build_cache(self, quiet: bool = False) -> None:
        n = len(self.df)
        self.cache = np.empty((n, self.img_size, self.img_size, 3), dtype=np.uint8)

        split = "train" if self.train else "eval"
        bar = None if quiet else tqdm(
            total=n, desc=f"  caching {split} ({n})", ncols=80, leave=False
        )

        # Decoding and CLAHE spend most of their time in file I/O and OpenCV,
        # both of which release the GIL, so threads give a real speedup here.
        def work(i: int) -> None:
            self.cache[i] = self._load_one(i)

        workers = min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(work, range(n), chunksize=64):
                if bar is not None:
                    bar.update(1)

        if bar is not None:
            bar.close()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        img = self.cache[idx] if self.cache is not None else self._load_one(idx)

        if self.train:
            # Seeded per (epoch-independent) index so runs are reproducible even
            # when workers > 0, where a shared global RNG would not be.
            rng = np.random.default_rng((self.seed * 1_000_003 + idx) % (2**32))
            img = augment(img, rng)

        return to_tensor(img, self.mean, self.std), int(self.labels[idx])

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=NUM_CLASSES)


def make_sampler(dataset: GTSRBDataset, seed: int = 42) -> WeightedRandomSampler:
    """Sample inversely to class frequency.

    GTSRB is imbalanced about 10x (roughly 210 to 2250 images per class), which
    biases a plain shuffle toward the common speed-limit signs.
    """
    counts = dataset.class_counts()
    per_class_w = 1.0 / np.maximum(counts, 1)
    weights = per_class_w[dataset.labels]

    g = torch.Generator()
    g.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=g,
    )


def class_weights(dataset: GTSRBDataset) -> torch.Tensor:
    """Inverse-frequency weights normalised to mean 1, for weighted CE loss."""
    counts = dataset.class_counts().astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (NUM_CLASSES * counts)
    return torch.as_tensor(w, dtype=torch.float32)


def build_loader(dataset: GTSRBDataset, batch_size: int, shuffle: bool = False,
                 sampler=None, workers: int = 0, seed: int = 42) -> DataLoader:
    generator = None
    if shuffle and sampler is None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        # DataLoader rejects shuffle=True alongside a sampler.
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=workers,
        pin_memory=False,       # CPU-only training; pinning buys nothing
        drop_last=False,
        persistent_workers=workers > 0,
        generator=generator,
    )
