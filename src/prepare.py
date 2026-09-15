"""Build train/val/test manifests from the extracted GTSRB archives.

The interesting part is the split. See ``track_aware_split``.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASS_NAMES, NUM_CLASSES
from .download import ensure_dataset
from .utils import ensure_dir, save_json

GT_COLUMNS = {
    "Filename": "filename",
    "Width": "width",
    "Height": "height",
    "Roi.X1": "x1",
    "Roi.Y1": "y1",
    "Roi.X2": "x2",
    "Roi.Y2": "y2",
    "ClassId": "class_id",
}


def parse_training_annotations(extract_root: Path) -> pd.DataFrame:
    """Read the 43 per-class GT CSVs into one frame with track IDs attached."""
    images_root = Path(extract_root) / "GTSRB" / "Final_Training" / "Images"
    if not images_root.exists():
        raise FileNotFoundError(f"missing {images_root}; run `prepare` first")

    frames = []
    for class_id in range(NUM_CLASSES):
        class_dir = images_root / f"{class_id:05d}"
        csv_path = class_dir / f"GT-{class_id:05d}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"missing annotation file {csv_path}")

        df = pd.read_csv(csv_path, sep=";").rename(columns=GT_COLUMNS)
        df["class_id"] = class_id
        df["path"] = df["filename"].map(lambda f: str(class_dir / f))
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # Filenames look like "00053_00024.ppm" -> track 53, frame 24. A track is
    # ~30 consecutive video frames of ONE physical sign.
    parts = df["filename"].str.replace(".ppm", "", regex=False).str.split("_", expand=True)
    df["track_num"] = parts[0].astype(int)
    df["frame_num"] = parts[1].astype(int)

    # Track numbers restart at 0 within every class, so "00000" exists in class 0
    # and class 1 and they are unrelated. The identifier must be the composite.
    df["track_id"] = df["class_id"].astype(str) + "_" + df["track_num"].astype(str)

    return df[
        ["path", "filename", "class_id", "track_id", "track_num", "frame_num",
         "width", "height", "x1", "y1", "x2", "y2"]
    ]


def parse_test_annotations(extract_root: Path) -> pd.DataFrame:
    """Read GT-final_test.csv and attach absolute image paths."""
    root = Path(extract_root) / "GTSRB"
    images_root = root / "Final_Test" / "Images"

    candidates = [
        root / "GT-final_test.csv",
        images_root / "GT-final_test.csv",
        Path(extract_root) / "GT-final_test.csv",
    ]
    csv_path = next((c for c in candidates if c.exists()), None)
    if csv_path is None:
        found = list(Path(extract_root).rglob("GT-final_test.csv"))
        if not found:
            raise FileNotFoundError("GT-final_test.csv not found in the extracted data")
        csv_path = found[0]

    df = pd.read_csv(csv_path, sep=";").rename(columns=GT_COLUMNS)
    df["path"] = df["filename"].map(lambda f: str(images_root / f))

    missing = [p for p in df["path"].head(50) if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"test images not where expected, e.g. {missing[0]}")

    return df[["path", "filename", "class_id", "width", "height", "x1", "y1", "x2", "y2"]]


def track_aware_split(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """Split train/val by TRACK, stratified by class.

    Why this matters: each track is ~30 near-identical frames of the same sign.
    Splitting on individual images puts frame 11 in train and frame 12 in val,
    so the model is validated on pictures it has effectively memorised. That
    reports ~99.9% validation accuracy and means nothing.

    Splitting whole tracks keeps every physical sign entirely on one side. The
    resulting number is lower and actually reflects generalisation.
    """
    train_parts, val_parts, summary = [], [], []

    for class_id, group in df.groupby("class_id", sort=True):
        tracks = sorted(group["track_id"].unique())
        # Per-class seed offset: deterministic, and avoids every class picking
        # the same positional tracks.
        random.Random(seed + int(class_id)).shuffle(tracks)

        n_val = max(1, round(val_frac * len(tracks)))
        n_val = min(n_val, len(tracks) - 1)  # never leave the train side empty

        val_tracks = set(tracks[:n_val])
        in_val = group["track_id"].isin(val_tracks)

        val_parts.append(group[in_val])
        train_parts.append(group[~in_val])

        summary.append({
            "class_id": int(class_id),
            "class_name": CLASS_NAMES[int(class_id)],
            "n_tracks": len(tracks),
            "n_val_tracks": n_val,
            "n_train_images": int((~in_val).sum()),
            "n_val_images": int(in_val.sum()),
        })

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)

    overlap = set(train_df["track_id"]) & set(val_df["track_id"])
    if overlap:
        raise RuntimeError(
            f"track leakage: {len(overlap)} tracks in both splits, e.g. {list(overlap)[:5]}"
        )

    missing_train = set(range(NUM_CLASSES)) - set(train_df["class_id"])
    missing_val = set(range(NUM_CLASSES)) - set(val_df["class_id"])
    if missing_train or missing_val:
        raise RuntimeError(
            f"classes absent after split -- train: {sorted(missing_train)}, val: {sorted(missing_val)}"
        )

    return train_df, val_df, summary


def compute_norm_stats(train_df: pd.DataFrame, img_size: int, use_clahe: bool,
                       sample_size: int = 8000, seed: int = 42) -> dict:
    """Channel mean/std over the TRAINING split only.

    Including val here would leak its statistics into the model's input scaling.
    A random sample is plenty -- these converge quickly and it saves ~30s.
    """
    from .transforms import load_image, preprocess_deterministic

    rng = np.random.default_rng(seed)
    n = min(sample_size, len(train_df))
    idx = rng.choice(len(train_df), size=n, replace=False)
    rows = train_df.iloc[idx]

    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0

    for row in rows.itertuples():
        img = load_image(row.path)
        img = preprocess_deterministic(
            img, (row.x1, row.y1, row.x2, row.y2), img_size, use_clahe
        )
        x = img.astype(np.float64) / 255.0
        total += x.sum(axis=(0, 1))
        total_sq += (x**2).sum(axis=(0, 1))
        count += x.shape[0] * x.shape[1]

    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 1e-12))
    return {"mean": mean.tolist(), "std": std.tolist(), "n_images": int(n)}


def run_prepare(data_dir: Path, val_frac: float = 0.2, seed: int = 42,
                img_size: int = 32, use_clahe: bool = True,
                force_download: bool = False) -> dict:
    data_dir = Path(data_dir)
    extract_root = ensure_dataset(data_dir, force=force_download)

    manifest_dir = ensure_dir(data_dir / "manifests")

    print("[prepare] parsing annotations")
    train_all = parse_training_annotations(extract_root)
    test_df = parse_test_annotations(extract_root)
    print(f"  training images: {len(train_all)}  tracks: {train_all['track_id'].nunique()}")
    print(f"  test images:     {len(test_df)}")

    print(f"[prepare] track-aware split (val_frac={val_frac}, seed={seed})")
    train_df, val_df, summary = track_aware_split(train_all, val_frac, seed)

    train_df.to_csv(manifest_dir / "train.csv", index=False)
    val_df.to_csv(manifest_dir / "val.csv", index=False)
    test_df.to_csv(manifest_dir / "test.csv", index=False)
    save_json(CLASS_NAMES, manifest_dir / "class_names.json")

    print("[prepare] computing normalisation statistics over the train split")
    norm = compute_norm_stats(train_df, img_size, use_clahe, seed=seed)
    save_json(norm, manifest_dir / "norm_stats.json")
    print(f"  mean={[round(v, 4) for v in norm['mean']]} std={[round(v, 4) for v in norm['std']]}")

    counts = train_df["class_id"].value_counts()
    split_summary = {
        "seed": seed,
        "val_frac": val_frac,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_train_tracks": int(train_df["track_id"].nunique()),
        "n_val_tracks": int(val_df["track_id"].nunique()),
        "track_overlap": 0,
        "imbalance_ratio": round(counts.max() / counts.min(), 2),
        "min_class_count": int(counts.min()),
        "max_class_count": int(counts.max()),
        "per_class": summary,
    }
    save_json(split_summary, manifest_dir / "split_summary.json")

    print()
    print(f"  train: {len(train_df):>6} images / {split_summary['n_train_tracks']:>5} tracks")
    print(f"  val:   {len(val_df):>6} images / {split_summary['n_val_tracks']:>5} tracks")
    print(f"  test:  {len(test_df):>6} images")
    print(f"  class imbalance: {split_summary['imbalance_ratio']}x "
          f"(min {split_summary['min_class_count']}, max {split_summary['max_class_count']})")
    print(f"  TRACK OVERLAP: {len(set(train_df['track_id']) & set(val_df['track_id']))}")
    print(f"\n[prepare] manifests written to {manifest_dir}")

    return split_summary
