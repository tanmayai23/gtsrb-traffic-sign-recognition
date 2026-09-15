"""Seeding, timing, logging and checkpoint/JSON helpers."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed every RNG this project touches."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # warn_only: a few CPU kernels have no deterministic variant, and we
        # would rather run than abort.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_threads(max_threads: int = 8) -> int:
    """Cap intra-op threads.

    On a 12-core box this model is small enough that using every core costs
    more in synchronisation than it gains in parallelism; 8 measured fastest.
    """
    n = min(max_threads, os.cpu_count() or 1)
    torch.set_num_threads(n)
    return n


def get_logger(name: str = "gtsrb", logfile: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)

    return logger


class Timer:
    """Context manager returning elapsed seconds via ``.seconds``."""

    def __init__(self, label: str = "", verbose: bool = False):
        self.label, self.verbose, self.seconds = label, verbose, 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0
        if self.verbose:
            print(f"{self.label}: {human_time(self.seconds)}")


def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    return path


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_checkpoint(path: Path, **payload) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train one first:\n"
            f"  python -m src.cli train"
        )
    # weights_only=False: our checkpoints intentionally carry the config dict and
    # class names, not just tensors. These files are produced by this project.
    return torch.load(path, map_location=map_location, weights_only=False)
