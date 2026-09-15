"""Constants and the run configuration object."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

NUM_CLASSES = 43

# Official GTSRB class order (ClassId 0-42).
CLASS_NAMES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5t",
    "Right-of-way at next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5t prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5t",
]

assert len(CLASS_NAMES) == NUM_CLASSES

# Fallback only. prepare computes real statistics over the training split and
# writes them to manifests/norm_stats.json.
DEFAULT_NORM = {
    "mean": [0.3403, 0.3121, 0.3214],
    "std": [0.2724, 0.2608, 0.2669],
}


@dataclass
class Config:
    """Everything that affects a training run.

    Stored inside the checkpoint so that predict/gradcam can rebuild an
    identical model and preprocessing pipeline with no flags.
    """

    img_size: int = 32
    batch_size: int = 128
    epochs: int = 15
    lr: float = 1e-3
    max_lr: float = 3e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    dropout: float = 0.3
    width_mult: float = 1.0
    balance: str = "sampler"  # sampler | loss | none
    use_clahe: bool = True
    seed: int = 42
    workers: int = 0
    patience: int = 6
    max_minutes: float = 30.0
    grad_clip: float = 5.0
    channels: tuple = (32, 64, 128)
    norm_mean: list = field(default_factory=lambda: list(DEFAULT_NORM["mean"]))
    norm_std: list = field(default_factory=lambda: list(DEFAULT_NORM["std"]))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channels"] = list(self.channels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "channels" in known:
            known["channels"] = tuple(known["channels"])
        return cls(**known)
