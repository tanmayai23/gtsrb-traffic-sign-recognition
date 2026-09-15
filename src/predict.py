"""Single-image inference."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import CLASS_NAMES, Config
from .model import build_model
from .transforms import load_image, preprocess_deterministic, to_tensor
from .utils import configure_threads, get_device, load_checkpoint


def load_for_inference(checkpoint: Path, device=None):
    """Rebuild the model and its preprocessing from a checkpoint alone.

    Checkpoints store the config, class names and normalisation statistics, so
    no flags need to match what training used.
    """
    device = device or get_device()
    ck = load_checkpoint(checkpoint, map_location=str(device))
    cfg = Config.from_dict(ck["config"])

    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    return model, cfg, ck.get("class_names", CLASS_NAMES), ck["norm_stats"], device


def predict_image(model, cfg, norm, device, image_path: Path, topk: int = 5):
    img = load_image(image_path)
    # No ROI available for an arbitrary user image; the whole frame is used.
    proc = preprocess_deterministic(img, None, cfg.img_size, cfg.use_clahe)
    tensor = to_tensor(proc, norm["mean"], norm["std"]).unsqueeze(0).to(device)

    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1)[0]

    k = min(topk, probs.numel())
    conf, idx = torch.topk(probs, k)
    return idx.cpu().numpy().tolist(), conf.cpu().numpy().tolist(), img, proc


def run_predict(checkpoint: Path, image_path: Path, topk: int = 5,
                as_json: bool = False) -> dict:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"no such image: {image_path}")

    configure_threads()
    model, cfg, class_names, norm, device = load_for_inference(checkpoint)
    ids, confs, _, _ = predict_image(model, cfg, norm, device, image_path, topk)

    result = {
        "image": str(image_path),
        "predicted_class_id": int(ids[0]),
        "predicted_class_name": class_names[ids[0]],
        "confidence": round(float(confs[0]), 6),
        "topk": [
            {"class_id": int(i), "class_name": class_names[i],
             "confidence": round(float(c), 6)}
            for i, c in zip(ids, confs)
        ],
    }

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nimage: {image_path.name}")
        print(f"prediction: [{ids[0]}] {class_names[ids[0]]}  ({confs[0]:.2%})\n")
        print(f"  top-{len(ids)}:")
        for rank, (i, c) in enumerate(zip(ids, confs), 1):
            bar = "#" * int(round(c * 30))
            print(f"    {rank}. [{i:>2}] {class_names[i][:38]:<38} {c:6.2%} {bar}")

    return result
