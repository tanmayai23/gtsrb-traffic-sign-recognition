"""Test-set evaluation: metrics, confusion matrix, error analysis."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from .config import CLASS_NAMES, NUM_CLASSES, Config
from .dataset import GTSRBDataset, build_loader
from .engine import validate
from .model import build_model, count_parameters
from .plots import (plot_confusion_matrix, plot_misclassified_grid,
                    plot_per_class_f1)
from .utils import (configure_threads, ensure_dir, get_device, human_time,
                    load_checkpoint, save_json, set_seed)


def _top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([labels[i] in topk[i] for i in range(len(labels))]))


def run_evaluation(checkpoint: Path, data_dir: Path, results_dir: Path,
                   split: str = "test", batch_size: int = 256,
                   workers: int = 0) -> dict:
    manifest = Path(data_dir) / "manifests" / f"{split}.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest}. Run:\n  python -m src.cli prepare"
        )

    results_dir = ensure_dir(results_dir)
    ck = load_checkpoint(checkpoint)
    cfg = Config.from_dict(ck["config"])
    class_names = ck.get("class_names", CLASS_NAMES)
    norm = ck["norm_stats"]

    set_seed(cfg.seed)
    threads = configure_threads()
    device = get_device()

    print(f"[evaluate] checkpoint {checkpoint}")
    print(f"[evaluate] trained {ck.get('epochs', ck.get('epoch'))} epochs, "
          f"val acc {ck.get('val_acc', float('nan')):.4f}")
    print(f"[evaluate] split={split} device={device} threads={threads}")

    ds = GTSRBDataset(manifest, img_size=cfg.img_size, train=False,
                      use_clahe=cfg.use_clahe, norm_mean=norm["mean"],
                      norm_std=norm["std"], seed=cfg.seed)
    loader = build_loader(ds, batch_size, workers=workers)
    print(f"[evaluate] {len(ds)} images")

    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model_state"])

    t0 = time.perf_counter()
    _, acc, y_true, y_pred, y_prob = validate(
        model, loader, nn.CrossEntropyLoss(), device, collect=True
    )
    infer_seconds = time.perf_counter() - t0

    balanced = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    top5 = _top_k_accuracy(y_prob, y_true, k=5)

    report = classification_report(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    report_txt = classification_report(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), target_names=class_names,
        digits=4, zero_division=0,
    )
    (results_dir / f"classification_report_{split}.txt").write_text(
        report_txt, encoding="utf-8"
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    per_class = []
    for i in range(NUM_CLASSES):
        r = report[class_names[i]]
        per_class.append({
            "class_id": i, "name": class_names[i],
            "precision": r["precision"], "recall": r["recall"],
            "f1": r["f1-score"], "support": int(r["support"]),
        })
    pd.DataFrame(per_class).to_csv(
        results_dir / f"per_class_metrics_{split}.csv", index=False
    )

    # Off-diagonal confusion pairs, most frequent first.
    confusions = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    "true_id": i, "true_name": class_names[i],
                    "pred_id": j, "pred_name": class_names[j],
                    "count": int(cm[i, j]),
                    "pct_of_true": round(100 * cm[i, j] / max(cm[i].sum(), 1), 2),
                })
    confusions.sort(key=lambda d: -d["count"])
    pd.DataFrame(confusions[:20]).to_csv(
        results_dir / f"top_confusions_{split}.csv", index=False
    )

    print(f"\n  accuracy           {acc:.4f}")
    print(f"  balanced accuracy  {balanced:.4f}")
    print(f"  macro F1           {macro_f1:.4f}")
    print(f"  weighted F1        {weighted_f1:.4f}")
    print(f"  top-5 accuracy     {top5:.4f}")
    print(f"  inference          {1000 * infer_seconds / len(ds):.3f} ms/image")

    if confusions:
        print("\n  most frequent confusions:")
        for c in confusions[:5]:
            print(f"    {c['true_id']:>2} -> {c['pred_id']:>2}  {c['count']:>4}x  "
                  f"({c['true_name'][:28]} -> {c['pred_name'][:28]})")

    print("\n[evaluate] writing figures")
    plot_confusion_matrix(cm, class_names, results_dir / f"confusion_matrix_{split}.png")
    plot_per_class_f1([p["f1"] for p in per_class], class_names,
                      results_dir / f"per_class_f1_{split}.png")

    # Most confident errors: these expose systematic confusions rather than noise.
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong):
        conf = y_prob[wrong, y_pred[wrong]]
        order = wrong[np.argsort(-conf)][:25]
        images = [ds.cache[i] if ds.cache is not None else ds._load_one(i) for i in order]
        plot_misclassified_grid(
            images, y_true[order], y_pred[order], y_prob[order, y_pred[order]],
            class_names, results_dir / f"misclassified_{split}.png",
        )

    metrics = {
        "run": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "torch_version": torch.__version__,
            "device": str(device), "seed": cfg.seed, "split": split,
            "checkpoint": str(checkpoint),
        },
        "model": {
            "name": "TrafficSignNet",
            "params": ck.get("n_params", count_parameters(model)),
            "img_size": cfg.img_size, "channels": list(cfg.channels),
            "use_clahe": cfg.use_clahe, "balance": cfg.balance,
        },
        "training": {
            "epochs_run": ck.get("epoch"),
            "best_val_acc": ck.get("val_acc"),
        },
        split: {
            "n_samples": int(len(ds)),
            "accuracy": float(acc),
            "balanced_accuracy": float(balanced),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "top5_accuracy": float(top5),
            "n_errors": int(len(wrong)),
            "inference_ms_per_image": round(1000 * infer_seconds / len(ds), 4),
        },
        "per_class": per_class,
        "top_confusions": confusions[:20],
    }

    out_json = results_dir / ("metrics.json" if split == "test" else f"metrics_{split}.json")
    save_json(metrics, out_json)

    print(f"\n{split.upper()} ACCURACY: {acc:.4f}")
    print(f"[evaluate] metrics written to {out_json}")
    return metrics
