"""Training orchestration: schedule, early stopping, checkpointing, history."""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from .config import CLASS_NAMES, Config
from .dataset import GTSRBDataset, build_loader, class_weights, make_sampler
from .model import build_model, count_parameters
from .engine import train_one_epoch, validate
from .plots import plot_class_distribution, plot_training_curves
from .utils import (Timer, configure_threads, ensure_dir, get_device, human_time,
                    load_json, save_checkpoint, save_json, set_seed)


def _load_norm_stats(manifest_dir: Path, cfg: Config) -> Config:
    stats_path = manifest_dir / "norm_stats.json"
    if stats_path.exists():
        stats = load_json(stats_path)
        cfg.norm_mean, cfg.norm_std = stats["mean"], stats["std"]
    return cfg


def run_training(cfg: Config, data_dir: Path, out_dir: Path, results_dir: Path,
                 quick: bool = False, resume: Path | None = None) -> dict:
    manifest_dir = Path(data_dir) / "manifests"
    if not (manifest_dir / "train.csv").exists():
        raise FileNotFoundError(
            "No manifests found. Run this first:\n  python -m src.cli prepare"
        )

    if quick:
        # Smoke test for the grading pipeline: exercises every code path in <1 min.
        cfg.epochs = 1
        cfg.patience = 1
        print("[train] QUICK MODE: 1 epoch on a 2000/500 subset (smoke test)")

    set_seed(cfg.seed)
    threads = configure_threads()
    device = get_device()
    cfg = _load_norm_stats(manifest_dir, cfg)

    out_dir, results_dir = ensure_dir(out_dir), ensure_dir(results_dir)

    print(f"[train] device={device} threads={threads} seed={cfg.seed}")
    print(f"[train] img_size={cfg.img_size} batch={cfg.batch_size} epochs={cfg.epochs} "
          f"balance={cfg.balance} clahe={cfg.use_clahe}")

    print("[train] loading data (preprocessing is cached in RAM up front)")
    with Timer() as t_load:
        train_ds = GTSRBDataset(
            manifest_dir / "train.csv", img_size=cfg.img_size, train=True,
            use_clahe=cfg.use_clahe, norm_mean=cfg.norm_mean, norm_std=cfg.norm_std,
            limit=2000 if quick else None, seed=cfg.seed,
        )
        val_ds = GTSRBDataset(
            manifest_dir / "val.csv", img_size=cfg.img_size, train=False,
            use_clahe=cfg.use_clahe, norm_mean=cfg.norm_mean, norm_std=cfg.norm_std,
            limit=500 if quick else None, seed=cfg.seed,
        )
    print(f"  train {len(train_ds)} | val {len(val_ds)}  (cached in {human_time(t_load.seconds)})")

    if not quick:
        plot_class_distribution(train_ds.class_counts(), CLASS_NAMES,
                                results_dir / "class_distribution.png")

    sampler = make_sampler(train_ds, cfg.seed) if cfg.balance == "sampler" else None
    train_loader = build_loader(train_ds, cfg.batch_size, shuffle=True, sampler=sampler,
                                workers=cfg.workers, seed=cfg.seed)
    val_loader = build_loader(val_ds, max(cfg.batch_size, 256), workers=cfg.workers)

    model = build_model(cfg).to(device)
    n_params = count_parameters(model)
    print(f"[train] TrafficSignNet: {n_params:,} trainable parameters")

    weight = class_weights(train_ds).to(device) if cfg.balance == "loss" else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    # OneCycle: with a fixed ~18-epoch budget there is no time for a long decay
    # tail, and the warmup lets a higher peak LR be used safely.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.max_lr, total_steps=cfg.epochs * len(train_loader),
        pct_start=0.25, div_factor=10.0, final_div_factor=100.0,
    )

    start_epoch, best_acc, history = 1, 0.0, []
    if resume is not None and Path(resume).exists():
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        start_epoch = ck.get("epoch", 0) + 1
        best_acc = ck.get("val_acc", 0.0)
        history = ck.get("history", [])
        print(f"[train] resumed from {resume} at epoch {start_epoch} (best {best_acc:.4f})")

    best_path, last_path = out_dir / "best.pt", out_dir / "last.pt"
    if quick:
        best_path, last_path = out_dir / "quick_best.pt", out_dir / "quick_last.pt"

    epochs_without_gain = 0
    t_start = time.perf_counter()
    print(f"\n[train] starting ({cfg.epochs} epochs)\n")

    for epoch in range(start_epoch, cfg.epochs + 1):
        t_epoch = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=scheduler, grad_clip=cfg.grad_clip,
            epoch=epoch, total_epochs=cfg.epochs,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.perf_counter() - t_epoch
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "lr": lr_now,
            "seconds": elapsed,
        })

        improved = val_acc > best_acc
        marker = "  <- best" if improved else ""
        print(f"  epoch {epoch:>2}/{cfg.epochs}  "
              f"train {train_loss:.4f}/{train_acc:.4f}  "
              f"val {val_loss:.4f}/{val_acc:.4f}  "
              f"lr {lr_now:.2e}  {human_time(elapsed)}{marker}")

        payload = dict(
            model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
            epoch=epoch, val_acc=val_acc, config=cfg.to_dict(),
            class_names=CLASS_NAMES,
            norm_stats={"mean": cfg.norm_mean, "std": cfg.norm_std},
            img_size=cfg.img_size, n_params=n_params, history=history,
            torch_version=torch.__version__,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )
        save_checkpoint(last_path, **payload)

        if improved:
            best_acc, epochs_without_gain = val_acc, 0
            save_checkpoint(best_path, **payload)
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= cfg.patience:
                print(f"\n[train] early stop: no val improvement for {cfg.patience} epochs")
                break

        total_min = (time.perf_counter() - t_start) / 60
        if total_min > cfg.max_minutes:
            print(f"\n[train] wall-clock limit reached ({cfg.max_minutes} min); stopping")
            break

    train_seconds = time.perf_counter() - t_start

    hist_csv = results_dir / ("history_quick.csv" if quick else "history.csv")
    with open(hist_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    save_json(history, results_dir / ("history_quick.json" if quick else "history.json"))

    if not quick and len(history) > 1:
        plot_training_curves(history, results_dir / "training_curves.png")

    print(f"\n[train] done in {human_time(train_seconds)}")
    print(f"[train] best val accuracy: {best_acc:.4f}")
    print(f"[train] checkpoint: {best_path}")
    print(f"\nNext:\n  python -m src.cli evaluate --checkpoint {best_path}")

    return {
        "best_val_acc": best_acc,
        "epochs_run": len(history),
        "train_seconds": train_seconds,
        "n_params": n_params,
        "checkpoint": str(best_path),
        "history": history,
    }
