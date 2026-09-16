"""Command line interface. Every capability of this project is reachable here.

    python -m src.cli <command> [options]

Modules are imported inside each handler so that --help stays instant and
commands that do not need torch do not pay to import it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (CHECKPOINT_DIR, CLASS_NAMES, DATA_DIR, NUM_CLASSES,
                     RESULTS_DIR, Config)

EPILOG = """\
typical run:
  python -m src.cli prepare                     download + split (~6 min, once)
  python -m src.cli train                       train the CNN (~20 min on CPU)
  python -m src.cli evaluate                    metrics + figures on the test set
  python -m src.cli predict --image path.ppm    classify one image
  python -m src.cli gradcam --image path.ppm    save an explanation heatmap

smoke test (under a minute, verifies the whole pipeline):
  python -m src.cli train --quick
"""


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="dataset root (default: ./data)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="GTSRB traffic sign recognition: a CNN trained from scratch.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    # ---- prepare ----
    p = sub.add_parser("prepare", help="download the dataset and build manifests",
                       description="Download GTSRB, extract it, and write "
                                   "track-aware train/val/test manifests.")
    _add_common(p)
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="fraction of TRACKS held out for validation (default: 0.2)")
    p.add_argument("--img-size", type=int, default=32,
                   help="size used for normalisation statistics (default: 32)")
    p.add_argument("--no-clahe", action="store_true",
                   help="compute statistics without CLAHE")
    p.add_argument("--force-download", action="store_true",
                   help="re-download and re-extract even if data exists")

    # ---- train ----
    p = sub.add_parser("train", help="train the model",
                       description="Train TrafficSignNet on the prepared split.")
    _add_common(p)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-lr", type=float, default=3e-3, help="OneCycle peak LR")
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--width-mult", type=float, default=1.0,
                   help="scale every conv width (capacity ablation)")
    p.add_argument("--balance", choices=["sampler", "loss", "none"], default="sampler",
                   help="how to handle the 10x class imbalance (default: sampler)")
    p.add_argument("--no-clahe", action="store_true", help="disable CLAHE (ablation)")
    p.add_argument("--workers", type=int, default=0,
                   help="dataloader workers; 0 is fastest here since data is cached")
    p.add_argument("--patience", type=int, default=6, help="early-stopping patience")
    p.add_argument("--max-minutes", type=float, default=30.0,
                   help="wall-clock budget; stops cleanly when exceeded")
    p.add_argument("--quick", action="store_true",
                   help="1 epoch on a small subset: smoke test, under a minute")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--out", type=Path, default=CHECKPOINT_DIR)
    p.add_argument("--results", type=Path, default=RESULTS_DIR)

    # ---- evaluate ----
    p = sub.add_parser("evaluate", help="evaluate a checkpoint on the test set",
                       description="Compute metrics and write figures.")
    _add_common(p)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    p.add_argument("--split", choices=["test", "val", "train"], default="test")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--results", type=Path, default=RESULTS_DIR)

    # ---- predict ----
    p = sub.add_parser("predict", help="classify a single image")
    p.add_argument("--image", type=Path, required=True, help="path to .ppm/.png/.jpg")
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="emit JSON only (for scripting)")

    # ---- gradcam ----
    p = sub.add_parser("gradcam", help="save a Grad-CAM explanation for an image")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    p.add_argument("--target-class", type=int, default=None,
                   help=f"explain this class instead of the prediction (0-{NUM_CLASSES - 1})")
    p.add_argument("--alpha", type=float, default=0.45, help="overlay strength")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "gradcam")

    # ---- figures ----
    p = sub.add_parser("figures", help="write preprocessing/augmentation figures")
    _add_common(p)
    p.add_argument("--results", type=Path, default=RESULTS_DIR)
    p.add_argument("--img-size", type=int, default=32)

    # ---- report ----
    p = sub.add_parser("report", help="build the project report (HTML + PDF)",
                       description="Regenerate report figures and build the report. "
                                   "PDF export needs Chrome or Edge installed.")
    _add_common(p)
    p.add_argument("--no-pdf", action="store_true",
                   help="write only the HTML (print it to PDF yourself)")

    # ---- info ----
    p = sub.add_parser("info", help="print environment and project status")
    _add_common(p)
    p.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")

    # ---- classes ----
    sub.add_parser("classes", help="list the 43 class ids and names")

    return parser


def cmd_prepare(args) -> int:
    from .prepare import run_prepare
    run_prepare(args.data_dir, val_frac=args.val_frac, seed=args.seed,
                img_size=args.img_size, use_clahe=not args.no_clahe,
                force_download=args.force_download)
    print("\nNext:\n  python -m src.cli train")
    return 0


def cmd_train(args) -> int:
    from .train import run_training
    cfg = Config(
        img_size=args.img_size, batch_size=args.batch_size, epochs=args.epochs,
        lr=args.lr, max_lr=args.max_lr, dropout=args.dropout,
        width_mult=args.width_mult, balance=args.balance,
        use_clahe=not args.no_clahe, seed=args.seed, workers=args.workers,
        patience=args.patience, max_minutes=args.max_minutes,
    )
    run_training(cfg, args.data_dir, args.out, args.results,
                 quick=args.quick, resume=args.resume)
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import run_evaluation
    run_evaluation(args.checkpoint, args.data_dir, args.results,
                   split=args.split, batch_size=args.batch_size,
                   workers=args.workers)
    return 0


def cmd_predict(args) -> int:
    from .predict import run_predict
    run_predict(args.checkpoint, args.image, topk=args.topk, as_json=args.as_json)
    return 0


def cmd_gradcam(args) -> int:
    from .gradcam import run_gradcam
    run_gradcam(args.checkpoint, args.image, args.out,
                target_class=args.target_class, alpha=args.alpha)
    return 0


def cmd_figures(args) -> int:
    import numpy as np
    import pandas as pd

    from .plots import plot_augmentation_samples, plot_preprocessing_samples
    from .transforms import augment, load_image, preprocess_deterministic
    from .utils import ensure_dir

    manifest = Path(args.data_dir) / "manifests" / "train.csv"
    if not manifest.exists():
        print("No manifests. Run `python -m src.cli prepare` first.", file=sys.stderr)
        return 1

    results = ensure_dir(args.results)
    df = pd.read_csv(manifest)
    rng = np.random.default_rng(args.seed)

    # One example from each of six well-separated classes.
    picks = []
    for cid in [14, 1, 17, 25, 33, 12]:
        rows = df[df["class_id"] == cid]
        if len(rows):
            picks.append(rows.iloc[int(rng.integers(len(rows)))])

    raws, procs, labels = [], [], []
    for row in picks:
        raw = load_image(row["path"])
        raws.append(raw)
        procs.append(preprocess_deterministic(
            raw, (row["x1"], row["y1"], row["x2"], row["y2"]), args.img_size, True))
        labels.append(row["class_id"])

    out1 = plot_preprocessing_samples(raws, procs, labels, CLASS_NAMES,
                                      results / "preprocessing_samples.png")
    print(f"  wrote {out1}")

    base = procs[0]
    variants = [augment(base, np.random.default_rng(args.seed + i)) for i in range(5)]
    out2 = plot_augmentation_samples(base, variants,
                                     results / "augmentation_samples.png")
    print(f"  wrote {out2}")
    return 0


def cmd_report(args) -> int:
    from .build_report import build
    from .report_figures import main as figures_main

    figures_main()
    html = build()

    if args.no_pdf:
        return 0

    pdf = html.parent / "GTSRB_Project_Report.pdf"
    browsers = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "/usr/bin/google-chrome", "/usr/bin/chromium",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]
    exe = next((b for b in browsers if Path(b).exists()), None)
    if exe is None:
        print("\n[report] no Chrome/Edge found for PDF export.")
        print(f"[report] open {html} in a browser and print to PDF.")
        return 0

    import subprocess
    subprocess.run([exe, "--headless", "--disable-gpu", "--no-sandbox",
                    "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    html.resolve().as_uri()],
                   capture_output=True, timeout=300)
    if pdf.exists():
        print(f"[report] wrote {pdf}  ({pdf.stat().st_size / 1e6:.1f} MB)")
    return 0


def cmd_info(args) -> int:
    import torch

    from .model import TrafficSignNet, count_parameters
    from .utils import configure_threads, get_device

    print("environment")
    print(f"  python          {sys.version.split()[0]}")
    print(f"  torch           {torch.__version__}")
    print(f"  device          {get_device()}")
    print(f"  cuda available  {torch.cuda.is_available()}")
    print(f"  threads         {configure_threads()}")

    print("\nmodel")
    print(f"  TrafficSignNet  {count_parameters(TrafficSignNet()):,} parameters")
    print(f"  classes         {NUM_CLASSES}")

    print("\ndata")
    manifest_dir = Path(args.data_dir) / "manifests"
    for name in ("train", "val", "test"):
        path = manifest_dir / f"{name}.csv"
        if path.exists():
            n = sum(1 for _ in open(path, encoding="utf-8")) - 1
            print(f"  {name:<15} {n:>6} images  ({path})")
        else:
            print(f"  {name:<15} missing -- run `python -m src.cli prepare`")

    summary = manifest_dir / "split_summary.json"
    if summary.exists():
        from .utils import load_json
        s = load_json(summary)
        print(f"  tracks          {s['n_train_tracks']} train / {s['n_val_tracks']} val")
        print(f"  track overlap   {s['track_overlap']}")
        print(f"  imbalance       {s['imbalance_ratio']}x")

    print("\ncheckpoint")
    if Path(args.checkpoint).exists():
        from .utils import load_checkpoint
        ck = load_checkpoint(args.checkpoint)
        print(f"  {args.checkpoint}")
        print(f"  epoch {ck.get('epoch')}  val acc {ck.get('val_acc', float('nan')):.4f}"
              f"  trained {ck.get('timestamp', 'n/a')}")
    else:
        print(f"  none at {args.checkpoint} -- run `python -m src.cli train`")

    return 0


def cmd_classes(args) -> int:
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i:>2}  {name}")
    return 0


HANDLERS = {
    "prepare": cmd_prepare, "train": cmd_train, "evaluate": cmd_evaluate,
    "predict": cmd_predict, "gradcam": cmd_gradcam, "figures": cmd_figures,
    "report": cmd_report, "info": cmd_info, "classes": cmd_classes,
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return HANDLERS[args.command](args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
