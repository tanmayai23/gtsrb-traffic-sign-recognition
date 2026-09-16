"""Figures made specifically for the report.

These complement the ones `evaluate` writes: they argue a point rather than
just display a metric.

    python -m src.report_figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CLASS_NAMES, PROJECT_ROOT, RESULTS_DIR
from .utils import ensure_dir

# One palette for every report figure, so they read as a set.
BLUE, RED, GREY, GREEN, ORANGE = "#2c6fbb", "#c0392b", "#7f8c8d", "#27ae60", "#e67e22"


def _save(fig, path: Path, dpi: int = 200) -> Path:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}")
    return path


def fig_split_illustration(out: Path) -> Path:
    """Why a random split leaks: the same sign lands on both sides."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.4))
    rng = np.random.default_rng(3)

    n_tracks, frames = 6, 10

    # Top: random per-image split.
    for t in range(n_tracks):
        for f in range(frames):
            is_val = rng.random() < 0.25
            ax1.add_patch(mpatches.Rectangle(
                (f, n_tracks - t - 1), 0.88, 0.8,
                facecolor=ORANGE if is_val else BLUE, edgecolor="white", lw=1.2))
    ax1.set_title("Random split by image — frames of one sign land on both sides",
                  fontsize=11.5, fontweight="bold", color=RED, pad=10)

    # Bottom: split by track.
    val_tracks = {1, 4}
    for t in range(n_tracks):
        for f in range(frames):
            is_val = t in val_tracks
            ax2.add_patch(mpatches.Rectangle(
                (f, n_tracks - t - 1), 0.88, 0.8,
                facecolor=ORANGE if is_val else BLUE, edgecolor="white", lw=1.2))
    ax2.set_title("Split by track — every frame of a sign stays on one side",
                  fontsize=11.5, fontweight="bold", color=GREEN, pad=10)

    for ax in (ax1, ax2):
        ax.set_xlim(-0.2, frames + 2.6)
        ax.set_ylim(-0.3, n_tracks + 0.1)
        ax.set_xticks([])
        ax.set_yticks([n_tracks - t - 0.6 for t in range(n_tracks)])
        ax.set_yticklabels([f"sign {t + 1}" for t in range(n_tracks)], fontsize=8.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(frames + 0.35, n_tracks - 0.6, "← one row = ~30 frames\n    of the SAME sign",
                fontsize=8, color=GREY, va="top")

    fig.legend(handles=[mpatches.Patch(facecolor=BLUE, label="training"),
                        mpatches.Patch(facecolor=ORANGE, label="validation")],
               loc="lower center", ncol=2, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return _save(fig, out)


def fig_ablation_comparison(results_dir: Path, ablations_dir: Path, out: Path) -> Path:
    """Each design decision, measured by removing it."""
    base = json.load(open(results_dir / "metrics.json"))
    runs = [("Baseline", base, BLUE)]
    for key, label in (("no_clahe", "No CLAHE"),
                       ("no_balance", "No rebalancing"),
                       ("half_width", "Half width")):
        p = ablations_dir / key / "metrics.json"
        if p.exists():
            runs.append((label, json.load(open(p)), RED))

    labels = [r[0] for r in runs]
    acc = [r[1]["test"]["accuracy"] * 100 for r in runs]
    bal = [r[1]["test"]["balanced_accuracy"] * 100 for r in runs]
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    b1 = ax.bar(x - w / 2, acc, w, label="accuracy", color=BLUE, edgecolor="black", lw=0.5)
    b2 = ax.bar(x + w / 2, bal, w, label="balanced accuracy", color="#8fb8e0",
                edgecolor="black", lw=0.5)

    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 2.5), textcoords="offset points",
                        ha="center", fontsize=8.5)

    # Deltas go above each pair, in clear space, never over a filled bar.
    for i in range(1, len(runs)):
        ax.annotate(f"{acc[i] - acc[0]:+.2f} pp",
                    (i, max(acc[i], bal[i])),
                    xytext=(0, 17), textcoords="offset points",
                    ha="center", fontsize=10, fontweight="bold", color=RED)

    ax.axhline(acc[0], color=GREY, ls=":", lw=1.2, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("percent")
    ax.set_ylim(88, 100)
    ax.set_title("Ablation: removing one design decision at a time",
                 fontweight="bold", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, out)


def fig_confusion_families(results_dir: Path, out: Path) -> Path:
    """Errors are concentrated in visually similar groups, not spread evenly."""
    conf = pd.read_csv(results_dir / "top_confusions_test.csv")

    speed = set(range(0, 9)) | {6}
    triangles = {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
    derestrict = {6, 32, 41, 42}

    def family(t, p):
        if t in speed and p in speed:
            return "speed limit ↔ speed limit"
        if t in triangles and p in triangles:
            return "warning triangle ↔ warning triangle"
        if t in derestrict and p in derestrict:
            return "end-of-restriction ↔ end-of-restriction"
        return "other"

    conf["family"] = [family(t, p) for t, p in zip(conf["true_id"], conf["pred_id"])]
    grouped = conf.groupby("family")["count"].sum().sort_values()

    colors = [GREY if f == "other" else RED for f in grouped.index]
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    bars = ax.barh(range(len(grouped)), grouped.values, color=colors,
                   edgecolor="black", lw=0.5)
    for bar, v in zip(bars, grouped.values):
        ax.annotate(f"{v}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)

    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels(grouped.index, fontsize=9.5)
    ax.set_xlabel("misclassified images (top-20 confusion pairs)")
    ax.set_title("Errors cluster inside visually similar families",
                 fontweight="bold", fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, out)


def fig_architecture(out: Path) -> Path:
    """Layer diagram with tensor shapes."""
    fig, ax = plt.subplots(figsize=(11.5, 3.2))

    stages = [
        ("input\n3x32x32", 3, BLUE),
        ("Conv 3→32\nBN, ReLU\nMaxPool", 32, "#4a86c8"),
        ("Conv 32→64\nBN, ReLU\nMaxPool", 64, "#6b9fd6"),
        ("Conv 64→128\nBN, ReLU\nMaxPool", 128, "#8fb8e0"),
        ("GlobalAvgPool\nDropout 0.3", 128, GREEN),
        ("Linear\n128→43", 43, ORANGE),
    ]
    shapes = ["3x32x32", "32x16x16", "64x8x8", "128x4x4", "128", "43"]

    x, width, gap = 0.0, 1.55, 0.42
    for i, ((label, _, color), shape) in enumerate(zip(stages, shapes)):
        h = 1.5
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, (2 - h) / 2), width, h, boxstyle="round,pad=0.03",
            facecolor=color, edgecolor="black", lw=0.8, alpha=0.9))
        ax.text(x + width / 2, 1.0, label, ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold")
        ax.text(x + width / 2, (2 - h) / 2 - 0.22, shape, ha="center",
                va="top", fontsize=8, color=GREY, family="monospace")
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + width + gap, 1.0), xytext=(x + width, 1.0),
                        arrowprops=dict(arrowstyle="->", lw=1.3, color="#444"))
        x += width + gap

    ax.set_xlim(-0.2, x)
    ax.set_ylim(-0.1, 2.15)
    ax.axis("off")
    ax.set_title("TrafficSignNet — 99,019 parameters", fontweight="bold",
                 fontsize=12, pad=4)
    fig.tight_layout()
    return _save(fig, out)


def fig_per_class_accuracy(results_dir: Path, out: Path) -> Path:
    """Per-class recall against class frequency: is the tail being neglected?"""
    m = json.load(open(results_dir / "metrics.json"))
    pc = m["per_class"]
    support = np.array([p["support"] for p in pc])
    recall = np.array([p["recall"] for p in pc])

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.scatter(support, recall * 100, s=46, c=[RED if r < 0.9 else BLUE for r in recall],
               edgecolors="black", linewidths=0.6, zorder=3)

    for p in pc:
        if p["recall"] < 0.9 or p["support"] >= 700:
            ax.annotate(f"{p['class_id']}", (p["support"], p["recall"] * 100),
                        xytext=(5, 4), textcoords="offset points", fontsize=8,
                        fontweight="bold")

    ax.axhline(m["test"]["accuracy"] * 100, color=GREY, ls="--", lw=1.2,
               label=f"overall accuracy {m['test']['accuracy'] * 100:.2f}%")
    ax.set_xlabel("test images in class (log scale)")
    ax.set_ylabel("recall (%)")
    ax.set_xscale("log")
    ax.set_ylim(55, 102)
    ax.set_title("Per-class recall against class size", fontweight="bold", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _save(fig, out)


def main() -> int:
    out_dir = ensure_dir(RESULTS_DIR / "report")
    results_dir, ablations_dir = RESULTS_DIR, PROJECT_ROOT / "ablations"

    print("[report-figures] writing")
    fig_split_illustration(out_dir / "split_illustration.png")
    fig_architecture(out_dir / "architecture.png")
    if (ablations_dir / "no_clahe" / "metrics.json").exists():
        fig_ablation_comparison(results_dir, ablations_dir,
                                out_dir / "ablation_comparison.png")
    fig_confusion_families(results_dir, out_dir / "confusion_families.png")
    fig_per_class_accuracy(results_dir, out_dir / "per_class_recall.png")
    print("[report-figures] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
