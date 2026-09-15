"""Figure writers.

Every function saves to a path and returns it. Nothing calls plt.show(): the
project must run over SSH or in a grading container with no display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import

import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir


def _save(fig, out_path: Path, dpi: int = 150) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)  # matplotlib warns and leaks after 20 open figures
    return out_path


def plot_training_curves(history: list[dict], out_path: Path) -> Path:
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, [h["train_loss"] for h in history], "o-", label="train", lw=1.8, ms=4)
    ax1.plot(epochs, [h["val_loss"] for h in history], "s-", label="val", lw=1.8, ms=4)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [h["train_acc"] for h in history], "o-", label="train", lw=1.8, ms=4)
    ax2.plot(epochs, [h["val_acc"] for h in history], "s-", label="val", lw=1.8, ms=4)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    best = max(history, key=lambda h: h["val_acc"])
    ax2.axhline(best["val_acc"], color="grey", ls=":", lw=1)
    ax2.annotate(
        f"best val {best['val_acc']:.4f} (epoch {best['epoch']})",
        xy=(best["epoch"], best["val_acc"]),
        xytext=(0.35, 0.12), textcoords="axes fraction", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="grey", lw=1),
    )

    fig.suptitle("Training history", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path,
                          normalize: bool = True) -> Path:
    m = cm.astype(np.float64)
    if normalize:
        m = m / np.maximum(m.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="fraction of true class" if normalize else "count")

    n = len(class_names)
    labels = [f"{i}: {c}" for i, c in enumerate(class_names)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (row-normalised)" if normalize else "Confusion matrix",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_misclassified_grid(images, true_labels, pred_labels, confidences,
                            class_names, out_path: Path,
                            rows: int = 5, cols: int = 5) -> Path:
    """Grid of errors.

    Callers pass the most confident mistakes: those reveal systematic confusions,
    whereas randomly chosen errors are mostly unreadable crops.
    """
    n = min(len(images), rows * cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(images[i])
        t, p = int(true_labels[i]), int(pred_labels[i])
        ax.set_title(
            f"true {t}: {class_names[t][:20]}\npred {p}: {class_names[p][:20]}\n"
            f"({confidences[i]:.2f})",
            fontsize=6.5, color="darkred",
        )

    fig.suptitle("Highest-confidence misclassifications", fontweight="bold", y=0.995)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_class_distribution(counts, class_names, out_path: Path,
                            title: str = "Training images per class") -> Path:
    counts = np.asarray(counts)
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(range(len(counts)), counts, color="steelblue", edgecolor="black", lw=0.4)

    hi, lo = int(counts.argmax()), int(counts.argmin())
    bars[hi].set_color("darkgreen")
    bars[lo].set_color("darkred")

    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([str(i) for i in range(len(counts))], fontsize=6)
    ax.set_xlabel("class id")
    ax.set_ylabel("images")
    ax.set_title(f"{title}  (imbalance {counts.max() / max(counts.min(), 1):.1f}x)",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(f"max {counts.max()}\n{class_names[hi][:24]}", xy=(hi, counts.max()),
                xytext=(hi + 1.5, counts.max() * 0.94), fontsize=7, color="darkgreen")
    ax.annotate(f"min {counts.min()}\n{class_names[lo][:24]}", xy=(lo, counts.min()),
                xytext=(lo + 1.5, counts.max() * 0.55), fontsize=7, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))
    fig.tight_layout()
    return _save(fig, out_path)


def plot_preprocessing_samples(raw_images, processed_images, labels, class_names,
                               out_path: Path, n: int = 6) -> Path:
    """Before/after strip, the clearest evidence that CLAHE does something."""
    n = min(n, len(raw_images))
    fig, axes = plt.subplots(2, n, figsize=(n * 2.1, 4.8))
    axes = np.atleast_2d(axes)

    for i in range(n):
        axes[0, i].imshow(raw_images[i])
        axes[0, i].axis("off")
        axes[0, i].set_title(class_names[int(labels[i])][:18], fontsize=7)
        axes[1, i].imshow(processed_images[i])
        axes[1, i].axis("off")

    fig.text(0.02, 0.72, "raw", rotation=90, fontsize=10, fontweight="bold")
    fig.text(0.02, 0.24, "ROI crop\n+ CLAHE\n+ resize", rotation=90, fontsize=9,
             fontweight="bold")
    fig.suptitle("Preprocessing pipeline", fontweight="bold")
    fig.tight_layout(rect=[0.05, 0, 1, 0.96])
    return _save(fig, out_path)


def plot_augmentation_samples(base_image, variants, out_path: Path) -> Path:
    fig, axes = plt.subplots(1, len(variants) + 1,
                             figsize=((len(variants) + 1) * 1.9, 2.4))
    axes[0].imshow(base_image)
    axes[0].axis("off")
    axes[0].set_title("original", fontsize=8, fontweight="bold")
    for ax, v in zip(axes[1:], variants):
        ax.imshow(v)
        ax.axis("off")
        ax.set_title("augmented", fontsize=8)
    fig.suptitle("Augmentation samples (no flips: mirroring changes sign meaning)",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_gradcam_panel(original, heatmap, overlay, title: str, out_path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6))
    for ax, img, name in zip(axes, [original, heatmap, overlay],
                             ["input", "Grad-CAM", "overlay"]):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(name, fontsize=10)
    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_per_class_f1(f1_scores, class_names, out_path: Path) -> Path:
    order = np.argsort(f1_scores)
    fig, ax = plt.subplots(figsize=(9, 11))
    colors = ["darkred" if f1_scores[i] < 0.9 else "steelblue" for i in order]
    ax.barh(range(len(order)), [f1_scores[i] for i in order], color=colors,
            edgecolor="black", lw=0.4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{i}: {class_names[i][:34]}" for i in order], fontsize=7)
    ax.set_xlabel("F1 score")
    ax.set_xlim(0, 1.02)
    ax.set_title("Per-class F1 (ascending)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, out_path)
