"""Design artefacts: system architecture, workflow, use case, class and sequence
diagrams.

These are drawn from code for the same reason the result figures are: a diagram
pasted in from a drawing tool drifts the moment the code moves, and nobody
notices. Here the module names and call order below are the ones in `src/`, so a
rename that is not mirrored here shows up as a wrong diagram rather than a stale
one.

    python -m src.design_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from .config import PROJECT_ROOT, RESULTS_DIR
from .utils import ensure_dir

# Same palette as report_figures, so the design diagrams and the result figures
# read as one set inside the report.
BLUE, RED, GREY, GREEN, ORANGE = "#2c6fbb", "#c0392b", "#7f8c8d", "#27ae60", "#e67e22"
LIGHT = "#eef3f9"


def _save(fig, path: Path, dpi: int = 200) -> Path:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}")
    return path


def _box(ax, x, y, w, h, label, facecolor, *, fontsize=8.5, textcolor="white",
         boxstyle="round,pad=0.02", lw=0.9, edgecolor="black", weight="bold"):
    """A labelled rounded rectangle, centred text. Returns the centre."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle=boxstyle, facecolor=facecolor,
        edgecolor=edgecolor, lw=lw))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=textcolor, fontweight=weight, linespacing=1.45)
    return x + w / 2, y + h / 2


def _arrow(ax, xy_from, xy_to, *, style="->", color="#444", lw=1.3, ls="-",
           label=None, label_offset=(0, 0.12), fontsize=7.5, rad=0.0):
    ax.annotate("", xy=xy_to, xytext=xy_from,
                arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                                linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx = (xy_from[0] + xy_to[0]) / 2 + label_offset[0]
        my = (xy_from[1] + xy_to[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
                color="#333", bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                        ec="none", alpha=0.85))


# --------------------------------------------------------------------------
# 1. System architecture
# --------------------------------------------------------------------------

def fig_system_architecture(out: Path) -> Path:
    """Layered view: interface, application modules, domain core, storage."""
    fig, ax = plt.subplots(figsize=(11.5, 7.4))

    # Layer bands, drawn first so the module boxes sit on top.
    bands = [
        (5.55, 1.15, "Interface layer", "#dce8f5"),
        (3.75, 1.55, "Application layer  —  use cases", "#e6f0e6"),
        (1.95, 1.55, "Domain layer  —  data, model, transforms", "#fdf0e2"),
        (0.15, 1.5, "Storage layer  —  filesystem artefacts", "#efefef"),
    ]
    for y, h, title, color in bands:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.1, y), 13.8, h, boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="#b9c4cf", lw=1.0))
        ax.text(0.32, y + h - 0.16, title, ha="left", va="top", fontsize=9,
                color="#4a5560", style="italic")

    _box(ax, 5.15, 5.75, 3.7, 0.68, "src/cli.py\nargparse subcommands", BLUE, fontsize=9)

    app = [
        (0.45, "prepare.py\ndownload + split"),
        (3.15, "train.py\nengine.py"),
        (5.85, "evaluate.py\nmetrics"),
        (8.55, "predict.py\ninference"),
        (11.25, "gradcam.py\nexplainability"),
    ]
    app_centres = []
    for x, label in app:
        app_centres.append(_box(ax, x, 3.95, 2.3, 0.95, label, "#4a86c8"))

    dom = [
        (0.45, "dataset.py\nDataset + cache"),
        (3.15, "transforms.py\nROI, CLAHE, augment"),
        (5.85, "model.py\nTrafficSignNet"),
        (8.55, "config.py\nrun config, classes"),
        (11.25, "utils.py / plots.py\nseeding, figures"),
    ]
    dom_centres = []
    for x, label in dom:
        dom_centres.append(_box(ax, x, 2.15, 2.3, 0.95, label, ORANGE))

    store = [
        (0.45, "data/\nimages + manifests"),
        (3.15, "checkpoints/\nbest.pt"),
        (5.85, "results/\nmetrics + figures"),
        (8.55, "ablations/\nper-run metrics"),
        (11.25, "report/\nHTML + PDF"),
    ]
    store_centres = []
    for x, label in store:
        store_centres.append(_box(ax, x, 0.4, 2.3, 0.85, label, GREY))

    # CLI fans out to every use case; each use case rests on the domain layer.
    for cx, _ in app_centres:
        _arrow(ax, (7.0, 5.75), (cx, 4.9), color="#5a6b7a", lw=1.0, rad=0.0)
    for (cx, _), (dx, _) in zip(app_centres, dom_centres):
        _arrow(ax, (cx, 3.95), (dx, 3.1), color="#8a6a3a", lw=1.0)
    for (cx, _), (sx, _) in zip(dom_centres, store_centres):
        _arrow(ax, (cx, 2.15), (sx, 1.25), color="#6f6f6f", lw=1.0, ls="--")

    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.1)
    ax.axis("off")
    ax.set_title("System architecture — GTSRB Traffic Sign Recognition",
                 fontweight="bold", fontsize=13, pad=8)
    fig.tight_layout()
    return _save(fig, out)


# --------------------------------------------------------------------------
# 2. Process flow / workflow
# --------------------------------------------------------------------------

def fig_workflow(out: Path) -> Path:
    """End-to-end pipeline, with the decision points that actually branch."""
    fig, ax = plt.subplots(figsize=(11.5, 6.6))

    def ellipse(x, y, w, h, label, color):
        ax.add_patch(mpatches.Ellipse((x, y), w, h, facecolor=color,
                                      edgecolor="black", lw=0.9))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5,
                color="white", fontweight="bold")

    def diamond(x, y, w, h, label):
        ax.add_patch(mpatches.Polygon(
            [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
            facecolor="#f4d35e", edgecolor="black", lw=0.9))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.8,
                color="#333", fontweight="bold", linespacing=1.4)

    ellipse(1.15, 5.6, 1.7, 0.66, "start", GREEN)

    _box(ax, 2.5, 5.25, 2.5, 0.7, "prepare\ndownload + extract", BLUE)
    diamond(7.05, 5.6, 2.2, 1.0, "archives\ncached?")
    _box(ax, 9.6, 5.25, 2.6, 0.7, "parse annotations\n(class, ROI, track)", BLUE)

    _box(ax, 9.6, 3.75, 2.6, 0.7, "track-aware split\ntrain / val manifests", ORANGE)
    _box(ax, 5.9, 3.75, 2.7, 0.7, "assert: no track\non both sides", RED)
    _box(ax, 2.2, 3.75, 2.9, 0.7, "train\naugment → forward → step", BLUE)

    diamond(1.55, 2.25, 2.4, 1.0, "val acc\nimproved?")
    _box(ax, 3.7, 1.9, 2.4, 0.7, "checkpoint\nbest.pt", GREEN)
    diamond(8.3, 2.25, 2.5, 1.0, "patience hit\nor time cap?")

    _box(ax, 9.9, 0.5, 2.6, 0.7, "evaluate\ntest metrics + figures", BLUE)
    _box(ax, 6.4, 0.5, 2.9, 0.7, "predict / gradcam\nsingle image", BLUE)
    ellipse(4.35, 0.85, 1.7, 0.66, "end", GREEN)

    # Happy path.
    _arrow(ax, (2.0, 5.6), (2.5, 5.6))
    _arrow(ax, (5.0, 5.6), (5.95, 5.6))
    _arrow(ax, (8.15, 5.6), (9.6, 5.6), label="no", label_offset=(0, 0.2))
    _arrow(ax, (10.9, 5.25), (10.9, 4.45))
    _arrow(ax, (9.6, 4.1), (8.6, 4.1))
    _arrow(ax, (5.9, 4.1), (5.1, 4.1))
    _arrow(ax, (3.65, 3.75), (2.5, 2.8))
    _arrow(ax, (2.75, 2.25), (3.7, 2.25), label="yes", label_offset=(0, 0.22))
    _arrow(ax, (6.1, 2.25), (7.05, 2.25), label="no", label_offset=(0, 0.22))
    _arrow(ax, (9.55, 2.25), (11.2, 2.25), color="#444", label="yes",
           label_offset=(0, 0.22))
    _arrow(ax, (11.2, 2.25), (11.2, 1.2))
    _arrow(ax, (9.9, 0.85), (9.3, 0.85))
    _arrow(ax, (6.4, 0.85), (5.2, 0.85))

    # Cache hit skips straight to the split, routed around the assert box
    # rather than through it.
    _arrow(ax, (7.05, 5.1), (7.05, 4.68), label="yes", label_offset=(0.42, 0.06),
           ls="--", color="#777")
    _arrow(ax, (7.05, 4.68), (10.9, 4.68), ls="--", color="#777")
    _arrow(ax, (10.9, 4.68), (10.9, 4.45), ls="--", color="#777")

    # Another epoch: leave below the train box and re-enter from the left.
    _arrow(ax, (8.3, 2.75), (8.3, 3.32), label="no", label_offset=(0.42, 0.0),
           ls="--", color="#777")
    _arrow(ax, (8.3, 3.32), (1.75, 3.32), ls="--", color="#777")
    _arrow(ax, (1.75, 3.32), (1.75, 4.1), ls="--", color="#777")
    _arrow(ax, (1.75, 4.1), (2.2, 4.1), ls="--", color="#777")
    ax.text(5.3, 3.2, "next epoch", ha="center", va="top", fontsize=7.5,
            color="#777", style="italic")

    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.set_title("Process flow — prepare → train → evaluate → infer",
                 fontweight="bold", fontsize=13, pad=8)
    fig.tight_layout()
    return _save(fig, out)


# --------------------------------------------------------------------------
# 3. Use case diagram
# --------------------------------------------------------------------------

def fig_use_case(out: Path) -> Path:
    """Two actors, eight use cases, and the include edges between them."""
    fig, ax = plt.subplots(figsize=(11.0, 6.8))

    def actor(x, y, label):
        ax.plot([x], [y + 0.42], marker="o", ms=11, mfc="white",
                mec="black", mew=1.4)
        ax.plot([x, x], [y + 0.34, y - 0.12], color="black", lw=1.4)
        ax.plot([x - 0.3, x + 0.3], [y + 0.2, y + 0.2], color="black", lw=1.4)
        ax.plot([x, x - 0.26], [y - 0.12, y - 0.6], color="black", lw=1.4)
        ax.plot([x, x + 0.26], [y - 0.12, y - 0.6], color="black", lw=1.4)
        ax.text(x, y - 0.85, label, ha="center", va="top", fontsize=9,
                fontweight="bold")

    RX, RY = 1.62, 0.46  # ellipse radii; edges are computed from these

    def usecase(x, y, label, color=LIGHT):
        ax.add_patch(mpatches.Ellipse((x, y), RX * 2, RY * 2, facecolor=color,
                                      edgecolor=BLUE, lw=1.2))
        ax.text(x, y, label, ha="center", va="center", fontsize=8.2,
                color="#1f3b57", linespacing=1.4)
        return x, y

    # System boundary.
    ax.add_patch(mpatches.FancyBboxPatch(
        (2.75, -1.15), 7.1, 7.95, boxstyle="round,pad=0.05",
        facecolor="white", edgecolor="#9aa7b4", lw=1.4, linestyle="--"))
    ax.text(6.3, 6.62, "GTSRB Traffic Sign Recognition system",
            ha="center", va="center", fontsize=9.5, color="#4a5560",
            style="italic")

    actor(1.15, 4.7, "Student /\nResearcher")
    actor(1.15, 1.7, "Evaluator /\nReviewer")

    uc_prepare = usecase(4.9, 6.0, "Prepare dataset\n(download + split)")
    uc_train = usecase(4.9, 4.9, "Train model")
    uc_eval = usecase(4.9, 3.8, "Evaluate on test set")
    uc_ablate = usecase(4.9, 2.7, "Run ablation study")
    uc_predict = usecase(4.9, 1.6, "Classify an image")
    uc_cam = usecase(4.9, 0.85, "Explain prediction\n(Grad-CAM)")

    uc_split = usecase(8.6, 5.45, "Build track-aware\nmanifests", "#fdf0e2")
    uc_fig = usecase(8.6, 3.25, "Write metrics\n+ figures", "#fdf0e2")
    # Sits low and left so the Evaluator reaches it without crossing the two
    # inference ellipses.
    uc_tests = usecase(4.9, -0.55, "Run test suite", "#e6f0e6")

    def associate(ax_pt, uc):
        """Line from an actor to the ellipse edge, aimed at its centre."""
        import math
        dx, dy = uc[0] - ax_pt[0], uc[1] - ax_pt[1]
        t = math.hypot(dx / RX, dy / RY)
        ax.plot([ax_pt[0], uc[0] - dx / t], [ax_pt[1], uc[1] - dy / t],
                color="#555", lw=1.0, zorder=0)

    for uc in (uc_prepare, uc_train, uc_eval, uc_ablate):
        associate((1.62, 4.7), uc)
    for uc in (uc_predict, uc_cam, uc_eval, uc_tests):
        associate((1.62, 1.7), uc)

    for src, dst in ((uc_prepare, uc_split), (uc_eval, uc_fig),
                     (uc_ablate, uc_fig), (uc_train, uc_fig)):
        _arrow(ax, (src[0] + RX * 0.88, src[1]), (dst[0] - RX * 0.95, dst[1]),
               style="->", ls=(0, (5, 4)), color="#8a6a3a", lw=1.0,
               label="«include»", fontsize=6.8, label_offset=(0, 0.16))

    ax.set_xlim(0.2, 10.6)
    ax.set_ylim(-1.35, 7.0)
    ax.axis("off")
    ax.set_title("Use case diagram", fontweight="bold", fontsize=13, pad=8)
    fig.tight_layout()
    return _save(fig, out)


# --------------------------------------------------------------------------
# 4. Class / component diagram
# --------------------------------------------------------------------------

def fig_class_diagram(out: Path) -> Path:
    """UML class boxes for the types that carry state, plus their relations."""
    fig, ax = plt.subplots(figsize=(12.0, 7.6))

    def uml(x, y, w, name, attrs, methods, header=BLUE):
        """Three-compartment UML class box. Height follows the content."""
        line_h = 0.235
        head_h = 0.44
        body = len(attrs) + len(methods)
        h = head_h + body * line_h + 0.26
        ax.add_patch(mpatches.Rectangle((x, y - h), w, h, facecolor="white",
                                        edgecolor="#33475b", lw=1.1))
        ax.add_patch(mpatches.Rectangle((x, y - head_h), w, head_h,
                                        facecolor=header, edgecolor="#33475b",
                                        lw=1.1))
        ax.text(x + w / 2, y - head_h / 2, name, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

        cy = y - head_h - 0.17
        for a in attrs:
            ax.text(x + 0.12, cy, a, ha="left", va="center", fontsize=7.2,
                    color="#2c3e50", family="monospace")
            cy -= line_h
        if attrs and methods:
            ax.plot([x, x + w], [cy + line_h / 2 - 0.02, cy + line_h / 2 - 0.02],
                    color="#33475b", lw=0.8)
            cy -= 0.06
        for m in methods:
            ax.text(x + 0.12, cy, m, ha="left", va="center", fontsize=7.2,
                    color="#1f3b57", family="monospace")
            cy -= line_h
        return (x + w / 2, y - h / 2), (x, y, w, h)

    cfg, cfg_b = uml(0.3, 7.35, 3.3, "Config", [
        "+ img_size: int", "+ batch_size: int", "+ epochs: int",
        "+ lr: float", "+ use_clahe: bool", "+ balance: str",
        "+ width_mult: float", "+ seed: int",
    ], ["+ from_args(args)", "+ to_dict()"], header=ORANGE)

    ds, ds_b = uml(4.35, 7.35, 3.7, "GTSRBDataset", [
        "- records: list[Record]", "- cache: ndarray | None",
        "- train_mode: bool", "- mean/std: ndarray",
    ], ["+ __len__()", "+ __getitem__(i)", "- _load_and_process(i)"])

    tf, tf_b = uml(8.8, 7.35, 3.0, "Transforms", [
        "+ roi_margin: float", "+ clahe_clip: float",
    ], ["+ crop_roi(img, box)", "+ apply_clahe(img)",
        "+ augment(img, rng)", "+ normalise(img)"], header=ORANGE)

    model, model_b = uml(0.3, 4.05, 3.3, "TrafficSignNet", [
        "+ features: Sequential", "+ head: Sequential",
        "+ num_classes: int = 43",
    ], ["+ forward(x)", "+ last_conv_layer()"], header=GREEN)

    eng, eng_b = uml(4.35, 4.05, 3.7, "Engine", [
        "- model: TrafficSignNet", "- optimiser: AdamW",
        "- scheduler: OneCycleLR", "- criterion: CrossEntropy",
    ], ["+ train_one_epoch(loader)", "+ validate(loader)"])

    cam, cam_b = uml(8.8, 4.05, 3.0, "GradCAM", [
        "- model: TrafficSignNet", "- activations: Tensor",
        "- gradients: Tensor",
    ], ["+ __call__(x, target)", "- _hook_forward()", "+ remove_hooks()"],
        header=GREEN)

    split, split_b = uml(2.2, 1.35, 4.2, "TrackAwareSplitter", [
        "+ val_frac: float", "+ seed: int",
        "- groups: dict[(cls, track), list]",
    ], ["+ split(records)", "- _assert_disjoint()"], header=RED)

    ckpt, ckpt_b = uml(7.4, 1.35, 4.4, "Checkpoint", [
        "+ state_dict: dict", "+ config: Config",
        "+ class_names: list[str]", "+ mean/std: ndarray",
    ], ["+ save(path)", "+ load(path)"], header=GREY)

    def edge(a, b, label, *, rad=0.0, style="->"):
        _arrow(ax, a, b, style=style, color="#55636f", lw=1.1, label=label,
               fontsize=6.9, rad=rad)

    edge((4.35, 6.4), (3.6, 6.4), "uses")           # dataset -> config
    edge((8.05, 6.4), (8.8, 6.4), "applies")        # dataset -> transforms
    edge((6.2, 5.55), (6.2, 4.05), "batches")       # dataset -> engine
    edge((4.35, 3.3), (3.6, 3.3), "optimises")      # engine -> model
    edge((8.05, 3.3), (8.8, 3.3), "hooks")          # gradcam -> model
    # Routed through the gutters between boxes: a straight centre-to-centre
    # line would cross Engine and TrackAwareSplitter respectively. The
    # coordinates below are the real gaps — Engine ends at y=1.94, the bottom
    # row starts at y=1.35, and nothing extends below y=-0.76.
    edge((7.6, 1.94), (8.6, 1.35), "writes")        # engine -> checkpoint
    edge((3.95, 1.35), (3.95, 5.00), "produces")    # splitter -> dataset
    edge((11.4, -0.95), (1.95, -0.95), None)        # checkpoint -> model
    edge((1.95, -0.95), (1.95, 2.17), "restores")

    ax.set_xlim(0, 12.2)
    ax.set_ylim(-1.45, 7.75)
    ax.axis("off")
    ax.set_title("Class / component diagram", fontweight="bold", fontsize=13,
                 pad=8)
    fig.tight_layout()
    return _save(fig, out)


# --------------------------------------------------------------------------
# 5. Sequence diagram
# --------------------------------------------------------------------------

def fig_sequence(out: Path) -> Path:
    """One training epoch, as messages between the participating objects."""
    fig, ax = plt.subplots(figsize=(12.0, 7.4))

    actors = [
        ("User", 0.9, GREY),
        ("cli.py", 2.85, BLUE),
        ("train.py", 4.8, BLUE),
        ("GTSRBDataset", 6.9, ORANGE),
        ("Engine", 9.0, BLUE),
        ("TrafficSignNet", 11.0, GREEN),
    ]
    top, bottom = 6.75, 0.45
    for name, x, color in actors:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.82, top), 1.64, 0.5, boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="black", lw=0.9))
        ax.text(x, top + 0.25, name, ha="center", va="center", fontsize=8.2,
                color="white", fontweight="bold")
        ax.plot([x, x], [bottom, top], color="#9aa7b4", lw=1.0,
                linestyle=(0, (4, 4)))

    X = {name: x for name, x, _ in actors}

    def msg(y, a, b, label, *, ret=False):
        x1, x2 = X[a], X[b]
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(
                        arrowstyle="->", lw=1.15,
                        color="#6f7b86" if ret else "#33475b",
                        linestyle=(0, (4, 3)) if ret else "-"))
        ax.text((x1 + x2) / 2, y + 0.13, label, ha="center", va="bottom",
                fontsize=7.4, color="#33475b" if not ret else "#6f7b86")

    msg(6.45, "User", "cli.py", "python -m src.cli train")
    msg(6.05, "cli.py", "train.py", "run(config)")
    msg(5.65, "train.py", "GTSRBDataset", "load manifests + cache to RAM")
    msg(5.25, "GTSRBDataset", "train.py", "train / val loaders", ret=True)

    # Epoch loop frame.
    ax.add_patch(mpatches.Rectangle((0.25, 1.55), 11.6, 3.35, fill=False,
                                    edgecolor="#8a6a3a", lw=1.1))
    ax.add_patch(mpatches.Rectangle((0.25, 4.62), 1.5, 0.28,
                                    facecolor="#8a6a3a", edgecolor="none"))
    ax.text(1.0, 4.76, "loop [epoch]", ha="center", va="center", fontsize=7.4,
            color="white", fontweight="bold")

    msg(4.35, "train.py", "Engine", "train_one_epoch(loader)")
    msg(3.98, "Engine", "GTSRBDataset", "__getitem__(i) → augment")
    msg(3.61, "GTSRBDataset", "Engine", "tensor batch", ret=True)
    msg(3.24, "Engine", "TrafficSignNet", "forward(x)")
    msg(2.87, "TrafficSignNet", "Engine", "logits", ret=True)
    msg(2.50, "Engine", "TrafficSignNet", "backward() + optimiser.step()")
    msg(2.13, "train.py", "Engine", "validate(val_loader)")
    msg(1.76, "Engine", "train.py", "val accuracy", ret=True)

    # Conditional checkpoint.
    ax.add_patch(mpatches.Rectangle((0.25, 0.75), 11.6, 0.62, fill=False,
                                    edgecolor=GREEN, lw=1.1))
    ax.add_patch(mpatches.Rectangle((0.25, 1.09), 1.95, 0.28,
                                    facecolor=GREEN, edgecolor="none"))
    ax.text(1.22, 1.23, "alt [improved]", ha="center", va="center",
            fontsize=7.4, color="white", fontweight="bold")
    msg(0.92, "train.py", "TrafficSignNet", "save best.pt (weights + config)")

    ax.set_xlim(0, 12.3)
    ax.set_ylim(0.25, 7.55)
    ax.axis("off")
    ax.set_title("Sequence diagram — one training epoch", fontweight="bold",
                 fontsize=13, pad=8)
    fig.tight_layout()
    return _save(fig, out)


def main() -> int:
    out_dir = ensure_dir(RESULTS_DIR / "design")

    print("[design-figures] writing")
    fig_system_architecture(out_dir / "system_architecture.png")
    fig_workflow(out_dir / "workflow.png")
    fig_use_case(out_dir / "use_case.png")
    fig_class_diagram(out_dir / "class_diagram.png")
    fig_sequence(out_dir / "sequence_training.png")
    print("[design-figures] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
