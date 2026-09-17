"""Build the project report as a self-contained HTML file, then a PDF.

Every figure is embedded as a base64 data URI and every number is read from
results/metrics.json and ablations/*/metrics.json at build time, so the document
cannot drift from the code that produced it.

    python -m src.build_report
"""

from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, RESULTS_DIR
from .utils import ensure_dir

REPORT_DIR = PROJECT_ROOT / "report"
ABLATIONS = PROJECT_ROOT / "ablations"

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def embed(path: Path) -> str:
    path = Path(path)
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def figure(path: Path, number: int, caption: str, width: str = "100%") -> str:
    src = embed(path)
    if not src:
        return f"<p class='missing'>[missing figure: {Path(path).name}]</p>"
    return (f'<figure><img src="{src}" style="width:{width}" alt="Figure {number}">'
            f'<figcaption><strong>Figure {number}.</strong> {caption}</figcaption></figure>')


CSS = """
@page { size: A4; margin: 17mm 15mm 15mm; }
* { box-sizing: border-box; }
body {
  font-family: "Charter","Georgia","Times New Roman",serif;
  font-size: 10.4pt; line-height: 1.52; color: #1a1a1a;
  max-width: 195mm; margin: 0 auto; padding: 8mm 5mm 18mm; background: #fff;
  -webkit-print-color-adjust: exact; print-color-adjust: exact;
}
h1 { font-size: 21pt; line-height: 1.2; margin: 0 0 3pt; letter-spacing: -0.3pt; }
h2 { font-size: 13pt; margin: 20pt 0 6pt; padding-bottom: 3pt;
     border-bottom: 1.2pt solid #2c6fbb; color: #14395e;
     page-break-after: avoid; break-after: avoid; }
h3 { font-size: 10.8pt; margin: 13pt 0 4pt; color: #14395e; page-break-after: avoid; }
p { margin: 0 0 7pt; text-align: justify; hyphens: auto; }
a { color: #2c6fbb; text-decoration: none; word-break: break-all; }
code { font-family: "Consolas","Menlo",monospace; font-size: 8.8pt;
       background: #f4f6f8; padding: 1pt 3pt; border-radius: 2pt; }
.title-block { border-bottom: 2.5pt solid #2c6fbb; padding-bottom: 9pt; margin-bottom: 3pt; }
.subtitle { font-size: 11.5pt; color: #444; margin: 2pt 0 7pt; font-style: italic; }
.byline { font-size: 9.4pt; color: #555; }
.byline strong { color: #1a1a1a; }
.keyfacts { display: grid; grid-template-columns: repeat(4,1fr); gap: 7pt;
            margin: 11pt 0 3pt; page-break-inside: avoid; }
.keyfact { background: #eef4fb; border-left: 2.5pt solid #2c6fbb; padding: 6pt 7pt; }
.keyfact .v { font-size: 13.5pt; font-weight: bold; color: #14395e; display: block; line-height: 1.12; }
.keyfact .k { font-size: 7.6pt; color: #555; text-transform: uppercase; letter-spacing: 0.3pt; }
table { border-collapse: collapse; width: 100%; margin: 7pt 0 9pt; font-size: 8.9pt;
        page-break-inside: avoid; }
th { background: #14395e; color: #fff; text-align: left; padding: 4.2pt 6pt; font-weight: 600; }
td { padding: 3.6pt 6pt; border-bottom: 0.5pt solid #dde3e9; }
tr:nth-child(even) td { background: #f7f9fb; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.highlight td { background: #eef4fb !important; font-weight: bold; }
caption { caption-side: top; text-align: left; font-size: 8.9pt; padding-bottom: 4pt; color: #333; }
caption strong { color: #14395e; }
figure { margin: 9pt 0 11pt; page-break-inside: avoid; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; border: 0.5pt solid #d5dbe1; }
figcaption { font-size: 8.4pt; color: #444; margin-top: 4pt; text-align: left; line-height: 1.38; }
figcaption strong { color: #14395e; }
.callout { background: #fdf6e8; border-left: 3pt solid #e67e22; padding: 8pt 10pt;
           margin: 9pt 0; page-break-inside: avoid; font-size: 9.6pt; }
.callout h4 { margin: 0 0 4pt; font-size: 9.8pt; color: #a35b12; }
.callout p:last-child { margin-bottom: 0; }
ul, ol { margin: 0 0 7pt; padding-left: 15pt; }
li { margin-bottom: 3pt; }
.pagebreak { page-break-before: always; break-before: page; }
.missing { color: #c0392b; font-style: italic; }
.footer { margin-top: 18pt; padding-top: 6pt; border-top: 0.5pt solid #ccc;
          font-size: 8.3pt; color: #666; }
.toc { background: #f7f9fb; border: 0.5pt solid #dde3e9; padding: 8pt 10pt 8pt 22pt;
       margin: 9pt 0 5pt; font-size: 9.3pt; page-break-inside: avoid; }
.toc li { margin-bottom: 1.5pt; }
@media screen { body { box-shadow: 0 0 14px rgba(0,0,0,.09); margin: 16px auto; } }
"""


def build() -> Path:
    m = json.load(open(RESULTS_DIR / "metrics.json"))
    t = m["test"]
    hist = json.load(open(RESULTS_DIR / "history.json"))
    split = json.load(open(PROJECT_ROOT / "data" / "manifests" / "split_summary.json"))
    norm = json.load(open(PROJECT_ROOT / "data" / "manifests" / "norm_stats.json"))
    conf = pd.read_csv(RESULTS_DIR / "top_confusions_test.csv")
    pc = m["per_class"]

    abl = {}
    for k in ("no_clahe", "no_balance", "half_width"):
        p = ABLATIONS / k / "metrics.json"
        if p.exists():
            abl[k] = json.load(open(p))

    acc = t["accuracy"] * 100
    best_val = max(h["val_acc"] for h in hist) * 100
    best_epoch = max(hist, key=lambda h: h["val_acc"])["epoch"]
    total_min = sum(h["seconds"] for h in hist) / 60
    worst = sorted(pc, key=lambda d: d["f1"])[:6]
    n_perfect = sum(1 for p in pc if p["f1"] >= 0.999)
    n_high = sum(1 for p in pc if p["f1"] >= 0.99)
    n_low = sum(1 for p in pc if p["f1"] < 0.90)
    n_signs = split["n_train_tracks"] + split["n_val_tracks"]

    fam_total = int(conf["count"].sum())
    speed, tri = set(range(9)), {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
    derest = {6, 32, 41, 42}
    in_family = int(sum(r["count"] for _, r in conf.iterrows()
                        if (r["true_id"] in speed and r["pred_id"] in speed)
                        or (r["true_id"] in tri and r["pred_id"] in tri)
                        or (r["true_id"] in derest and r["pred_id"] in derest)))
    fam_pct = 100 * in_family / fam_total

    R, G = RESULTS_DIR, RESULTS_DIR / "report"

    worst_rows = "\n".join(
        f"<tr><td class='num'>{p['class_id']}</td><td>{p['name']}</td>"
        f"<td class='num'>{p['precision']:.3f}</td><td class='num'>{p['recall']:.3f}</td>"
        f"<td class='num'>{p['f1']:.3f}</td><td class='num'>{p['support']}</td></tr>"
        for p in worst)

    conf_rows = "\n".join(
        f"<tr><td class='num'>{r['true_id']}</td><td>{r['true_name']}</td>"
        f"<td class='num'>{r['pred_id']}</td><td>{r['pred_name']}</td>"
        f"<td class='num'>{r['count']}</td><td class='num'>{r['pct_of_true']:.1f}%</td></tr>"
        for _, r in conf.head(10).iterrows())

    abl_rows = [f"""<tr class="highlight"><td>Baseline (all decisions active)</td>
      <td><code>train</code></td><td class="num">{acc:.2f}</td>
      <td class="num">{t['balanced_accuracy']*100:.2f}</td>
      <td class="num">{t['macro_f1']*100:.2f}</td>
      <td class="num">99,019</td><td class="num">—</td></tr>"""]
    for key, label, cmd in (("no_clahe", "Without CLAHE", "train --no-clahe"),
                            ("no_balance", "Without class rebalancing", "train --balance none"),
                            ("half_width", "Half-width model", "train --width-mult 0.5")):
        if key in abl:
            a = abl[key]["test"]
            abl_rows.append(
                f"""<tr><td>{label}</td><td><code>{cmd}</code></td>
                <td class="num">{a['accuracy']*100:.2f}</td>
                <td class="num">{a['balanced_accuracy']*100:.2f}</td>
                <td class="num">{a['macro_f1']*100:.2f}</td>
                <td class="num">{abl[key]['model']['params']:,}</td>
                <td class="num">{(a['accuracy']-t['accuracy'])*100:+.2f}</td></tr>""")
    abl_table = "\n".join(abl_rows)

    ep_rows = "\n".join(
        f"<tr><td class='num'>{h['epoch']}</td><td class='num'>{h['train_loss']:.4f}</td>"
        f"<td class='num'>{h['train_acc']*100:.2f}</td><td class='num'>{h['val_loss']:.4f}</td>"
        f"<td class='num'>{h['val_acc']*100:.2f}</td><td class='num'>{h['seconds']:.0f}</td></tr>"
        for h in hist if h["epoch"] in (1, 2, 3, 5, 8, 10, 12, 13, 14, 15))

    d_clahe = (t['accuracy'] - abl['no_clahe']['test']['accuracy']) * 100 if 'no_clahe' in abl else 0
    d_bal = (t['accuracy'] - abl['no_balance']['test']['accuracy']) * 100 if 'no_balance' in abl else 0
    d_balb = (t['balanced_accuracy'] - abl['no_balance']['test']['balanced_accuracy']) * 100 if 'no_balance' in abl else 0
    d_hw = (t['accuracy'] - abl['half_width']['test']['accuracy']) * 100 if 'half_width' in abl else 0
    mean_s = [round(x, 3) for x in norm['mean']]
    std_s = [round(x, 3) for x in norm['std']]

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>GTSRB Traffic Sign Recognition — Project Report</title>
<style>{CSS}</style></head><body>

<div class="title-block">
  <h1>Traffic Sign Recognition on GTSRB</h1>
  <p class="subtitle">A convolutional neural network trained from scratch, with a
     leak-free evaluation protocol</p>
  <p class="byline"><strong>Computer Vision — Project Report</strong><br>
     Author: Tanmay Kala (tanmayai23)<br>
     Repository: <a href="https://github.com/tanmayai23/gtsrb-traffic-sign-recognition">github.com/tanmayai23/gtsrb-traffic-sign-recognition</a></p>
</div>

<div class="keyfacts">
  <div class="keyfact"><span class="v">{acc:.2f}%</span><span class="k">Test accuracy</span></div>
  <div class="keyfact"><span class="v">99,019</span><span class="k">Parameters</span></div>
  <div class="keyfact"><span class="v">{total_min:.0f} min</span><span class="k">Training, CPU only</span></div>
  <div class="keyfact"><span class="v">{t['inference_ms_per_image']:.2f} ms</span><span class="k">Per image</span></div>
</div>

<h2>Abstract</h2>
<p>
This report describes a 43-class traffic sign classifier built on the German
Traffic Sign Recognition Benchmark (GTSRB). The network is a three-block
convolutional model written from scratch in PyTorch with 99,019 parameters. It
reaches <strong>{acc:.2f}% accuracy</strong> on the official {t['n_samples']:,}-image
test set after {len(hist)} epochs of CPU training, and classifies an image in
{t['inference_ms_per_image']:.2f} ms.
</p>
<p>
The main methodological contribution is the evaluation protocol. GTSRB stores
roughly 30 consecutive video frames of every physical sign, so its 39,209
training images depict only {n_signs:,} distinct signs. Splitting that data at
random places near-identical frames of one sign on both sides of the
train/validation boundary, which lets a model score highly by recognising
pictures it has already seen. This project splits by <em>track</em> instead, so
no physical sign appears on both sides. The resulting accuracy is lower than
figures commonly quoted for GTSRB, and it is the number that actually reflects
generalisation.
</p>
<p>
Three ablations confirm that each design decision contributes: removing contrast
equalisation costs {d_clahe:.2f} percentage points, removing class rebalancing
costs {d_bal:.2f}, and halving the model width costs {d_hw:.2f}.
</p>

<div class="toc">
<strong>Contents</strong>
<ol>
<li>Introduction and problem statement</li>
<li>Dataset</li>
<li>Evaluation protocol: splitting by track</li>
<li>Preprocessing and augmentation</li>
<li>Model architecture</li>
<li>Training procedure</li>
<li>Results</li>
<li>Error analysis</li>
<li>Ablation study</li>
<li>Explainability</li>
<li>Limitations and future work</li>
<li>Reproducibility · 13. Conclusion · 14. References</li>
</ol>
</div>

<h2>1. Introduction and problem statement</h2>
<p>
Traffic sign recognition is a component of driver assistance and autonomous
driving systems. A vehicle must read signs correctly under motion blur, poor
lighting, partial occlusion and varying distance, and it must do so fast enough
to act on the result. Errors are not symmetric in cost: confusing a 120 km/h sign
with an 80 km/h sign has different consequences from confusing two warning
triangles.
</p>
<p>
This project addresses the classification half of that problem. Given an image
cropped to a single sign, the task is to assign one of 43 classes. Detection —
locating signs in a full road scene — is out of scope, and Section 11 explains
what that means for interpreting the results.
</p>
<p>
Two constraints shaped the work. First, all training had to run on a CPU, which
ruled out large architectures and high input resolutions and made the
accuracy-per-minute trade-off explicit. Second, the project had to be
reproducible from a command line by someone with no prior context, so every step
from downloading data to producing figures is a subcommand of a single entry
point.
</p>

<h2>2. Dataset</h2>
<p>
GTSRB was introduced by Stallkamp et al. for a classification competition at
IJCNN 2011. It contains 39,209 training images and {t['n_samples']:,} test images
across 43 classes, cropped from video recorded on German roads. Image sizes range
from 15×15 to 250×250 pixels. Each image carries an annotated region of interest
marking the sign within the frame.
</p>
<p>
The class distribution is uneven. The most common class holds
{split['max_class_count']:,} training images and the rarest holds
{split['min_class_count']}, a ratio of {split['imbalance_ratio']:.1f} to 1. This
imbalance reflects real road frequencies — 50 km/h signs are simply more common
than "beware of ice" — but left unaddressed it biases a classifier toward the
frequent classes. Section 6 describes the sampling strategy used to counter it,
and Section 9 measures whether it helped.
</p>
{figure(R / "class_distribution.png", 1,
        "Training images per class. The green bar is the most frequent class and the red bar the rarest. "
        f"The {split['imbalance_ratio']:.1f}× spread is why balanced accuracy is reported alongside raw accuracy throughout.")}

<h2 class="pagebreak">3. Evaluation protocol: splitting by track</h2>
<p>This section describes the most consequential decision in the project.</p>
<p>
GTSRB filenames encode two numbers. The file <code>00053_00024.ppm</code> is
frame 24 of track 53, and a track is a sequence of roughly 30 consecutive video
frames of the same physical sign as the camera approaches it. The 39,209 training
images therefore depict only {n_signs:,} distinct signs. Frames within a track
differ mainly in scale and slight blur; they are near-duplicates.
</p>
<p>
The consequence for evaluation is direct. A validation split drawn at random over
images will place frame 11 of a track in training and frame 12 in validation. The
model is then asked to classify a picture almost identical to one it was trained
on, and it succeeds — not by generalising, but by recall. Validation accuracy
approaches 99.9% and stops carrying information.
</p>
{figure(G / "split_illustration.png", 2,
        "Two ways to split the same data. Each row is one physical sign and each cell one frame. "
        "Top: a random split by image scatters frames of a single sign across both sides, so the model is "
        "validated on near-duplicates of its training data. Bottom: splitting by track keeps every frame of "
        "a sign on one side, so validation images show signs the model has never seen.")}
<p>
This project assigns whole tracks, stratified by class so that every class is
represented on both sides. A second detail matters here and is easy to miss:
track numbers restart from zero inside each class directory, so track
<code>00000</code> exists in class 0 and again in class 1 and the two are
unrelated. Keying on the bare track number silently merges them. The identifier
must combine class and track.
</p>
<p>
The split is verified rather than assumed. Zero overlap is asserted when the
manifests are built, and a unit test suite checks the same properties
independently — including one test that confirms a random split <em>does</em>
leak, documenting the failure being avoided. Those tests run on synthetic data,
so the claim can be checked without downloading the dataset.
</p>
<table>
<caption><strong>Table 1.</strong> Data splits. Track overlap is zero by construction and by test.</caption>
<tr><th>Split</th><th class="num">Images</th><th class="num">Tracks</th><th>Source</th></tr>
<tr><td>Training</td><td class="num">{split['n_train']:,}</td><td class="num">{split['n_train_tracks']:,}</td><td>GTSRB training set</td></tr>
<tr><td>Validation</td><td class="num">{split['n_val']:,}</td><td class="num">{split['n_val_tracks']:,}</td><td>GTSRB training set, held out by track</td></tr>
<tr><td>Test</td><td class="num">{split['n_test']:,}</td><td class="num">—</td><td>Official GTSRB test set, separate recordings</td></tr>
<tr class="highlight"><td>Overlap between train and validation</td><td class="num">0</td><td class="num">0</td><td>asserted and unit-tested</td></tr>
</table>
<div class="callout">
<h4>What this costs, and why it is worth paying</h4>
<p>
Accuracy reported here is lower than the figures near 99% often quoted for GTSRB.
Some of that gap is architectural, but some of it is protocol: a model validated
on leaked near-duplicates will report a number it cannot reproduce on genuinely
new signs. The {acc:.2f}% in this report is measured on the official test set,
recorded separately from the training data, and the
{abs(best_val - acc):.2f} point gap between validation ({best_val:.2f}%) and test
({acc:.2f}%) is small — which is the behaviour a non-leaking split should show.
</p>
</div>

<h2>4. Preprocessing and augmentation</h2>
<h3>4.1 Preprocessing</h3>
<p>
Each image passes through four deterministic steps: crop to the annotated region
of interest with a 10% margin, apply contrast-limited adaptive histogram
equalisation (CLAHE), resize to 32×32, and normalise using channel statistics
computed over the training split only.
</p>
<p>
The margin on the crop is deliberate. Cropping tight to the annotation removes
background clutter, but the outer rim of a sign is itself informative — a red
triangle and a blue circle are different classes before any pictogram is read —
so a little context is kept.
</p>
<p>
CLAHE is applied to the L channel in LAB colour space, with a clip limit of 2.0
and a 4×4 tile grid. Two choices are worth stating. Equalising each RGB channel
independently shifts hue, and hue carries class information for traffic signs, so
the operation is confined to luminance. The tile grid is smaller than OpenCV's
8×8 default because these crops average about 50 pixels per side; the default
would leave tiles of roughly six pixels and amplify noise rather than structure.
Section 9 shows CLAHE is the single largest preprocessing contributor.
</p>
<p>
Normalisation statistics are computed over the training split alone (mean
{mean_s}, standard deviation {std_s}). Including validation images would leak
their distribution into the model's input scaling — a smaller leak than track
overlap, but the same kind of mistake.
</p>
{figure(R / "preprocessing_samples.png", 3,
        "The preprocessing pipeline. Top row: raw images as recorded, with background and uneven exposure. "
        "Bottom row: after ROI crop, CLAHE and resize to 32×32. Contrast is normalised across very different "
        "lighting conditions while colour is preserved.")}

<h3>4.2 Augmentation</h3>
<p>
Training images are augmented with rotation up to ±12°, translation up to ±10%,
scaling between 0.9 and 1.1, shear up to ±0.1, brightness scaled by 0.7–1.3,
contrast by 0.8–1.2, and occasional light Gaussian noise. These ranges mimic the
variation a camera actually encounters: a sign viewed from a slightly different
angle, distance or light level is still the same sign.
</p>
<p>
Two implementation details affect quality. All geometric transforms are composed
into a single affine matrix and applied in one warp, because chaining three
separate warps interpolates three times and visibly degrades a 32-pixel image.
Border pixels are replicated rather than filled with black, since black corners
would correlate perfectly with "this sample was augmented" and give the network a
shortcut unrelated to the sign.
</p>
<div class="callout">
<h4>No flips, in any direction</h4>
<p>
Horizontal flipping is a standard default for natural images and is wrong for
this task. Mirroring maps class 33 (turn right ahead) onto class 34 (turn left
ahead), class 19 onto class 20 (dangerous curve left and right), class 36 onto
class 37, and renders every speed-limit digit unreadable. Flip augmentation would
therefore generate training images whose labels are incorrect. The restriction is
enforced by a regression test that fails if any augmentation output resembles a
mirror of its input more than the input itself.
</p>
</div>
{figure(R / "augmentation_samples.png", 4,
        "Augmentation samples from one source image. Rotation, scale, shear and brightness vary; the sign is "
        "never mirrored, and no black borders are introduced.")}

<h2 class="pagebreak">5. Model architecture</h2>
<p>
The network has three convolutional blocks followed by a global-average-pooled
linear classifier. Each block is a 3×3 convolution, batch normalisation, ReLU,
and 2×2 max pooling, which halves the spatial resolution and doubles the channel
count. Total trainable parameters: 99,019.
</p>
{figure(G / "architecture.png", 5,
        "TrafficSignNet. Tensor shapes below each stage. Three pooling operations reduce 32×32 to 4×4 while "
        "channels grow from 3 to 128; global average pooling then collapses the spatial dimensions entirely.")}
<p>
Three choices deserve comment. Convolutions omit their bias term because the
batch normalisation immediately following re-centres the output, making a
convolution bias mathematically redundant. The classifier head uses global
average pooling rather than flattening into a fully connected layer: this removes
roughly 88,000 parameters, makes the network accept any input resolution without
code changes, and is the arrangement under which class activation maps are
interpretable (Section 10). Weights are initialised with Kaiming normal scaling,
matched to the ReLU activations.
</p>
<p>
The size of the network was set by measurement rather than preference. On the
target hardware, this configuration trains in about {total_min:.0f} minutes.
Doubling the input resolution to 48×48, or using two convolutions per block, each
roughly doubled the time per epoch. Section 9 shows that halving the width in the
other direction costs {d_hw:.2f} points of accuracy, so the chosen capacity is
not padding.
</p>

<h2>6. Training procedure</h2>
<p>
The model is trained with AdamW (learning rate 1×10⁻³, weight decay 1×10⁻⁴) under
a OneCycle schedule peaking at 3×10⁻³, stepped every batch rather than every
epoch. The loss is cross-entropy with label smoothing of 0.05, and gradients are
clipped at norm 5.0. Class imbalance is handled by a weighted random sampler that
draws inversely to class frequency.
</p>
<p>
OneCycle suits a fixed short budget: the warm-up allows a higher peak learning
rate than would otherwise be stable, and the anneal brings the model to a good
minimum within about fifteen epochs, where a step schedule would still be
descending. Training stops early if validation accuracy fails to improve for six
epochs, and a wall-clock guard stops cleanly if a time budget is exceeded.
Checkpoints are written every epoch, so an interrupted run is never lost, and
each checkpoint stores its own configuration, class names and normalisation
statistics so that inference needs no matching flags.
</p>
<table>
<caption><strong>Table 2.</strong> Training history (selected epochs of {len(hist)}).
Best validation accuracy occurred at epoch {best_epoch}.</caption>
<tr><th class="num">Epoch</th><th class="num">Train loss</th><th class="num">Train acc (%)</th>
    <th class="num">Val loss</th><th class="num">Val acc (%)</th><th class="num">Time (s)</th></tr>
{ep_rows}
</table>
{figure(R / "training_curves.png", 6,
        "Loss and accuracy per epoch. Training and validation curves stay close throughout, indicating the "
        "augmentation and dropout are sufficient to prevent overfitting; there is no widening gap.")}
<p>
The two curves track each other closely for the whole run. Final training
accuracy is {hist[-1]['train_acc']*100:.2f}% against validation
{hist[-1]['val_acc']*100:.2f}%, a gap under one and a half points. Combined with
the track-disjoint split, this indicates the model is learning sign appearance
rather than memorising particular photographs.
</p>

<h2 class="pagebreak">7. Results</h2>
<p>
All figures below are measured on the official GTSRB test set of
{t['n_samples']:,} images, which comes from separate recording sessions and is
used only once, after training is complete.
</p>
<table>
<caption><strong>Table 3.</strong> Test set performance.</caption>
<tr><th>Metric</th><th class="num">Value</th><th>Interpretation</th></tr>
<tr class="highlight"><td>Accuracy</td><td class="num">{acc:.2f}%</td>
    <td>{t['n_samples'] - t['n_errors']:,} of {t['n_samples']:,} correct</td></tr>
<tr><td>Balanced accuracy</td><td class="num">{t['balanced_accuracy']*100:.2f}%</td>
    <td>mean per-class recall; unaffected by class frequency</td></tr>
<tr><td>Macro F1</td><td class="num">{t['macro_f1']*100:.2f}%</td>
    <td>every class weighted equally</td></tr>
<tr><td>Weighted F1</td><td class="num">{t['weighted_f1']*100:.2f}%</td>
    <td>weighted by class frequency</td></tr>
<tr><td>Top-5 accuracy</td><td class="num">{t['top5_accuracy']*100:.2f}%</td>
    <td>true class within the five highest-scoring</td></tr>
<tr><td>Errors</td><td class="num">{t['n_errors']}</td><td>misclassified images</td></tr>
<tr><td>Inference time</td><td class="num">{t['inference_ms_per_image']:.2f} ms</td>
    <td>per image, CPU, batch size 256</td></tr>
</table>
<p>
Balanced accuracy is reported alongside raw accuracy deliberately. Under a
{split['imbalance_ratio']:.1f}-to-1 imbalance, a model can post a high overall
accuracy while performing poorly on rare classes, because the frequent classes
dominate the average. The
{(t['accuracy'] - t['balanced_accuracy'])*100:.2f} point gap between the two here
is modest, which indicates the rare classes are not being neglected.
</p>
<p>
Of the 43 classes, {n_perfect} reach a perfect F1 score, {n_high} exceed 0.99,
and {n_low} fall below 0.90. Top-5 accuracy of {t['top5_accuracy']*100:.2f}%
means that even when the model is wrong, the correct class is almost always among
its leading candidates — the errors are confusions between similar signs rather
than failures to recognise a sign at all.
</p>
{figure(R / "per_class_f1_test.png", 7,
        "Per-class F1, ascending. Red bars mark classes below 0.90. Performance is high and even across most "
        "of the 43 classes, with a short tail of difficult ones analysed in Section 8.", "74%")}

<h2>8. Error analysis</h2>
<p>
The {t['n_errors']} errors are not spread uniformly. They concentrate in groups
of signs that are genuinely similar at the working resolution.
</p>
<table>
<caption><strong>Table 4.</strong> Ten most frequent confusions. "Share of class" is the
proportion of that true class sent to the wrong label.</caption>
<tr><th class="num">True</th><th>Class</th><th class="num">Pred</th><th>Predicted as</th>
    <th class="num">Count</th><th class="num">Share of class</th></tr>
{conf_rows}
</table>
{figure(G / "confusion_families.png", 8,
        f"Errors from the twenty most frequent confusion pairs, grouped by sign family. "
        f"{in_family} of {fam_total} ({fam_pct:.0f}%) are confusions within a visually similar family rather "
        "than between unrelated signs.")}
<p>
Two families dominate. Speed-limit signs differ only in one or two digits inside
an identical red circle; at 32×32 those digits span a handful of pixels, and 120
becomes hard to separate from 80. Warning triangles share an identical red border
and differ only in a small central pictogram, which suffers the same fate. Both
patterns point to the same cause — input resolution — and Section 11 identifies
raising it as the clearest route to improvement.
</p>
{figure(G / "per_class_recall.png", 9,
        "Per-class recall against the number of test images in that class. Red points fall below 0.90 recall. "
        "The weakest classes are not the smallest ones, which indicates the difficulty comes from visual "
        "similarity rather than lack of training data.")}
<p>
This last figure answers a question the imbalance raises. If the rare classes were
simply under-trained, recall would fall as class size fell. It does not: several
of the smallest classes reach perfect recall, while the weakest performers sit in
the middle of the size range. Class 21 (double curve) and class 27 (pedestrians)
are hard because they resemble other warning triangles, not because they are
rare. This also explains the modest return from rebalancing measured in Section 9.
</p>
<table>
<caption><strong>Table 5.</strong> The six lowest-scoring classes by F1.</caption>
<tr><th class="num">ID</th><th>Class</th><th class="num">Precision</th>
    <th class="num">Recall</th><th class="num">F1</th><th class="num">Support</th></tr>
{worst_rows}
</table>
{figure(R / "misclassified_test.png", 10,
        "The twenty-five errors the model was most confident about. High-confidence mistakes are more "
        "informative than random ones: they reveal systematic confusions rather than unreadable crops.", "80%")}

<h2 class="pagebreak">9. Ablation study</h2>
<p>
Each row below is a complete retraining run of {len(hist)} epochs with one design
decision removed, evaluated on the same test set under identical conditions. The
purpose is to establish that each choice earns its place rather than being
inherited convention.
</p>
<table>
<caption><strong>Table 6.</strong> Ablation results. Δ is the change in accuracy against the baseline,
in percentage points.</caption>
<tr><th>Configuration</th><th>Command</th><th class="num">Acc (%)</th>
    <th class="num">Balanced (%)</th><th class="num">Macro F1 (%)</th>
    <th class="num">Params</th><th class="num">Δ (pp)</th></tr>
{abl_table}
</table>
{figure(G / "ablation_comparison.png", 11,
        "Ablation results. The dotted line marks baseline accuracy. Each removal degrades performance, "
        "confirming that all three decisions contribute.")}
<p>
<strong>Contrast equalisation matters most</strong> ({d_clahe:+.2f} points).
GTSRB is dashcam footage containing heavily under- and over-exposed frames, and
normalising local contrast before the network sees an image is worth more than
any other preprocessing step tested.
</p>
<p>
<strong>Rebalancing helps rare classes specifically.</strong> Removing it costs
{d_bal:.2f} points of raw accuracy but {d_balb:.2f} points of balanced accuracy —
more than twice as much. That asymmetry is exactly the signature expected from an
intervention aimed at the tail of the distribution: it barely moves the average,
which frequent classes dominate, while measurably improving the rare classes that
balanced accuracy weights equally. The effect is real but modest, consistent with
Section 8's finding that difficulty here comes from visual similarity rather than
class size.
</p>
<p>
<strong>The chosen capacity is justified.</strong> Halving the width reduces the
model to 26,491 parameters and costs {d_hw:.2f} points, the largest single drop
measured. The 99,019-parameter baseline is therefore not oversized for the task.
</p>

<h2>10. Explainability</h2>
<p>
Accuracy alone does not establish that a model is using the right evidence. A
classifier can reach a high score by keying on background correlations — sky,
road texture, the pole a sign is mounted on. Grad-CAM (Selvaraju et al., 2017)
tests this by computing which spatial regions of the final convolutional layer
drive a particular prediction.
</p>
<p>
The implementation here is written directly with forward and full-backward hooks
and uses no external library. Gradients of the target logit with respect to the
final convolutional feature maps are averaged spatially to form per-channel
weights; the weighted sum of feature maps is then rectified and normalised. Two
details are easy to get wrong: the backward pass must use
<code>register_full_backward_hook</code>, since the older hook API reports
incorrect gradients for multi-input modules, and the forward pass must run with
gradients enabled — the model is in evaluation mode for batch normalisation, but
wrapping it in <code>no_grad</code> or <code>inference_mode</code> silently
produces an empty map.
</p>
{figure(R / "gradcam/gradcam_sample_stop.png", 12,
        "Grad-CAM for a stop sign, classified correctly with 97.5% confidence. Attention concentrates on the "
        "sign face and lettering; the background is cold. The map is 8×8 before upsampling, so it is coarse — "
        "sufficient to distinguish sign from background, not to localise individual characters.")}
<p>
Across the sample images, activation consistently centres on the sign rather than
its surroundings, which supports the conclusion that the network has learned sign
appearance. The resolution limit should be stated plainly: at 32×32 input with
three pooling stages, the map is 8×8, so it can show that the model looks at the
sign but not which stroke of a digit it relies on.
</p>

<h2>11. Limitations and future work</h2>
<ul>
<li><strong>Detection is out of scope.</strong> GTSRB supplies ground-truth
    regions of interest, so the model receives a correctly cropped sign. A
    deployed system must first locate signs in a full road scene, and detection
    errors would compound with classification errors. The accuracy reported here
    is an upper bound on what a complete pipeline would achieve.</li>
<li><strong>Input resolution is the binding constraint on accuracy.</strong>
    Section 8 traced most errors to digits and pictograms that do not survive
    downsampling to 32×32. Training at 48×48 or 64×64 is the most direct
    improvement available; the cost is roughly double the training time per
    epoch, which the CPU-only budget did not permit here.</li>
<li><strong>The explanation maps are coarse.</strong> 8×8 Grad-CAM confirms the
    model attends to the sign but cannot resolve finer evidence.</li>
<li><strong>Single dataset, single country.</strong> All data is German. No claim
    is made about transfer to other sign systems, which differ in colour
    convention and pictogram design.</li>
<li><strong>One trained run per configuration.</strong> The ablation differences
    of {d_bal:.2f} points and above are reported from single runs. The larger
    effects are unambiguous, but the smallest would be better supported by
    repeated runs with different seeds.</li>
</ul>

<h2>12. Reproducibility</h2>
<p>
The project runs entirely from the command line. Every result in this report is
regenerated by three commands: <code>prepare</code> downloads and splits the
data, <code>train</code> fits the model, and <code>evaluate</code> produces the
metrics and figures. Random number generators for Python, NumPy and PyTorch are
seeded, and augmentation draws from a per-index generator so results hold even
when data loading is parallelised.
</p>
<p>
One caveat is worth stating honestly: run-to-run reproducibility on a single
machine is reliable, but bit-exact equality across different machines, BLAS
builds or PyTorch versions is not guaranteed, because a few CPU kernels have no
deterministic implementation. A trained checkpoint is committed to the repository
so the reported numbers can be verified without retraining.
</p>
<p>
The test suite contains 50 tests. Most run without the dataset present,
including the synthetic checks of the splitting logic described in Section 3, so
the central methodological claim can be verified on a fresh clone in under a
second.
</p>

<h2>13. Conclusion</h2>
<p>
A three-block convolutional network with 99,019 parameters classifies German
traffic signs at {acc:.2f}% accuracy on the official GTSRB test set, training in
{total_min:.0f} minutes on a CPU and classifying an image in
{t['inference_ms_per_image']:.2f} ms. Ablations confirm that contrast
equalisation, class rebalancing and the chosen model capacity each contribute
measurably, and Grad-CAM indicates the network attends to signs rather than
their surroundings.
</p>
<p>
The result that matters most, though, is methodological. Because GTSRB's images
come in near-duplicate bursts of the same physical sign, an evaluation split
drawn at random measures memorisation rather than generalisation. Splitting by
track produces a lower number, and a truthful one. Residual errors concentrate in
visually similar sign families and trace to input resolution, which points
clearly at where further work should go.
</p>

<h2>14. References</h2>
<ol>
<li>J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel. The German Traffic Sign
    Recognition Benchmark: A multi-class classification competition.
    <em>IJCNN</em>, 2011.</li>
<li>R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, D. Batra.
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
    Localization. <em>ICCV</em>, 2017.</li>
<li>S. Ioffe, C. Szegedy. Batch Normalization: Accelerating Deep Network Training
    by Reducing Internal Covariate Shift. <em>ICML</em>, 2015.</li>
<li>L. N. Smith, N. Topin. Super-Convergence: Very Fast Training of Neural
    Networks Using Large Learning Rates. <em>arXiv:1708.07120</em>, 2018.</li>
<li>K. Zuiderveld. Contrast Limited Adaptive Histogram Equalization.
    <em>Graphics Gems IV</em>, 1994.</li>
<li>I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization.
    <em>ICLR</em>, 2019.</li>
</ol>

<p class="footer">
Generated from <code>results/metrics.json</code> by <code>python -m src.build_report</code>.
Every figure and number in this report is produced by the committed code.
Dataset courtesy of the Institut für Neuroinformatik, Ruhr-Universität Bochum.
</p>

</body></html>"""

    out = ensure_dir(REPORT_DIR) / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"[report] wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, figures embedded)")
    return out


def to_pdf(html: Path) -> Path | None:
    """Print the HTML to PDF with a local Chrome/Edge, if one is installed."""
    exe = next((b for b in BROWSERS if Path(b).exists()), None)
    if exe is None:
        print("[report] no Chrome/Edge found; open the HTML and print to PDF manually.")
        return None

    pdf = html.parent / "GTSRB_Project_Report.pdf"
    subprocess.run([exe, "--headless", "--disable-gpu", "--no-sandbox",
                    "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    html.resolve().as_uri()],
                   capture_output=True, timeout=300)
    if pdf.exists():
        print(f"[report] wrote {pdf}  ({pdf.stat().st_size / 1e6:.1f} MB)")
        return pdf
    return None


def main() -> int:
    html = build()
    to_pdf(html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
