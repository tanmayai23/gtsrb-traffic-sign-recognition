# Project Statement

**Project title:** GTSRB Traffic Sign Recognition — a track-disjoint evaluation of a
from-scratch CNN

---

## Problem statement

A vehicle that reads traffic signs from a camera has to answer one question
continuously: *which of the 43 regulatory signs am I looking at, if any?* The
inputs are not clean catalogue photographs. They are dashcam frames — motion
blurred, backlit at dusk, washed out at noon, partially occluded, and often only
a few dozen pixels across by the time the sign is far enough away to matter.

The technical problem is therefore multi-class image classification under heavy
class imbalance and severe photometric variation, with a hard constraint that the
result must be **honestly measured**.

That last clause is the real problem this project addresses. The German Traffic
Sign Recognition Benchmark (GTSRB) is widely reported at ~99% accuracy, and a
large share of those numbers are inflated by a subtle data-leakage trap. GTSRB
does not contain 39,209 independent photographs; it contains roughly **30
consecutive video frames of each of 1,307 physical signs**. Splitting the data by
*image* puts frame 11 of a sign into training and frame 12 into validation. Those
two frames are near-duplicates, so the model is being validated on pictures it has
effectively memorised, and the reported score measures memorisation rather than
generalisation.

A model selected against a leaked validation set is tuned for the wrong thing. It
looks excellent in a notebook and degrades on genuinely unseen signs — exactly the
failure mode that matters for a safety-relevant system.

**This project builds a traffic sign classifier and, more importantly, builds the
evaluation protocol that makes its reported accuracy mean something.**

---

## Scope of the project

### In scope

- **Data acquisition and preparation** — resumable download of the official GTSRB
  archives, safe extraction, parsing of the per-image annotations (class, ROI
  bounding box, track number).
- **A track-aware, stratified train/validation split** — whole tracks are assigned
  to exactly one side, with every class represented on both, and the disjointness
  asserted at build time and in the test suite.
- **Preprocessing and augmentation** — ROI cropping, CLAHE contrast equalisation
  in LAB space, resizing, and a geometry/photometry augmentation pipeline designed
  so that it never changes a sign's meaning.
- **A CNN written from scratch** — `TrafficSignNet`, 99,019 parameters, no
  pretrained weights and no transfer learning, trainable on a CPU in ~20 minutes.
- **Training with the usual controls** — OneCycle schedule, class rebalancing,
  label smoothing, gradient clipping, early stopping, and checkpointing.
- **Evaluation on the official held-out test set** — accuracy, balanced accuracy,
  macro/weighted F1, top-5, per-class metrics, confusion matrix and error analysis.
- **An ablation study** — each major design decision removed and retrained, so the
  claims about what helps are measured rather than asserted.
- **Explainability** — Grad-CAM implemented directly with forward/backward hooks.
- **A command-line application** exposing all of the above as subcommands, plus a
  test suite and a generated project report.

### Out of scope

These are deliberate exclusions, not oversights:

- **Detection and localisation.** The system classifies an already-cropped sign;
  it does not find signs in a full road scene. GTSRB ships ROI annotations, and
  detection is a separate problem (GTSDB is the corresponding benchmark).
- **Real-time video inference and tracking.** Single-image classification only.
- **Transfer learning from ImageNet.** A from-scratch network is the point — the
  course objective is to demonstrate understanding of CNN design, not to fine-tune
  someone else's weights.
- **GPU-dependent training.** Everything must run on a CPU so it is reproducible
  on any machine a reviewer has.
- **A graphical or web user interface.** The interface is a CLI; adding a GUI
  would add surface area without demonstrating any additional course concept.
- **Deployment to embedded hardware.** Inference latency is measured and reported
  (0.67 ms/image on CPU), but no embedded port is attempted.

---

## Target users

| user | what they need from the system | how they use it |
|---|---|---|
| **Student / researcher** (primary) | A correct, reproducible baseline for GTSRB that can be modified and re-run, and an evaluation protocol that does not lie. | `prepare`, `train`, `evaluate`, `train --no-clahe` and the other ablation flags. |
| **Course evaluator / reviewer** | To verify the claims quickly without a GPU, a long download, or a notebook environment. | `predict` and `gradcam` on the committed checkpoint and sample images; `python -m unittest discover tests` to check the split logic on synthetic data in under a second. |
| **ML practitioner evaluating the leakage problem** | A worked, testable demonstration that grouped splitting changes the reported number, and by how much. | `tests/test_split_logic.py`, the `Why the split matters` section of the README, and the measured val/test gap. |

The system assumes a user who is comfortable at a command line and has Python
3.10+. It does not assume a GPU, a specific operating system, or any prior GTSRB
familiarity — `prepare` fetches and lays out everything.

---

## High-level features

**1. Dataset preparation with a leak-proof split (`prepare`)**
Downloads and extracts the official archives, parses annotations, groups images by
the composite key `(class_id, track_number)` — necessary because track numbers
restart at zero inside every class directory — and emits stratified,
track-disjoint train and validation manifests. Disjointness is asserted when the
split is built.

**2. Training a CNN from scratch (`train`)**
Trains `TrafficSignNet` with AdamW + OneCycle, a `WeightedRandomSampler` for the
~10x class imbalance, label smoothing, gradient clipping, early stopping on
validation accuracy and a wall-clock budget. Writes a self-describing checkpoint
that carries its own config, class names and normalisation statistics.

**3. Evaluation and error analysis (`evaluate`)**
Reports accuracy, balanced accuracy, macro and weighted F1 and top-5 on the
official test set, plus per-class precision/recall/F1, a confusion matrix, the
most frequent confusion pairs, and a grid of the highest-confidence mistakes.

**4. Inference (`predict`)**
Classifies a single image and returns the label and confidence, with a `--json`
mode for scripting.

**5. Explainability (`gradcam`)**
Produces a Grad-CAM heatmap over the last convolutional layer showing which pixels
drove a prediction, implemented with hooks rather than an external library.

**6. Ablation study (`train --no-clahe`, `--balance none`, `--width-mult`)**
Retrains with individual design decisions disabled so their contribution is
measured on the same held-out test set.

**7. Reporting (`figures`, `report-figures`, `design-figures`, `build-report`)**
Regenerates every figure and rebuilds the project report from the current
`metrics.json`, so the document cannot drift from the code that produced it.

**8. Verification (`tests/`)**
A unittest suite whose central claims — that whole tracks stay together, that no
image is lost or duplicated, that augmentation never mirrors an image — run on
synthetic data and therefore pass on a bare clone with no dataset present.

---

## Success criteria

The project is successful if all of the following hold, and each is independently
checkable from the repository:

1. Train and validation share **zero** tracks and zero images, asserted in code
   and covered by tests.
2. Test accuracy is reported on the **official** GTSRB test set, which is never
   used for training or model selection.
3. The validation/test gap is small, indicating the split is not leaking.
4. Every design decision claimed to help is backed by an ablation run.
5. The whole pipeline runs on a CPU, and a reviewer can verify the core claims
   without downloading the dataset.
