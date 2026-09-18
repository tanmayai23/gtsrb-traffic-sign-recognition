# Requirements Specification

Every requirement below is traceable: the **Where** column names the code that
implements it, and **Verified by** names the test, command or artefact that shows
it holds. Nothing here is aspirational — if a requirement is only partly met, it
says so.

---

## 1. Functional requirements

The system is organised into **six functional modules**. Modules 1–3 are the three
major ones required by the brief; 4–6 extend it.

### FM-1 — Data acquisition and preparation

| id | requirement | where | verified by |
|---|---|---|---|
| FR-1.1 | Download the three official GTSRB archives over HTTPS, resuming a partial transfer rather than restarting it. | [src/download.py](src/download.py) | `python -m src.cli prepare` twice; second run re-uses the cache |
| FR-1.2 | Check each archive's size against its expected size before use, and extract safely, rejecting entries whose path escapes the target directory (zip-slip). | [src/download.py](src/download.py) | `_size_ok`; zip-slip guard in `safe_extract` |
| FR-1.3 | Parse the per-image CSV annotations into records of `(path, class_id, roi_box, track_number)`. | [src/prepare.py](src/prepare.py) | `tests/test_split_integrity.py` |
| FR-1.4 | Group images by the composite key `(class_id, track_number)`, because track numbers restart at 0 in every class directory. | [src/prepare.py](src/prepare.py) | `tests/test_split_logic.py::test_composite_track_key` |
| FR-1.5 | Produce a **track-disjoint**, class-stratified train/validation split: no physical sign may appear on both sides, and every class must appear on both. | [src/prepare.py](src/prepare.py) | `tests/test_split_logic.py`, `tests/test_split_integrity.py` |
| FR-1.6 | Assert disjointness at build time and fail loudly rather than emit a leaking split. | [src/prepare.py](src/prepare.py) | assertion raises on violation |
| FR-1.7 | Write train/val manifests to disk so the split is fixed, inspectable and reusable across runs. | [src/prepare.py](src/prepare.py) | `data/` manifests |

**Input:** the GTSRB archives (or a populated `data/raw/`).
**Output:** extracted images plus `train`/`val` manifests.

### FM-2 — Training

| id | requirement | where | verified by |
|---|---|---|---|
| FR-2.1 | Define a CNN from scratch, with no pretrained weights, parameterised by input size and width multiplier. | [src/model.py](src/model.py) | `tests/test_model.py` |
| FR-2.2 | Apply the preprocessing chain: ROI crop with margin → CLAHE on the LAB L-channel → resize → normalise. | [src/transforms.py](src/transforms.py) | `tests/test_transforms.py` |
| FR-2.3 | Augment training images with composed affine geometry and photometric jitter, applied as a **single** warp. | [src/transforms.py](src/transforms.py) | `tests/test_transforms.py` |
| FR-2.4 | **Never** mirror, flip or 90°-rotate an image, since that changes a sign's meaning (turn-left ↔ turn-right). | [src/transforms.py](src/transforms.py) | `tests/test_transforms.py::test_never_mirrors_the_image` |
| FR-2.5 | Compensate for the ~10x class imbalance via a weighted sampler or class weights, selectable at the CLI. | [src/dataset.py](src/dataset.py) | `--balance` flag; ablation run |
| FR-2.6 | Compute normalisation statistics from the **training split only**. | [src/dataset.py](src/dataset.py) | statistics stored in the checkpoint |
| FR-2.7 | Train with AdamW + OneCycle, label smoothing and gradient clipping, reporting per-epoch train/val loss and accuracy. | [src/engine.py](src/engine.py), [src/train.py](src/train.py) | `results/history.csv` |
| FR-2.8 | Stop early on a validation-accuracy plateau, and respect a wall-clock budget. | [src/train.py](src/train.py) | `--patience`, `--max-minutes` |
| FR-2.9 | Checkpoint the best model by validation accuracy, storing weights **and** the config, class names and normalisation statistics. | [src/train.py](src/train.py), [src/utils.py](src/utils.py) | `checkpoints/best.pt` loads without matching flags |

**Input:** manifests + hyperparameters. **Output:** `checkpoints/best.pt`, `history.csv`.

### FM-3 — Evaluation, prediction and analysis

| id | requirement | where | verified by |
|---|---|---|---|
| FR-3.1 | Evaluate on the **official** held-out GTSRB test set, never used for training or model selection. | [src/evaluate.py](src/evaluate.py) | `results/metrics.json` |
| FR-3.2 | Report accuracy, balanced accuracy, macro F1, weighted F1 and top-5 accuracy — not accuracy alone, because the classes are imbalanced. | [src/evaluate.py](src/evaluate.py) | `results/metrics.json` |
| FR-3.3 | Report per-class precision, recall, F1 and support. | [src/evaluate.py](src/evaluate.py) | `results/per_class_metrics_test.csv` |
| FR-3.4 | Produce a confusion matrix and rank the most frequent confusion pairs. | [src/evaluate.py](src/evaluate.py), [src/plots.py](src/plots.py) | `results/top_confusions_test.csv` |
| FR-3.5 | Surface the highest-confidence errors as an image grid, since confident mistakes are the informative ones. | [src/plots.py](src/plots.py) | `results/misclassified_test.png` |
| FR-3.6 | Classify a single image, returning label and confidence, with a machine-readable `--json` mode. | [src/predict.py](src/predict.py) | `python -m src.cli predict --image samples/sample_stop.ppm` |
| FR-3.7 | Measure and report inference latency per image. | [src/evaluate.py](src/evaluate.py) | `results/metrics.json` |

**Input:** a checkpoint + the test manifest (or a single image).
**Output:** `metrics.json`, CSVs, figures, or one prediction.

### FM-4 — Explainability

| id | requirement | where | verified by |
|---|---|---|---|
| FR-4.1 | Generate a Grad-CAM heatmap over the last convolutional layer for a chosen image. | [src/gradcam.py](src/gradcam.py) | `tests/test_model.py` |
| FR-4.2 | Allow explaining an arbitrary target class, not only the predicted one. | [src/gradcam.py](src/gradcam.py) | `--target-class` |
| FR-4.3 | Implement it with forward and **full** backward hooks, and remove them afterwards so the model is left clean. | [src/gradcam.py](src/gradcam.py) | hook-cleanup test |
| FR-4.4 | Save the original, heatmap and overlay as one comparable panel. | [src/plots.py](src/plots.py) | `results/gradcam/` |

### FM-5 — Ablation study

| id | requirement | where | verified by |
|---|---|---|---|
| FR-5.1 | Disable CLAHE, class rebalancing, or halve model width via CLI flags, so each design decision can be retrained in isolation. | [src/cli.py](src/cli.py), [src/config.py](src/config.py) | `run_ablations.sh` |
| FR-5.2 | Write each ablation's metrics to its own directory for comparison against the baseline. | [src/train.py](src/train.py) | `ablations/*/metrics.json` |

### FM-6 — Reporting and interface

| id | requirement | where | verified by |
|---|---|---|---|
| FR-6.1 | Expose every capability through a single CLI entry point with per-subcommand help. | [src/cli.py](src/cli.py) | `python -m src.cli --help` |
| FR-6.2 | Report environment and project status (torch version, device, parameter count, which data is present). | [src/cli.py](src/cli.py) | `python -m src.cli info` |
| FR-6.3 | Regenerate every results figure from the current metrics. | [src/plots.py](src/plots.py), [src/report_figures.py](src/report_figures.py) | `python -m src.cli report-figures` |
| FR-6.4 | Generate the design artefacts (architecture, workflow, use case, class and sequence diagrams) from code, so they cannot drift from the implementation. | [src/design_figures.py](src/design_figures.py) | `results/design/` |
| FR-6.5 | Build the project report as HTML and PDF with every number read from `metrics.json` at build time. | [src/build_report.py](src/build_report.py) | `report/GTSRB_Project_Report.pdf` |

### Workflow

```
prepare ──> train ──> evaluate ──> report
              │          │
              │          └──> predict / gradcam  (single image)
              └──> ablations (retrain with one decision removed)
```

A user runs `prepare` once, `train` to produce a checkpoint, `evaluate` to measure
it, and `predict`/`gradcam` for individual images. Because a trained checkpoint is
committed, steps 3–4 work immediately on a fresh clone.

---

## 2. Non-functional requirements

Eight are specified; the brief requires four.

### NFR-1 — Reproducibility *(the project's defining constraint)*

Identical inputs must give identical outputs on the same machine.

- `random`, `numpy` and `torch` are seeded from a single `--seed`;
  `torch.use_deterministic_algorithms(warn_only=True)` is set.
- Augmentation draws from a **per-index** generator, so results hold even with
  `num_workers > 0`, where global RNG state would otherwise be racy.
- The split is a deterministic function of `(seed, val_frac)`.
- Checkpoints are self-describing, so `predict` and `evaluate` need no flags that
  must be remembered to match the training run.

**Measured:** re-running `train --quick` with a fixed seed reproduces the same
metrics. **Honest caveat:** bit-exact equality *across* machines, BLAS builds or
torch versions is not guaranteed, and a few CPU kernels have no deterministic
variant. Implemented in [src/utils.py](src/utils.py).

### NFR-2 — Correctness of measurement

The reported accuracy must mean what it says.

- Train and validation are track-disjoint, asserted at build time.
- The official test set is used only for final evaluation.
- Normalisation statistics come from the training split only.
- Metrics robust to imbalance (balanced accuracy, macro F1) are reported beside
  raw accuracy.

**Measured:** validation 0.9798 vs test 0.9682 — a small gap, which is what a
non-leaking split should produce.

### NFR-3 — Resource efficiency

Must run on a normal laptop with no GPU.

| budget | actual |
|---|---|
| no CUDA required | CPU-only throughout |
| training time | ~20 min, 15 epochs, 8 CPU threads |
| memory | dataset cached as `uint8` (~96 MB for train) |
| model size | 99,019 parameters, ~1.2 MB checkpoint |
| inference | 0.67 ms/image (batch 256, CPU) |

`--quick`, `--epochs`, `--max-minutes`, `--batch-size` and `--img-size` let a
constrained machine trade accuracy for time or memory.

### NFR-4 — Reliability and error handling

- Downloads resume after interruption; each archive's size is checked against its
  expected size (within a tolerance) before it is used, so a truncated transfer is
  caught rather than extracted.
- Extraction rejects paths that escape the target directory.
- `last.pt` is written **after every epoch** and `best.pt` whenever validation
  accuracy improves, so an interrupted run is never wasted.
- Invalid CLI combinations and missing files fail with an actionable message
  rather than a traceback.
- The split asserts its own invariants instead of silently producing a leaking
  dataset.

### NFR-5 — Maintainability

- 15 single-purpose modules under `src/`, each with one clear responsibility;
  no module reaches past its layer (see the architecture diagram).
- Configuration is centralised in [src/config.py](src/config.py) rather than
  scattered as literals.
- Figures and design diagrams are generated from code, so documentation cannot
  silently drift from the implementation.
- Comments explain *why* a non-obvious decision was made (why LAB-space CLAHE, why
  a single composed warp, why `register_full_backward_hook`), not what the line
  does.

### NFR-6 — Testability

- 50 unittest tests across 5 files.
- The central claims are tested on **synthetic** data, so they pass on a bare
  clone with no dataset downloaded — a reviewer can verify them in under a second.
- One test deliberately demonstrates the bug being avoided: a random split *does*
  leak tracks.
- Dataset-dependent tests skip cleanly rather than fail when `data/` is absent.

**Command:** `python -m unittest discover tests -v`

### NFR-7 — Usability

- One entry point, `python -m src.cli`, with discoverable subcommands and
  per-command `--help`.
- Inference works immediately on a fresh clone via the committed checkpoint — no
  training or download required to see the system work.
- `info` reports exactly what is installed and what data is present.
- A troubleshooting section covers the failure modes that actually occur
  (accidental CUDA wheel, `libGL.so.1` on headless Linux, slow mirror).
- Progress bars and per-epoch metrics during long operations.

### NFR-8 — Portability

- Windows, Linux and macOS; run scripts provided for both PowerShell and bash.
- Python 3.10+; pinned dependency versions in `requirements.txt`.
- Matplotlib uses the `Agg` backend so figure generation works headless.
- `.gitattributes` marks `.ppm` files binary, preventing git from corrupting
  sample pixel data via line-ending translation — covered by
  `tests/test_samples.py`.

### Security note

The system processes a public research dataset and has no authentication, network
service or user data, so conventional security requirements largely do not apply.
The two that do — safe archive extraction (NFR-4) and integrity-checked downloads
— are treated as reliability requirements above.

---

## 3. Traceability summary

| requirement group | modules | tests |
|---|---|---|
| FM-1 data preparation | `download.py`, `prepare.py` | `test_split_logic.py`, `test_split_integrity.py` |
| FM-2 training | `model.py`, `transforms.py`, `dataset.py`, `engine.py`, `train.py` | `test_model.py`, `test_transforms.py` |
| FM-3 evaluation | `evaluate.py`, `predict.py`, `plots.py` | `test_model.py` |
| FM-4 explainability | `gradcam.py` | `test_model.py` |
| FM-5 ablations | `config.py`, `cli.py` | `ablations/*/metrics.json` |
| FM-6 reporting | `report_figures.py`, `design_figures.py`, `build_report.py`, `cli.py` | `test_samples.py` |
