# Report notes — GTSRB traffic sign recognition

> **These are working notes, not the report.** Headings, measured numbers and
> figure references are collected here so the report can be written from real
> data. Write the prose yourself, in your own words.
>
> Every number below is filled in from `results/metrics.json` after the full run.
> Regenerate with: `python -m src.cli evaluate`

---

## 1. Problem

- 43 classes of German traffic sign, real dashcam crops (Stallkamp et al., IJCNN 2011)
- 39,209 training images / 12,630 official test images
- sizes from 15×15 to 250×250 px; heavy variation in illumination, motion blur, occlusion
- why it matters: sign recognition is a required component of driver assistance
- framing: single-label classification on pre-cropped signs (detection is out of scope — see §10)

## 2. Data

- source: `sid.erda.dk` mirror, three zips, ~365 MB total, downloaded by `prepare`
- per-class annotation CSVs give a region of interest (ROI) per image
- class imbalance **10.0×** (min 180, max 1800 images per class) → fig: `results/class_distribution.png`
- preprocessing order: ROI crop (+10% margin) → CLAHE → resize 32×32
  → fig: `results/preprocessing_samples.png`
- CLAHE applied to the **L channel in LAB**, clip 2.0, tile 4×4
  - per-RGB-channel equalisation shifts hue, and sign colour carries class information
  - 4×4 tiles rather than OpenCV's 8×8 default because these crops average ~50 px

## 3. Split methodology — the key section

- GTSRB filenames encode `{track}_{frame}.ppm`, e.g. `00053_00024.ppm`
- a **track is ~30 consecutive video frames of one physical sign**
- 39,209 images are only **1,307 distinct signs**
- splitting on images puts frame 11 in train and frame 12 in val → model is
  validated on pictures it has effectively memorised → ~99.9%, meaningless
- **this project splits whole tracks**, stratified by class
- second trap: track numbers restart per class, so `00000` exists in class 0 and
  class 1; the key must be the composite `(class_id, track_number)`
- asserted at build time and unit-tested (`tests/test_split_integrity.py`)

| | images | tracks |
|---|---|---|
| train | 31,379 | 1,046 |
| val | 7,830 | 261 |
| test | 12,630 | — (official) |
| **track overlap** | **0** | **0** |

- consequence to state plainly in the report: the accuracy reported here is
  lower than the ~99% figure common in GTSRB tutorials, and that gap is the
  point — it is the difference between a leaked and an honest validation set

## 4. Augmentation

- rotation ±12°, translation ±10%, scale 0.9–1.1, shear ±0.1
- brightness ×[0.7, 1.3], contrast ×[0.8, 1.2], occasional gaussian noise (σ 0.02, p 0.2)
- composed into **one** affine matrix: three chained warps interpolate three times and smear a 32 px image
- `BORDER_REPLICATE`, not a zero fill — black corners would be a cue that a sample is augmented
- → fig: `results/augmentation_samples.png`

**No horizontal flip, no vertical flip, no 90° rotation.** Mirroring changes
what a sign means:

| | |
|---|---|
| 33 ↔ 34 | turn right ahead ↔ turn left ahead |
| 19 ↔ 20 | dangerous curve left ↔ right |
| 36 ↔ 37 | go straight or right ↔ or left |
| 0–8 | every speed-limit digit becomes unreadable |

Flip augmentation is a default for natural images and is actively wrong here.
Guarded by a regression test (`test_never_mirrors_the_image`).

## 5. Architecture — TrafficSignNet

| block | layers | output |
|---|---|---|
| 1 | Conv(3→32, 3×3) → BN → ReLU → MaxPool | 16×16×32 |
| 2 | Conv(32→64, 3×3) → BN → ReLU → MaxPool | 8×8×64 |
| 3 | Conv(64→128, 3×3) → BN → ReLU → MaxPool | 4×4×128 |
| head | GlobalAvgPool → Dropout(0.3) → Linear(128→43) | 43 |

- **99,019 trainable parameters**
- `bias=False` on convs: BatchNorm re-centres immediately, so a conv bias is redundant
- GAP head rather than flatten→Linear: −88k parameters, input-size agnostic,
  and the canonical arrangement for class activation maps
- sizing was measured, not guessed: at 32×32 with one conv per block a full
  15-epoch run takes 20 min on 8 CPU threads (~82 s/epoch). Larger variants
  benchmarked at roughly 2× that per epoch, for gains that do not justify the wait

## 6. Training

- AdamW, lr 1e-3, weight decay 1e-4
- OneCycleLR, peak 3e-3, 25% warmup, **stepped per batch**
- cross-entropy with label smoothing 0.05
- gradient clipping at norm 5.0
- `WeightedRandomSampler` (inverse class frequency) for the 10× imbalance
- early stopping, patience 6 on validation accuracy; wall-clock guard at 30 min
- actual run: 15 epochs in 20 min 1 s, best checkpoint at epoch 13
- → fig: `results/training_curves.png`, data: `results/history.csv`

## 7. Results

| metric | value |
|---|---|
| test accuracy | **0.9682** |
| balanced accuracy | 0.9582 |
| macro F1 | 0.9567 |
| weighted F1 | 0.9680 |
| top-5 accuracy | 0.9956 |
| errors | 402 / 12,630 |
| inference | 0.67 ms/image (CPU) |
| best val accuracy | 0.9798 |
| epochs run | 15 (best at 13) |
| training time | 20 min 1 s, 8 CPU threads |

Worst classes by F1:

| id | name | precision | recall | F1 | support |
|---|---|---|---|---|---|
| 21 | Double curve | 0.833 | 0.667 | 0.741 | 90 |
| 27 | Pedestrians | 0.886 | 0.650 | 0.750 | 60 |
| 42 | End of no passing by vehicles over 3.5t | 0.786 | 0.978 | 0.871 | 90 |
| 6 | End of speed limit (80km/h) | 0.977 | 0.840 | 0.903 | 150 |
| 5 | Speed limit (80km/h) | 0.909 | 0.932 | 0.920 | 630 |
| 29 | Bicycles crossing | 0.879 | 0.967 | 0.921 | 90 |

Top confusions:

| true | pred | count |
|---|---|---|
| 8 Speed limit (120km/h) | 5 Speed limit (80km/h) | 29 |
| 21 Double curve | 18 General caution | 27 |
| 5 Speed limit (80km/h) | 2 Speed limit (50km/h) | 23 |
| 6 End of speed limit (80km/h) | 42 End of no passing by vehicles over 3.5t | 23 |
| 3 Speed limit (60km/h) | 5 Speed limit (80km/h) | 20 |
| 17 No entry | 7 Speed limit (100km/h) | 17 |
| 2 Speed limit (50km/h) | 1 Speed limit (30km/h) | 12 |
| 5 Speed limit (80km/h) | 7 Speed limit (100km/h) | 12 |

Talking points for the writeup:
- errors are systematic, not random: speed-limit digits and warning-triangle
  pictograms, both of which degrade at 32x32
- val 0.9798 vs test 0.9682 - small gap,
  which is what a non-leaking split should give
- 3 classes reach F1 = 1.000; worst is class 21 (double curve) at 0.741
- balanced accuracy 0.9582 sits ~1pt below raw accuracy,
  the expected signature of 10x imbalance

## 8. Ablations

Each row is one command; fill in test accuracy from the resulting `metrics.json`.

| setting | command | test accuracy |
|---|---|---|
| baseline | `train` | |
| no CLAHE | `train --no-clahe` | |
| no rebalancing | `train --balance none` | |
| class-weighted loss | `train --balance loss` | |
| half width | `train --width-mult 0.5` | |
| 48×48 input | `train --img-size 48` | |

## 9. Explainability

- Grad-CAM (Selvaraju et al., ICCV 2017), implemented directly with forward and
  full-backward hooks on the last conv layer — no external library
- gradients of the target logit are averaged per feature map, used as weights,
  summed, and rectified
- map is **8×8**, upsampled for display: coarse, but enough to show whether the
  network is looking at the sign or at the background
- must run with gradients enabled — `model.eval()` for BatchNorm, but *not*
  `no_grad`/`inference_mode` (a common implementation error)
- → figs: `results/gradcam/*.png`

## 10. Limitations

- CPU-only budget caps depth and input resolution; 32×32 loses fine digit detail
- 8×8 CAM resolution is coarse
- GTSRB supplies ground-truth ROIs; a deployed system needs a detector first,
  so these numbers are an upper bound on a full pipeline
- single dataset, single country — no claim of generalisation to other sign systems
- val/test gap is worth commenting on: both are honest, but the test set comes
  from separate recordings

## 11. Reproducibility

- seeded RNGs (`random`, `numpy`, `torch`), `use_deterministic_algorithms(warn_only=True)`
- augmentation uses a per-index generator, so results hold with dataloader workers > 0
- checkpoints are self-describing (config, class names, normalisation statistics)
- caveat to state: run-to-run on one machine is reproducible; bit-exact equality
  across machines and BLAS builds is not guaranteed

## 12. References

- J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel. *The German Traffic Sign
  Recognition Benchmark: A multi-class classification competition.* IJCNN 2011.
- R. R. Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization.* ICCV 2017.
- S. Ioffe, C. Szegedy. *Batch Normalization.* ICML 2015.
- L. N. Smith, N. Topin. *Super-Convergence: Very Fast Training Using Large
  Learning Rates.* 2018. (OneCycle)
- K. Zuiderveld. *Contrast Limited Adaptive Histogram Equalization.* Graphics Gems IV, 1994.
