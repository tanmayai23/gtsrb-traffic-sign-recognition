# Writing kit — your GTSRB report

Everything needed to write the report yourself: per section, the verified facts,
the one point it has to land, a length target, and questions to get you started.

**Every number here is measured.** Sources: `results/metrics.json`,
`results/history.json`, `data/manifests/split_summary.json`, `ablations/*/metrics.json`.
Don't retype them from memory — copy them.

**Target: 3,000–4,000 words, 12–16 pages with figures.**

---

## How to use this

Work section by section. For each one:

1. Read the facts.
2. Answer the starter questions **out loud or in rough notes** — badly, in your own words.
3. Tidy those notes into sentences.

That order matters. Writing from your own rough answer produces your voice.
Writing from a blank page invites you to reach for a template.

Three habits that make technical prose read well:

- **One idea per paragraph.** If a paragraph needs "and also", split it.
- **Number, then meaning.** "Accuracy is 96.82%. That is 402 errors in 12,630 images."
- **Say why, not just what.** "CLAHE runs on the L channel" is a fact.
  "...because equalising RGB separately shifts hue, and colour identifies sign
  type" is an argument. Arguments are what earn marks.

---

## The through-line

Good reports argue something. Yours is:

> **GTSRB's standard evaluation is leaky, so the usual ~99% figures overstate
> real performance. Splitting by track instead of by image gives 96.82% — lower,
> and honest. The remaining errors are explained by input resolution, not by
> model capacity or data scarcity.**

Every section should serve that. If a paragraph doesn't, cut it.

---

# Section-by-section

## 1. Introduction and problem statement — ~250 words

**Facts**
- 43 classes of German traffic sign; GTSRB (Stallkamp et al., IJCNN 2011)
- 39,209 train / 12,630 official test images
- Image sizes 15×15 to 250×250 px
- Scope: classification of pre-cropped signs. Detection is out of scope.
- Constraints: CPU-only training; must run entirely from a command line

**The point:** why this task matters, and what you did and did not attempt.

**Starter questions**
- Why does a car need to read signs, and what goes wrong if it misreads one?
- Are all errors equally bad? (120 vs 80 km/h vs two warning triangles)
- Why state up front that detection is out of scope?
- How did CPU-only change what you could build?

---

## 2. Dataset — ~350 words + Figure

**Facts**
- Source: `sid.erda.dk` mirror, 3 archives, ~365 MB
- Per-class CSVs give a region of interest (ROI) per image
- Imbalance **10.0×**: max 1,800, min 180 training images per class
- Normalisation (train split only): mean [0.414, 0.382, 0.395], std [0.271, 0.261, 0.269]

**Figure:** `results/class_distribution.png`

**The point:** the data is imbalanced and that shapes later choices.

**Starter questions**
- Why is GTSRB imbalanced? (Hint: how often do you pass a 50 km/h sign vs "beware of ice"?)
- If you trained naively, which classes would suffer, and why?
- Why compute normalisation statistics on training data only? What leaks otherwise?

---

## 3. Split methodology — ~600 words + Figure — **THE KEY SECTION**

Spend the most effort here. It is what separates your report from a tutorial.

**Facts**
- Filenames encode `{track}_{frame}.ppm`, e.g. `00053_00024.ppm` = track 53, frame 24
- A track ≈ **30 consecutive video frames of one physical sign**
- 39,209 images → only **1,307 distinct signs**
- Split by whole track, stratified by class
- Train 31,379 images / 1,046 tracks · Val 7,830 / 261 · **overlap 0**
- Trap: track numbers restart per class, so `00000` exists in class 0 *and* class 1.
  Key must be composite `(class_id, track_number)`
- Verified in `tests/test_split_logic.py` and `tests/test_split_integrity.py`,
  including a test asserting a random split *does* leak

**Figure:** `results/report/split_illustration.png`

**The point:** a random split measures memorisation; a track split measures generalisation.

**Starter questions**
- In your own words, what is a "track"?
- Walk through the failure concretely: frame 11 in train, frame 12 in val — what
  has the model actually learned to do?
- Why would ~99.9% validation accuracy be a *warning sign* rather than a success?
- Why did you assert zero overlap in code instead of eyeballing it once?
- What does it cost you to do this correctly, and why is it worth paying?

**Worth saying explicitly:** your number is lower than most published GTSRB
figures, and you know why. A grader who sees you volunteer that reads confidence,
not weakness.

---

## 4. Preprocessing and augmentation — ~500 words + 2 Figures

**Facts — preprocessing**
- Order: ROI crop (+10% margin) → CLAHE → resize 32×32 → normalise
- CLAHE on the **L channel in LAB**, clip limit 2.0, tile grid 4×4
- Why L only: per-RGB equalisation shifts hue; sign colour is class information
- Why 4×4 not OpenCV's 8×8 default: crops average ~50 px, so 8×8 leaves ~6 px tiles

**Facts — augmentation**
- Rotation ±12°, translation ±10%, scale 0.9–1.1, shear ±0.1
- Brightness ×[0.7, 1.3], contrast ×[0.8, 1.2], occasional noise (σ 0.02, p 0.2)
- Single composed affine matrix, one `warpAffine` call
- `BORDER_REPLICATE`, not zero fill
- **No flips of any kind**

**Figures:** `results/preprocessing_samples.png`, `results/augmentation_samples.png`

**The point:** each preprocessing choice is reasoned, and flipping is *wrong* here.

**Starter questions**
- Why keep a 10% margin instead of cropping tight to the sign?
- Why does chaining three warps damage a 32 px image more than one combined warp?
- What could the model learn from black corners that you don't want it to learn?
- **The flip question:** what happens to class 33 (turn right) if you mirror it?
  Class 19? A speed-limit digit? Why does that make flip augmentation produce
  *mislabelled* training data?

---

## 5. Architecture — ~400 words + Figure

**Facts**
- 3 blocks: Conv3×3 → BatchNorm → ReLU → MaxPool; channels 32 → 64 → 128
- Head: GlobalAvgPool → Dropout(0.3) → Linear(128→43)
- **99,019 trainable parameters**
- Shapes: 3×32×32 → 32×16×16 → 64×8×8 → 128×4×4 → 128 → 43
- Conv layers use `bias=False` (BatchNorm re-centres, so bias is redundant)
- GAP head instead of flatten→Linear: saves ~88k params, resolution-agnostic, CAM-friendly
- Kaiming-normal init, matched to ReLU
- Sizing measured: this config ≈20 min; 48×48 and two-conv-per-block variants each ≈2× slower per epoch

**Figure:** `results/report/architecture.png`

**The point:** the architecture was sized by measurement against a real constraint.

**Starter questions**
- Why is a conv bias redundant when BatchNorm follows it?
- What three things does the GAP head buy you?
- You benchmarked before choosing. Why is that better than picking a size and hoping?

---

## 6. Training — ~350 words + Figure + Table

**Facts**
- AdamW, lr 1e-3, weight decay 1e-4
- OneCycleLR, peak 3e-3, **stepped per batch**
- Cross-entropy, label smoothing 0.05; gradient clip at norm 5.0
- `WeightedRandomSampler`, inverse class frequency
- 15 epochs, ~80 s/epoch, **total 20 min 1 s**
- Best validation **97.98%** at epoch 13
- Final: train 98.86%, val 97.84%

**Figure:** `results/training_curves.png` · **Table:** from `results/history.csv`

**The point:** the schedule suits a short fixed budget, and the curves show no overfitting.

**Starter questions**
- Why OneCycle rather than step decay, given only ~15 epochs?
- Train 98.86% vs val 97.84% — what does a gap that small tell you?
- Why does that reading mean *more* given the track-disjoint split?

---

## 7. Results — ~400 words + Figure + Table

**Facts**
| metric | value |
|---|---|
| accuracy | **96.82%** (12,228 / 12,630) |
| balanced accuracy | 95.82% |
| macro F1 | 95.67% |
| weighted F1 | 96.80% |
| top-5 accuracy | 99.56% |
| errors | 402 |
| inference | 0.67 ms/image (CPU, batch 256) |
| best val | 97.98% |

- 3 classes at F1 = 1.000; 6 at ≥0.99; 3 below 0.90

**Figure:** `results/per_class_f1_test.png`

**The point:** strong and even performance; balanced accuracy confirms the tail isn't neglected.

**Starter questions**
- Why report balanced accuracy *and* raw accuracy? What would raw alone hide?
- The gap between them is 1.00 point. Is that big or small, and what does it mean?
- Top-5 is 99.56%. What does that tell you about the *kind* of mistakes being made?
- Val 97.98% vs test 96.82% — why is a small gap here evidence your split worked?

---

## 8. Error analysis — ~500 words + 3 Figures

**Facts — top confusions**
| true → pred | count | share of true class |
|---|---|---|
| 8 (120 km/h) → 5 (80 km/h) | 29 | 6.4% |
| 21 (double curve) → 18 (general caution) | 27 | 30.0% |
| 5 (80) → 2 (50) | 23 | 3.7% |
| 6 (end of 80) → 42 (end of no passing >3.5t) | 23 | 15.3% |
| 3 (60) → 5 (80) | 20 | 4.4% |

- **208 of 254** top-20-pair errors (82%) fall inside visually similar families
- Worst classes: 21 double curve (F1 0.741), 27 pedestrians (0.750), 42 (0.871)
- **Class 27 has only 60 test images but several smaller classes score 100%**

**Figures:** `results/report/confusion_families.png`,
`results/report/per_class_recall.png`, `results/misclassified_test.png`

**The point:** errors are systematic, driven by resolution — not random, not scarcity.

**Starter questions**
- Look at the confusion list. What do 8→5, 5→2, 3→5 have in common?
- At 32×32, how many pixels does the "1" in "120" actually occupy?
- **The interesting one:** if rare classes were just under-trained, recall would
  fall with class size. Look at `per_class_recall.png` — does it? What does that
  rule out, and what does it leave as the explanation?
- Why show the model's *most confident* errors rather than random ones?

---

## 9. Ablations — ~400 words + Figure + Table

**Facts** (each = full 15-epoch retrain, same test set)
| config | accuracy | balanced | macro F1 | params | Δ acc |
|---|---|---|---|---|---|
| **Baseline** | **96.82%** | 95.82% | 95.67% | 99,019 | — |
| No CLAHE | 95.19% | 94.03% | 93.73% | 99,019 | −1.62 |
| No rebalancing | 96.31% | 94.67% | 95.03% | 99,019 | −0.51 |
| Half width | 91.43% | 92.03% | 90.73% | 26,491 | −5.39 |

**Figure:** `results/report/ablation_comparison.png`

**The point:** every design decision earns its place, measured.

**Starter questions**
- Why is CLAHE worth the most? (What is GTSRB footage actually like?)
- **The subtle one:** removing rebalancing costs 0.51 points of accuracy but 1.15
  of *balanced* accuracy — more than double. Why is that asymmetry exactly what
  you'd predict from something aimed at rare classes?
- Does the rebalancing result connect to your §8 finding about class size?
- Half width costs the most. What does that prove about your 99k baseline?

---

## 10. Explainability — ~300 words + Figure

**Facts**
- Grad-CAM (Selvaraju et al., ICCV 2017), hand-implemented with forward +
  `register_full_backward_hook`, no external library
- Gradients of target logit → per-channel spatial mean → weighted sum → ReLU → normalise
- Map is **8×8**, upsampled for display
- Must run with gradients enabled: `eval()` for BatchNorm, but *not* `no_grad`/`inference_mode`
- All 5 sample images classified correctly (89–98% confidence)

**Figure:** `results/gradcam/gradcam_sample_stop.png`

**The point:** the model attends to the sign, not the background.

**Starter questions**
- Why isn't high accuracy alone proof the model is using the right evidence?
- What would a *bad* Grad-CAM look like — where would the heat be?
- Be honest about the 8×8 limit: what can it show, and what can't it?

---

## 11. Limitations — ~350 words

**Facts to cover**
- Detection out of scope; GTSRB supplies ROIs, so these numbers are an upper bound
- 32×32 is the binding constraint on accuracy (traced directly in §8)
- 8×8 CAM is coarse
- Single dataset, single country
- One run per ablation config; the −0.51 result would be firmer with repeated seeds

**The point:** you know precisely where this is weak. That is a strength.

**Starter questions**
- If someone deployed this tomorrow, what breaks first?
- §8 pointed at resolution. What exactly would you try next, and what would it cost?
- Which of your own results are you *least* confident in, and why?

---

## 12. Reproducibility — ~200 words

**Facts**
- Seeded `random` / `numpy` / `torch`; `use_deterministic_algorithms(warn_only=True)`
- Per-index augmentation RNG, so results hold with dataloader workers > 0
- Self-describing checkpoints (config, class names, norm stats)
- 50 tests; 43 run without the dataset
- Caveat: same-machine reproducible; bit-exact across machines/BLAS is not guaranteed

**Starter question:** why state the caveat instead of just claiming "fully reproducible"?

---

## 13. Conclusion — ~250 words

Restate the through-line in your own words. No new facts. Land: 96.82%, honest
protocol, errors explained by resolution, clear next step.

**Starter questions**
- If someone remembers one sentence from this report, what should it be?
- What did you learn that you didn't expect when you started?

---

## 14. References

1. Stallkamp, Schlipsing, Salmen, Igel. *The German Traffic Sign Recognition
   Benchmark: A multi-class classification competition.* IJCNN 2011.
2. Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via
   Gradient-based Localization.* ICCV 2017.
3. Ioffe, Szegedy. *Batch Normalization.* ICML 2015.
4. Smith, Topin. *Super-Convergence.* arXiv:1708.07120, 2018.
5. Zuiderveld. *Contrast Limited Adaptive Histogram Equalization.* Graphics Gems IV, 1994.
6. Loshchilov, Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019.

---

# Figure index

| # | file | section |
|---|---|---|
| 1 | `results/class_distribution.png` | 2 |
| 2 | `results/report/split_illustration.png` | 3 |
| 3 | `results/preprocessing_samples.png` | 4 |
| 4 | `results/augmentation_samples.png` | 4 |
| 5 | `results/report/architecture.png` | 5 |
| 6 | `results/training_curves.png` | 6 |
| 7 | `results/per_class_f1_test.png` | 7 |
| 8 | `results/report/confusion_families.png` | 8 |
| 9 | `results/report/per_class_recall.png` | 8 |
| 10 | `results/misclassified_test.png` | 8 |
| 11 | `results/report/ablation_comparison.png` | 9 |
| 12 | `results/gradcam/gradcam_sample_stop.png` | 10 |

Optional: `results/confusion_matrix_test.png` (43×43, best as an appendix).

Regenerate everything: `python -m src.cli evaluate && python -m src.report_figures`

---

# Before you submit

- [ ] Every number matches `results/metrics.json` — spot-check five
- [ ] Every figure is referenced in the text by number and actually discussed
- [ ] Section 3 makes the leakage argument clearly — reread it last
- [ ] No sentence you couldn't explain if a grader asked
- [ ] Repo URL is the root form: `https://github.com/tanmayai23/gtsrb-traffic-sign-recognition`
- [ ] Your course's AI-use policy is followed, and disclosure matches what you actually did
