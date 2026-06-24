# AlphaDent — YOLO Instance Segmentation for Dental Finding Detection

A machine learning project for the Kaggle competition **AlphaDent: Teeth Marking**.  
The task is to detect and segment dental findings (Caries, Crown, Abrasion, etc.) from panoramic X-ray images using YOLO instance segmentation models.

---

## Current Best Result

**Best submission (public leaderboard): `0.31189`** — a **V6+V10 ensemble + horizontal-flip TTA**
(full-image inference, class-wise NMS), up from **`0.27047`** for the single V6 model (**+0.0414**).
This is the **first result to beat single-model V6 on the leaderboard**, and it took **zero
additional training** — `src/09` validated the gain on val, `src/10` produces the submission.

| Approach | Public LB | Notes |
|---|---:|---|
| **V6+V10 ensemble + hflip TTA** | **0.31189** | production submission (`src/10`) |
| V6 single model | 0.27047 | previous best submission |

The primary *development* metric is **Mask mAP50-95** (val), where the single-model full-image
approach plateaued at ~0.23–0.24 (V6 0.2336, V10 0.2341). Notably the **leaderboard ensemble gain
(+0.041) is far larger than the comparable-val-metric gain (+0.008** measured in `src/09`) — the
local metric was conservative. Neither the P2 head (V12), tile training (V13, −0.11), the two-stage
refiner, MedSAM, nor NWD-default (V15) beat the single-model baseline; the **inference-time ensemble
did**.

---

## Project Structure

```
AlphaDent/
├── README.md                   # This file
├── .gitignore
├── docs/
│   ├── training_overview.md                  # Workflow and model overview
│   ├── AlphaDent_training_summary_EN.md      # Detailed experiment log (English)
│   ├── AlphaDent_training_summary_CN.md      # Detailed experiment log (Chinese)
│   ├── future_loss_modification_notes.md     # Research notes: loss ideas (unimplemented)
│   ├── small_object_research_notes.md        # Research notes: two-stage detect-then-refine (CLOSED)
│   ├── medsam_refine_research_notes.md       # Research notes: keep V6 boxes, swap mask via MedSAM (NO-GO)
│   └── small_object_box_quality_notes.md     # Research notes: fix loose tiny boxes at the loss (NWD, V15)
├── results/
│   ├── version5_results.csv    # Training metrics per epoch, V5
│   ├── version6_results.csv    # V6
│   ├── version7_results.csv    # V7
│   ├── version8_results.csv    # V8
│   ├── version9_results.csv    # V9
│   ├── version10_results.csv   # V10
│   ├── version11_results.csv   # V11 (Plan D, regressed)
│   ├── version12_results.csv   # V12 (P2 head, did not beat baseline)
│   ├── version13_results.csv   # V13 (tile training, severe regression −0.11)
│   ├── version14_results.csv   # MedSAM Phase 0 eval (per-class AP per variant; NO-GO)
│   └── version15_results.csv   # V15 (NWD box loss, λ=0.5/C=5.0) — underwhelmed, sat at plateau
├── src/
│   ├── 01-yolo-seg-baseline-training-alphadent.ipynb   # V13: tile + train (self-contained)
│   ├── 02-alphadent-yolo-seg-submission.ipynb          # V13: tiled inference + submission
│   ├── 03-alphadent-val-map-eval.ipynb                 # comparable full-image mAP (V13 vs V6, same code)
│   ├── 04-stage2-oracle-roi.ipynb                      # Phase 0 oracle for two-stage detect-then-refine
│   ├── 05-stage1-recall-and-transfer.ipynb             # Phase 1a/1b: real V6 Stage-1 recall + transfer check
│   ├── 06-stage2-phase1c-real-boxes.ipynb              # Phase 1c: retrain Stage 2 on real V6 boxes + bg class (FAILED)
│   ├── 07-medsam-mask-refine.ipynb                     # MedSAM Phase 0: keep V6 boxes, swap mask (zero-training, NO-GO)
│   ├── 08-yolo-seg-nwd-training.ipynb                  # V15: yolov8s-seg + NWD box loss (single variable vs V6)
│   ├── 09-ensemble-tta-eval.ipynb                      # V6+V10 ensemble + hflip TTA, val gain check (comparable Mask mAP)
│   └── 10-ensemble-tta-submission.ipynb                # ensemble + TTA submission (conf floor tuned on val) — LB 0.31189
├── models/                     # Local trained checkpoints — NOT tracked in git (see .gitignore)
│   ├── version6_best.pt        # V6 detector (production; ensemble member)
│   ├── version10_best.pt       # V10 detector (production; ensemble member)
│   └── version15_best.pt       # V15 (NWD) — line on hold
└── tools/
    ├── val_native_yolo_seg.py      # canonical native YOLO Mask mAP baseline (Exp 1A)
    ├── sweep_yolo_conf.py          # submission confidence-threshold sweep (Exp 1D)
    └── make_clahe_yolo_dataset.py  # CLAHE preprocessing dataset builder (Exp 1B)
```

> **Cleanup 2026-06-24:** the failed-line files were removed — `stage2/` (two-stage run outputs +
> `stage2_best.pt`), `models/stage2_p1c_best.pt`, `tools/tile_yolo_seg.py` (V13 tiling),
> `tools/infer_sahi_yolo_seg.py` (SAHI), `experiments/train_small_object_friendly.py` (V11 Plan D),
> and `configs/yolov8s-seg-p2.yaml` (V12 P2 head). The experiments remain documented below and in the
> experiment log — only the now-dead files are gone.

> **Not tracked in git:** dataset images/labels, model weight files (`*.pt`, incl. the `models/` folder), YOLO training output directories (`runs/`).

---

## Experiment History

| Version | Model | Image Size | Key Change | Best Mask mAP50-95 | vs Previous |
|---|---|---:|---|---:|---|
| V5 | YOLOv8s-seg | 640 | Initial baseline | 0.1975 | — |
| V6 | YOLOv8s-seg | 768 | Higher resolution | 0.2336 | **+0.0361** |
| V7 | YOLOv8s-seg | 896 | Higher res + strong oversampling + reduced aug | 0.2260 | -0.0076 |
| V8 | YOLOv8s-seg | 896 | Image size only (controlled) | 0.2260 | -0.0076 |
| V9 | YOLOv8m-seg | 768 | Larger model | 0.2320 | -0.0016 |
| V10 | YOLOv8s-seg | 768 | Mild rare Caries oversampling | 0.2341 | **+0.0005** |
| V11 | YOLOv8s-seg | 768 | Plan D: `mosaic=0`, `mixup=0`, `copy_paste=0.2` | 0.2135 | **−0.0206** |
| V12 | YOLOv8s-seg + P2 head | 768 | Stride-4 (P2) small-object segment head | 0.2215* | **−0.0126** |
| V13 | YOLOv8s-seg (tiles) | 640/tile | Crop / tile-based training (changes the input) | 0.0993† | **−0.1106** |

\* V12's 0.2215 is a single-epoch spike (ep32); the sustained level is ~0.21. See the V12 section in the experiment log.

† V13's 0.0993 is the comparable full-image (tiled + merged) Mask mAP50-95, vs V6 re-scored with the same code (0.2099) — not vs the historical 0.234. See the V13 section.

See [`docs/AlphaDent_training_summary_EN.md`](docs/AlphaDent_training_summary_EN.md) for the full experiment log with per-version analysis, interpretation, and conclusions.

---

## Key Findings

### What helped
- Increasing image size from `640` → `768` gave the biggest single improvement (+0.036 Mask mAP50-95).

### What did not help
- Image size `768` → `896` decreased performance (V7, V8).
- Switching to the larger YOLOv8m model did not improve the result (V9).
- Both mild and strong rare Caries oversampling traded precision for recall without improving Mask mAP50-95 (V7, V10).
- Disabling mosaic/mixup and adding copy-paste (V11, Plan D) regressed Mask mAP50-95 by 0.020 — removing mosaic accelerated overfitting; copy-paste did not compensate. Retest copy-paste with mosaic kept on.
- Adding a P2 stride-4 small-object head (V12) did not break the plateau (best 0.2215, a one-epoch spike; sustained ~0.21, ≈−0.02 vs baseline). Decisive evidence: recall did **not** improve (0.393 vs V10's 0.468), so the extra high-resolution head did not detect more tiny lesions.
- Crop/tile-based training (V13) regressed severely (comparable Mask mAP50-95 0.0993 vs V6's re-scored 0.2099, **−0.11**). Tiling clips large objects out of training (`MIN_AREA_FRAC`), fragments them at inference, and the merge step never reassembles them, so the large classes — which carry most of the mAP — collapse (Abrasion −0.41, Crown −0.43).

### Main bottleneck (re-framed after V13)
Error analysis showed ~78% of validation objects occupy <1% of the image area — but that is the **object-count** distribution, **not** the **mAP-weight** distribution. mAP is averaged per class, and the score is carried by the large/common classes (V6: Abrasion 0.65, Crown 0.63), not by the rare, tiny Caries (low AP for every model, single-digit support). So the "small-object bottleneck" framing overstated the headroom: improving tiny Caries barely moves mAP, and any small-object effort that sacrifices the large classes (as tiling did) backfires. The full-image YOLO approach is plateaued at ~0.23–0.24 Mask mAP50-95, largely because the big classes are near saturation.

---

## V12 Result (P2 small-object head) — did not break the plateau

V12 attacked the small-object bottleneck at the **architecture** level: a stride-4 (P2) segment head added to YOLOv8s-seg (192×192 grid at `imgsz=768` vs 96×96 at P3), with augmentation reverted to the clean V6 baseline so the P2 head was the only change.

**Result: best Mask mAP50-95 = 0.2215 @ epoch 32, but that is a single-epoch spike (ep31 = 0.1965, ep33 = 0.1946); the sustained level over the final epochs is ~0.21.** Even taking the spike at face value, this is ≈−0.013 vs the V6/V10 baseline; the sustained level is ≈−0.02. The decisive evidence is that **recall did not improve** (0.393 vs V10's 0.468) and Mask mAP50 also fell (0.394 vs 0.41+) — the extra high-resolution head did not detect more tiny lesions, which was its entire purpose.

**Conclusion: the P2 head does not break the ~0.23–0.24 plateau.** Adding a small-object head to full-image training is not the answer for this dataset.

## V13 — crop / tile-based training (trained, FAILED −0.11)

V12 suggested the bottleneck was the **full-image input**, so V13 changed it: each panoramic image is sliced into overlapping tiles and the model trains on the tiles, so a lesion that was ~5 px in the downscaled full image becomes ~20–40 px in a tile. Stock `yolov8s-seg`, clean V6 augmentation, tiled input as the single change vs V6.

**Result (comparable full-image metric, same code for both models — `src/03-alphadent-val-map-eval.ipynb`):**

| Mask metric (full val) | V13 (tiled) | V6 (native, re-scored) | Delta |
|---|---:|---:|---:|
| mAP50    | 0.2428 | 0.3687 | −0.1259 |
| mAP50-95 | **0.0993** | **0.2099** | **−0.1106** |

V6 re-scored = 0.2099 vs the historical 0.234; the ~0.024 gap is the metric-implementation difference (applied identically to both models, so the −0.11 delta is real). The collapse is entirely in the **large** classes: Abrasion 0.234 vs 0.647 (−0.41), Crown 0.200 vs 0.631 (−0.43), Filling 0.181 vs 0.280.

**Why it failed:** (1) `MIN_AREA_FRAC=0.35` drops large objects that straddle tile boundaries from training; (2) inference only sees fragments of each large object per tile; (3) `merge_detections` de-duplicates *overlapping* detections but never stitches non-overlapping fragments back together. The large classes carry most of the per-class-averaged mAP, so destroying them is catastrophic. The intended tiny-Caries gain never materialised (±0.01–0.02 on classes with single-digit support).

**Conclusion:** naive tiling is the wrong global strategy for this dataset. **V6 (≈0.234) remains the best model and should be used for submissions.** A small-object approach must not sacrifice the large classes (e.g. a hybrid: full-image model + tiling as an auxiliary small-object branch only).

**Evaluation tooling:** `src/03-alphadent-val-map-eval.ipynb` scores both checkpoints on the full val images with one self-contained mask-mAP implementation (mask-IoU matching → 10 IoU thresholds → 101-point AP); it expects `V6_best.pt` and `V13_best.pt` as Kaggle Datasets.

## Stage-2 detect-then-refine — Phase 0 oracle (direction validated)

After V13, the next direction is a **two-stage detect-then-refine** pipeline (full design in
[`docs/small_object_research_notes.md`](docs/small_object_research_notes.md)): a detector (V6) finds
boxes, small boxes are re-cropped from the **original-resolution** image and refined by a second
model. `src/04-stage2-oracle-roi.ipynb` runs **Phase 0** — an oracle that uses validation **GT
boxes** as a "perfect Stage 1" to measure the upper bound of a Stage-2 refiner (U-Net + ImageNet
ResNet18 encoder, class + fine mask).

**Phase 0 result (30 epochs, oracle):** the small Caries classes *with adequate support* clearly
beat V6 — Caries 1/2/3/5 by **+0.11 to +0.22** Mask AP. Oracle mAP50-95 = **0.312**, Hybrid
(large→V6, small→Stage2) = **0.331**, vs V6 0.210. Crown regressed (−0.165), confirming large
objects should route to V6. (Per-class run outputs were under `stage2/`, removed in the 2026-06-24
cleanup; the numbers above are the record.)

**Honest caveats (these define Phase 1):** the oracle assumes **perfect recall** (GT boxes), so
part of the gain is perfect localization, not refinement — the real ceiling depends on a real
Stage-1 detector's small-box recall. And small classes are **low-weight**, so the aggregate
competition mAP will rise only modestly even in the best case. **Phase 1** therefore measures real
Stage-1 recall first, then retrains Stage 2 on real detector boxes **with an added background class**
(to reject false-positive boxes). V6 (≈0.234) remains the production model.

`src/05-stage1-recall-and-transfer.ipynb` runs **Phase 1a** (V6-as-Stage-1 per-class localization
recall — the gate) and **Phase 1b** (transfer check: V6 boxes → current `stage2_best.pt`, `full` and
`TP-only` pipeline Mask mAP). It needs the V6 detector + `stage2_best.pt` as Kaggle inputs.

**Phase 1a/1b result:** the gate **passed** —
at conf=0.05 V6 localizes the supported small Caries (recall@IoU0.3: Caries 1/2/3/5 ≈ 0.89/0.73/0.58/0.80;
recall collapses 40–60% if conf is raised to 0.25, so Stage 1 must stay at conf≈0.05). But the transfer
was **weak**: `full@0.05 = 0.182` (below V6, no background class to reject FPs) and even the perfect-FP
`TPonly@0.05 = 0.218` only matched V6, with the oracle's Caries gains gone — attributable to the GT→V6
box-framing gap + the missing background class.

`src/06-stage2-phase1c-real-boxes.ipynb` runs **Phase 1c** (trained 2026-06-18): retrain Stage 2 on
**V6's predicted TRAIN boxes at conf=0.05** (IoU≥0.5→foreground, <0.3→background, [0.3,0.5)→ignored;
background subsampled ~3:1) with an added **background class** (`nc+1`), warm-started from `stage2_best.pt`.

**Phase 1c result — FAILED, NO-GO.** Every
pipeline variant scored **below V6 0.2099**: `full@0.05` = 0.157, `TPonly@0.05` (perfect FP rejection,
the ceiling) = 0.178, and the derived hybrid (large→V6, small→Stage2) ≈ 0.203 — all inside or below the
noise band. The oracle's Caries gains **evaporated on real boxes** (TPonly Caries 1/2/5 = 0.079/0.061/0.107
vs V6 0.120/0.085/0.110, vs oracle 0.234/0.259/0.329). Because `TPonly` removes false positives entirely
and still ≈V6, the whole oracle→real gap is **Stage-1 box quality**, not Stage-2 capability: Stage 1 must
run at conf≈0.05 to recall small Caries, but those boxes are too loose to frame the ROI like the tight GT
boxes the oracle used. **The two-stage detect-then-refine line is closed; V6 (≈0.234) remains the
production model.** Next direction is all-class / capacity levers (TTA, V6+V10 ensemble, larger backbone).

## MedSAM mask refinement — Phase 0 (run 2026-06-23, NO-GO for a zero-shot swap)

A *different* lever from the two-stage line (full design in
[`docs/medsam_refine_research_notes.md`](docs/medsam_refine_research_notes.md)): keep V6's box +
class + confidence and only **swap the coarse YOLO mask for a MedSAM (box-prompted) mask**, targeting
**large-class mask IoU** (where the mAP weight is and where V6's boxes are trustworthy), not
small-object localization. `src/07-medsam-mask-refine.ipynb` runs Phase 0 (zero-training); results in
`results/version14_results.csv`.

**Result (comparable mask-mAP, same matcher for every variant; V6 native baseline in-notebook):**

| Variant | Aggregate (9 cls) | Large (Abr/Crown/Fill) |
|---|---:|---:|
| `v6_native@0.05` | 0.1970 | 0.4938 |
| `v6box_medsam@0.05` (real pipeline) | **0.1822** (−0.015) | **0.4989** (+0.005) |
| `oracle_medsam` (GT-box ceiling) | **0.3568** | **0.6928** |

**NO-GO for a blanket swap:** the large-class gain (+0.005) is inside the ~0.003 noise band and the
aggregate regressed (−0.015). The win is real but **concentrated in Abrasion alone** (0.618 → 0.665,
+0.047); Crown regressed (−0.035) and the small Caries **collapsed** (Caries 1/2: ~0.1 → 0.017,
MedSAM segments the whole tooth on a loose box). The GT-box oracle is the project's highest (0.357 /
0.693) and even rescues the small Caries — so the failure is **box quality, not MedSAM mask
quality**: the same wall the two-stage line hit. V6 (≈0.234) stays production; a decoder-only
fine-tune (helps the domain gap, not the box gap) or a pivot to all-class levers are the open options.

## V15 — NWD box loss (built 2026-06-23, not yet trained)

Both closed lines above identified the **same wall: loose tiny boxes.** V15 attacks it at the source
instead of refining after the fact. The root cause is that **IoU/CIoU is unstable for tiny boxes** (a
2 px shift on an 8 px lesion swings IoU wildly), so the small-Caries boxes never tighten. V15 blends
**NWD (Normalized Gaussian Wasserstein Distance)** — which models each box as a 2-D Gaussian and is
smooth under small shifts — into the box regression loss:

```
box_loss = λ·(1 − CIoU) + (1 − λ)·(1 − NWD)
```

Large boxes keep being driven by CIoU; small boxes get the stable NWD signal. **Single variable vs
V6:** only the regression loss changes (model, full-image input, `imgsz=768`, augmentation, and data
are the clean V6 baseline), so the in-training Mask mAP50-95 is directly comparable to V6 ≈0.234.

Implemented in `src/08-yolo-seg-nwd-training.ipynb` (a new training notebook; `src/01` is left
untouched) by monkey-patching `ultralytics.utils.loss.BboxLoss.forward`. Full design, knobs
(`NWD_IOU_RATIO`, `NWD_CONSTANT`), the recommended sweep order, and the pre-registered eval (leading
indicator = small-Caries localization recall@IoU0.5 via `src/05`) are in
[`docs/small_object_box_quality_notes.md`](docs/small_object_box_quality_notes.md).

**V15 result (trained, default λ=0.5 / C=5.0 — UNDERWHELMED).** `results/version15_results.csv`:
best Mask mAP50-95 ≈ 0.24 (spiky; sustained ~0.228), i.e. roughly at the V6 plateau, no clear win.
The pre-registered leading indicator (re-ran `src/05` with V15 vs V6) **regressed**: small-Caries
recall@IoU0.5 fell on all supported classes (Caries 1/2/3/5 mean −0.035) and the large classes also
dropped — blending NWD globally at λ=0.5 diluted the CIoU signal the large/medium boxes relied on,
and C=5.0 bought no compensating small-box tightening. This is "this knob setting failed," not "NWD
is dead" (a C-sweep or a size-gated NWD remain untried), but the line is **on hold** — the
inference-time ensemble below is the productive direction.

## V6+V10 ensemble + hflip TTA — first leaderboard gain (LB 0.31189)

After the small-object lines (two-stage, MedSAM, NWD) all hit the same box-quality wall, the
productive lever turned out to be the **all-class, zero-training** one: ensemble the two near-tied
diverse models (V6, V10) and add test-time augmentation.

- **Pipeline:** full-image inference; V6 and V10 each predict on the image **and its horizontal
  flip** (detections mirrored back); all detections are pooled and merged by **class-wise NMS**
  (IoU=0.6). The confidence floor is **tuned on val** (`src/10` sweeps it against the comparable
  Mask mAP and picks the highest floor within the 0.003 noise band of the best). Ultralytics
  `augment=True` is a no-op for seg models, so TTA is done manually (hflip).
- **Val check (`src/09`, comparable Mask mAP):** `Ensemble+TTA` = **0.2134** vs V6 anchor **0.2053**
  (**+0.0082**); large classes all up (Abrasion/Filling/Crown), no regression. Attribution: TTA
  alone (+0.0026) and ensemble alone (+0.0031) each sit at the noise edge — only the **combination**
  clears it, so the full 4-pass `Ensemble+TTA` is needed.
- **Leaderboard:** `0.31189` vs single V6 `0.27047` (**+0.0414**) — the LB gain is ~5× the val-metric
  delta, i.e. the local comparable metric badly under-predicted the real gain.
- **Status: this is the production submission.** `src/09` = the val gain check, `src/10` = the
  submission builder. Zero additional training.

---

## Setup

```bash
pip install ultralytics
```

Model training is done in the notebook at `src/01-yolo-seg-baseline-training-alphadent.ipynb`.  
The dataset (images + YOLO-format labels) must be placed locally and is not included in this repository.

---

## Notes

- Always use `weights/best.pt` for evaluation, not `weights/last.pt`.
- Keep experiments controlled — change one major factor at a time.
- The `results/` folder contains per-epoch validation metrics for every completed run.
