# AlphaDent — YOLO Instance Segmentation for Dental Finding Detection

A machine learning project for the Kaggle competition **AlphaDent: Teeth Marking**.  
The task is to detect and segment dental findings (Caries, Crown, Abrasion, etc.) from panoramic X-ray images using YOLO instance segmentation models.

---

## Current Best Result

| Model | Image Size | Training Strategy | Best Mask mAP50-95 |
|---|---:|---|---:|
| YOLOv8s-seg | 768 | Mild rare Caries oversampling | **0.2341** (V10) |
| YOLOv8s-seg | 768 | Baseline | 0.2336 (V6) |

The primary development metric is **Mask mAP50-95**, which reflects strict segmentation quality.
Neither the P2 small-object head (V12) nor crop/tile-based training (V13) beat this baseline —
V13 in fact regressed severely (−0.11), because tiling destroys the large objects that carry
most of the score.

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
│   └── small_object_research_notes.md        # Research notes: two-stage detect-then-refine
├── results/
│   ├── version5_results.csv    # Training metrics per epoch, V5
│   ├── version6_results.csv    # V6
│   ├── version7_results.csv    # V7
│   ├── version8_results.csv    # V8
│   ├── version9_results.csv    # V9
│   ├── version10_results.csv   # V10
│   ├── version11_results.csv   # V11 (Plan D, regressed)
│   ├── version12_results.csv   # V12 (P2 head, did not beat baseline)
│   └── version13_results.csv   # V13 (tile training, severe regression −0.11)
├── src/
│   ├── 01-yolo-seg-baseline-training-alphadent.ipynb   # V13: tile + train (self-contained)
│   ├── 02-alphadent-yolo-seg-submission.ipynb          # V13: tiled inference + submission
│   ├── 03-alphadent-val-map-eval.ipynb                 # comparable full-image mAP (V13 vs V6, same code)
│   ├── 04-stage2-oracle-roi.ipynb                      # Phase 0 oracle for two-stage detect-then-refine
│   └── 05-stage1-recall-and-transfer.ipynb             # Phase 1a/1b: real V6 Stage-1 recall + transfer check
├── stage2/                     # Stage-2 (detect-then-refine) run outputs
│   ├── stage2_history.csv      # Phase 0 per-epoch training curve
│   └── stage2_results.csv      # Phase 0 per-class oracle AP vs V6
└── tools/
    └── tile_yolo_seg.py        # V13 canonical tiling library (mirrored inline into 01 & 02)
```

> **Not tracked in git:** dataset images/labels, model weight files (`*.pt`), YOLO training output directories (`runs/`).

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
objects should route to V6. Per-class numbers are archived in `stage2/stage2_results.csv`.

**Honest caveats (these define Phase 1):** the oracle assumes **perfect recall** (GT boxes), so
part of the gain is perfect localization, not refinement — the real ceiling depends on a real
Stage-1 detector's small-box recall. And small classes are **low-weight**, so the aggregate
competition mAP will rise only modestly even in the best case. **Phase 1** therefore measures real
Stage-1 recall first, then retrains Stage 2 on real detector boxes **with an added background class**
(to reject false-positive boxes). V6 (≈0.234) remains the production model.

`src/05-stage1-recall-and-transfer.ipynb` runs **Phase 1a** (V6-as-Stage-1 per-class localization
recall — the gate) and **Phase 1b** (transfer check: V6 boxes → current `stage2_best.pt`, `full` and
`TP-only` pipeline Mask mAP). It needs the V6 detector + `stage2_best.pt` as Kaggle inputs. The
decision to build Phase 1c (retrain on real boxes + background class) is made from its numbers.

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
