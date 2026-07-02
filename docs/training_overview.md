# AlphaDent Teeth Marking - YOLO Segmentation Baseline

This repository/notebook set implements a YOLO-based instance segmentation workflow for the Kaggle competition **AlphaDent: Teeth Marking**.

The task is to detect and segment dental findings from images using YOLO segmentation models.  
The current workflow focuses on training YOLO segmentation baselines, analyzing validation errors, and iteratively improving the training configuration.

---

## 1. Project Goal

The goal of this project is to build a practical computer vision pipeline for dental finding segmentation.

The current implementation uses Ultralytics YOLO segmentation models to predict:

- bounding boxes,
- segmentation masks,
- finding classes.

The main evaluation focus during development is **Mask mAP50-95**, because this reflects strict segmentation quality.

---

## 2. Current Workflow

The project has been organized into multiple notebooks:

1. **EDA and label checking**
   - Verify image-label matching.
   - Check whether YOLO segmentation labels are readable.
   - Visualize ground-truth polygons.

2. **Training-only notebook**
   - Train YOLO segmentation models.
   - Save only the important outputs:
     - `weights/best.pt`
     - `weights/last.pt`
     - `results.csv`

3. **Error analysis notebook**
   - Re-evaluate `best.pt`.
   - Compare train vs validation performance.
   - Inspect per-class performance.
   - Test confidence threshold sensitivity.
   - Analyze object size distribution.
   - Visualize GT masks vs predicted masks.

4. **Submission notebook**
   - Load a trained model.
   - Run inference on test images.
   - Generate the required submission file.

---

## 3. Dataset Format

The training pipeline uses YOLO segmentation format.

Expected structure:

```text
dataset/
├── images/
│   ├── train/
│   └── valid/
├── labels/
│   ├── train/
│   └── valid/
└── yolo_seg_train.yaml
```

Each label file contains one object per line:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

The polygon coordinates are normalized YOLO segmentation coordinates.

---

## 4. Models Tested So Far

Several controlled experiments have been performed.

| Version | Model | Image Size | Main Purpose | Best Mask mAP50-95 |
|---|---|---:|---|---:|
| V5 | YOLOv8s-seg | 640 | Initial stable baseline | 0.1975 |
| V6 | YOLOv8s-seg | 768 | Test higher resolution | 0.2336 |
| V7 | YOLOv8s-seg | 896 | Higher resolution + rare Caries oversampling + reduced augmentation | 0.2260 |
| V8 | YOLOv8s-seg | 896 | Controlled image-size-only experiment | 0.2260 |
| V9 | YOLOv8m-seg | 768 | Test larger model capacity | 0.2320 |
| V10 | YOLOv8s-seg | 768 | Mild rare Caries oversampling | 0.2341 |
| V11 | YOLOv8s-seg | 768 | Plan D: `mosaic=0`, `mixup=0`, `copy_paste=0.2` | 0.2135 |
| V12 | YOLOv8s-seg + P2 head | 768 | Stride-4 (P2) small-object segment head | 0.2215* |
| V13 | YOLOv8s-seg (tiles) | 640/tile | Crop / tile-based training | 0.0993† |
| V14 | (eval-only) | 768 | MedSAM box-prompted mask swap, zero training | NO-GO‡ |
| V15 | YOLOv8s-seg | 768 | NWD-blended box regression loss (λ=0.5, C=5.0) | ~0.24§ |
| V16 | (eval-only) hybrid | 512×1024 | Boxless **semantic**-seg for small caries + V6 large | NO-GO¶ |
| V17 | (eval-only) hybrid | 512×1024 | Boxless **center+offset INSTANCE**-seg small caries + V6 large | NO-GO‖ |

\* V12's 0.2215 is a single-epoch spike (ep32); the sustained level is ~0.21.

† V13's 0.0993 is the comparable full-image (tiled + merged) Mask mAP50-95, vs V6 re-scored with
the same code (0.2099) — not vs the historical 0.234. See the experiment log.

‡ V14 is eval-only (MedSAM Phase 0): keep V6's box/class/score, swap only the mask. Blanket swap
regressed the aggregate (−0.015); the win was Abrasion-only. Failure = box quality, not mask quality.

§ V15's best 0.2415 is a single-epoch spike; sustained ~0.228 = at the V6 plateau. The box-quality
leading indicator (small-Caries recall@IoU0.5) **regressed**, so the NWD-default run failed (line on hold).

¶ V16 (eval-only hybrid): boxless semantic-seg for small caries. Supported-small AP 0.032 vs V6 0.081
(≈−60%); hybrid aggregate 0.1855 vs V6 0.2099. Failure = low resolution + a weak semseg→instance conversion.

‖ V17 (eval-only hybrid): boxless center+offset INSTANCE seg (learned grouping + peak score). After a
fix the machinery is validated (center-recall 0.82, 526 instances) but every supported-small AP = 0.000
because val fg-mIoU ≈0.02 never clears mask-IoU≥0.5 → capped by pixel signal (resolution), not machinery.

**The best full-image single model is V6/V10 (≈0.234) — every version after it (V11–V17) failed to
beat it.** But the best *submission* is not a single trained model: it is the **V6+V10 ensemble +
multi-view TTA** (public LB **0.31753**), an inference-time combination — see §6.

---

## 5. Main Findings

### 5.1 Image size

Increasing image size from `640` to `768` improved performance:

```text
Mask mAP50-95: 0.1975 -> 0.2336
```

However, increasing from `768` to `896` did not improve the result.

Current conclusion:

```text
imgsz=768 is the best tested image size so far.
```

### 5.2 Model size

Changing from `YOLOv8s-seg` to `YOLOv8m-seg` at `imgsz=768` did not improve validation performance.

Current conclusion:

```text
A larger model alone does not solve the main bottleneck.
```

### 5.3 Oversampling and rare Caries classes

Both strong oversampling (V7) and mild oversampling (V10) have been tested.  
Both produced the same pattern: recall improved, precision fell, and Mask mAP50-95 was essentially unchanged.

Current conclusion:

```text
Oversampling alone cannot break through the current performance ceiling.
Both mild and strong oversampling shift the precision-recall trade-off but do not improve strict mask quality.
```

### 5.4 Augmentation (V11)

Plan D disabled the destructive augmentations (`mosaic=0`, `mixup=0`) and enabled mask-aware
`copy_paste=0.2`. It **regressed** by 0.020 (0.2336 → 0.2135). Removing mosaic stripped the
small dataset's main regulariser and accelerated overfitting; copy-paste did not compensate.
Because two variables changed at once, this does not condemn copy-paste itself — retest it with
mosaic kept on.

### 5.5 Architecture (V12)

Adding a stride-4 (P2) small-object segment head did not break the plateau (best 0.2215, a
one-epoch spike; sustained ~0.21). Decisively, recall did **not** improve (0.393 vs V10's 0.468),
so the high-resolution head failed at its one job — detecting more tiny lesions. The bottleneck is
the full-image input, not the detection head.

### 5.6 Input scale / tiling (V13)

Crop/tile-based training regressed severely: comparable Mask mAP50-95 0.0993 vs V6's re-scored
0.2099 (**−0.11**, the worst result in the project). Tiling clips large objects out of training
(`MIN_AREA_FRAC`), fragments them at inference, and the merge step never reassembles them, so the
large classes that carry most of the per-class-averaged mAP collapse (Abrasion −0.41, Crown −0.43).

> **Key reframing (mAP weight ≠ object count):** the "~78% of objects occupy <1% of the image"
> finding is the *object-count* distribution, not the *mAP-weight* distribution. mAP is averaged
> per class and is carried by the large/common classes, not the rare tiny Caries. So the
> "small-object bottleneck" framing overstated the headroom — improving tiny Caries barely moves
> mAP, and any small-object effort that sacrifices the large classes (as tiling did) backfires.

### 5.7 Error analysis

The error analysis showed that:

- Caries classes are much harder than larger classes such as Crown and Abrasion.
- Most objects are very small relative to the full image.
- Lowering confidence threshold recovers more predictions but also creates many false positives.
- Train loss keeps decreasing while validation loss often stops improving, suggesting overfitting.
- `best.pt` should be used instead of `last.pt`.

---

### 5.8 Inference-time ensemble + TTA (the one lever that helped the leaderboard)

After every training-time lever plateaued, the **zero-training, all-class** lever finally moved the
leaderboard: ensemble the two near-tied diverse models (V6 + V10) and add manual test-time
augmentation (Ultralytics `augment=True` is a no-op for seg). Full-image inference, each model on the
image + augmented views (mirrored/rescaled back), pooled and merged by class-wise NMS.

- **hflip-only ensemble** (`src/09` §1–8): val Mask mAP 0.2134 vs V6 0.2053 (+0.008) → **public LB
  0.31189** vs single V6 0.27047 (**+0.041**).
- **+vflip+multi-scale (640/896)** (`src/09` §7b, production): val 0.2173 → **public LB 0.31753**
  (**+0.0056** over hflip-only). This is the current best submission (`src/10`).
- **3-model (add V9 yolov8m):** no aggregate help (−0.0019 vs 2-model) — V9 lifts the large classes
  but drags the small Caries; the ensemble/diversity lever is **exhausted**. Keep the 2-model config.

Note the **LB gain (+0.041) is ~5× the comparable-val-metric gain (+0.008)** — the local metric is a
conservative directional signal, not an absolute predictor of the leaderboard.

### 5.9 Post-plateau lines that all hit the same wall

Four separate attempts to lift the small (Caries) classes were run after V13; **all failed at the
same root cause** — a loose/low-resolution pixel signal for tiny lesions:

- **Two-stage detect-then-refine (V-line via `src/04–06`, CLOSED):** oracle (GT boxes) validated a
  +0.11..+0.22 small-Caries ceiling, but on *real* V6 boxes every pipeline variant scored below V6
  even at the perfect-FP ceiling → the gap is **Stage-1 box quality**, unreachable with a real detector.
- **MedSAM mask swap (V14, NO-GO):** loose V6 Caries boxes make SAM segment the whole tooth → box
  quality, not mask quality.
- **NWD box loss (V15, on hold):** blending NWD globally at λ=0.5 diluted the CIoU signal; recall
  regressed. A C-sweep / size-gated NWD remain untried but the line is paused.
- **Boxless semantic (V16) & instance (V17) seg hybrids (NO-GO):** bypass the box entirely, but the
  512×1024 resize starves tiny caries of pixels (val fg-mIoU ≈0.02) → masks never clear mask-IoU≥0.5.

**Key reframing that explains all of them (mAP weight ≠ object count):** mAP is per-class-averaged and
carried by the large/common classes (Abrasion, Crown), which are near saturation; the rare tiny Caries
are low-weight, so even perfect small-class localization would move the aggregate only modestly.

---

## 6. Current Best (submission vs single model)

**Best submission (public LB 0.31753):** the **V6+V10 ensemble + multi-view TTA** (hflip + vflip +
multi-scale 640/896, full-image, class-wise NMS), produced by `src/10` with the confidence floor tuned
on val. This took **zero additional training**. It is an inference-time combination of two models, not
a single trained checkpoint.

**Best single full-image model (Mask mAP50-95 ≈0.234):**

```text
Model:              yolov8s-seg.pt
Image size:         768
Oversampling:       mild rare Caries oversampling (V10) / none (V6)
Best checkpoint:    weights/best.pt
Best Mask mAP50-95: ~0.2341 (V10) / ~0.2336 (V6) — essentially tied
```

V6 (higher precision) and V10 (higher recall) are the two ensemble members. Every training version
after them (V11–V17) failed to beat this single-model level; the productive gain came from the
inference-time ensemble, not a better-trained model.

---

## 7. Project status — all training lines closed

Every lever on the full-image YOLO approach — and every structural change meant to break it — has now
been tested and none beat the single-model baseline:

- Image size: `640` → `768` helped; `768` → `896` did not (V7, V8).
- Model size: YOLOv8m did not improve over YOLOv8s (V9).
- Oversampling: both mild (V10) and strong (V7) did not improve Mask mAP50-95.
- Augmentation (V11): disabling mosaic + adding copy-paste regressed by 0.020.
- Architecture (V12): a P2 small-object head did not help and did not improve recall.
- Input scale (V13): tile-based training regressed severely (−0.11) by destroying the large classes.
- Small-class refinement (two-stage, MedSAM, NWD, semseg/instance-seg hybrids): all hit the loose-box /
  low-resolution wall (§5.9).

**Conclusion: the productive gain is the inference-time V6+V10 ensemble + multi-view TTA (LB 0.31753);
that is production. No training line is currently active.** Should the plateau be revisited, the only
un-chased threads are the V15 NWD **C-sweep / size-gated** variants (`docs/small_object_box_quality_notes.md`)
and higher-resolution small-class seg — but the mAP-weight reframing bounds the upside of any small-class
work, so accepting the ~0.23–0.24 single-model ceiling (and shipping the ensemble) is the honest default.

---

## 8. Research notes index

The closed/paused experiment lines each keep a design + result note:

1. **Two-stage detect-then-refine** — [`small_object_research_notes.md`](small_object_research_notes.md) (CLOSED/NO-GO).
2. **MedSAM mask refinement** — [`medsam_refine_research_notes.md`](medsam_refine_research_notes.md) (NO-GO).
3. **NWD box-quality loss (V15)** — [`small_object_box_quality_notes.md`](small_object_box_quality_notes.md) (on hold; C-sweep/size-gate untried).
4. **Semantic-seg small-class hybrid (V16)** — [`semseg_small_hybrid_notes.md`](semseg_small_hybrid_notes.md) (NO-GO).
5. **Instance-seg small-class hybrid (V17)** — [`instance_seg_small_hybrid_notes.md`](instance_seg_small_hybrid_notes.md) (CLOSED/NO-GO).
6. **Unimplemented loss ideas** — [`future_loss_modification_notes.md`](future_loss_modification_notes.md) (focal / class-weighted BCE / Tversky, research only).

The authoritative per-version narrative lives in [`AlphaDent_training_summary_EN.md`](AlphaDent_training_summary_EN.md)
(and its `_CN` mirror).

---

## 9. Important Notes

- Use `weights/best.pt` for validation and submission.
- Do not use `weights/last.pt` unless specifically comparing checkpoints.
- Do not judge performance only by final epoch.
- Keep future experiments controlled by changing one major factor at a time.