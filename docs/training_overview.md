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

\* V12's 0.2215 is a single-epoch spike (ep32); the sustained level is ~0.21.

† V13's 0.0993 is the comparable full-image (tiled + merged) Mask mAP50-95, vs V6 re-scored with
the same code (0.2099) — not vs the historical 0.234. See the experiment log.

Current best baseline (V10 is technically highest; improvement over V6 is negligible; V11/V12/V13
all failed to beat it):

```text
YOLOv8s-seg + imgsz=768 + mild rare Caries oversampling
```

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

## 6. Current Best Baseline

The current best run is V10 (though the improvement over V6 is negligible). V11, V12, and V13 all
failed to beat it, and V13 (tiling) regressed severely (−0.11):

```text
Model:           yolov8s-seg.pt
Image size:      768
Oversampling:    mild rare Caries oversampling
Best checkpoint: weights/best.pt
Best Mask mAP50-95: approximately 0.2341  (V10)
Previous best:   approximately 0.2336       (V6, essentially the same)
```

For practical purposes, either V6 (higher precision) or V10 (higher recall) can be used as the
reference baseline. **Submissions should use V6/V10 — V13 tiling is not a viable submission model.**

---

## 7. Recommended Next Experiment

Every lever on the full-image YOLO approach — and the two structural changes meant to break it —
has now been tested and none beat the baseline:

- Image size: `640` → `768` helped; `768` → `896` did not (V7, V8).
- Model size: YOLOv8m did not improve over YOLOv8s (V9).
- Oversampling: both mild (V10) and strong (V7) did not improve Mask mAP50-95.
- Augmentation (V11): disabling mosaic + adding copy-paste regressed by 0.020.
- Architecture (V12): a P2 small-object head did not help and did not improve recall.
- Input scale (V13): tile-based training regressed severely (−0.11) by destroying the large classes.

Crop/tile training — once the recommended next step — has now been run as V13 and **failed**, so it
is no longer recommended as a wholesale replacement for full-image training. The remaining options,
in order of preference:

1. **Keep V6/V10 (≈0.234) as the production baseline.** Still the best model; use it for submissions.
2. **Hybrid (only if small lesions are still pursued).** Full-image model for the large classes +
   tiling as an *auxiliary* small-object branch, never replacing full-image training. This is the
   only tiling variant that does not sacrifice the large objects.
3. **Clean copy-paste ablation (low effort, does not touch the large classes).** Keep `mosaic=1.0`,
   `mixup=0`, add `copy_paste=0.2–0.3`; isolates copy-paste without the mosaic-removal side effect
   that sank V11.
4. **Accept the ceiling.** The large classes are near saturation and the tiny Caries are low-weight,
   so ~0.23–0.24 may be the practical ceiling for this model/dataset.

---

## 8. Future Optimization Directions

The full-image approach is plateaued and naive tiling has been ruled out (V13). Future work should
preserve the large classes that carry the metric while still attacking the tiny lesions.

Possible directions:

1. **Hybrid full-image + auxiliary tiling**
   - Full-image model handles the large classes (Abrasion, Crown, Filling).
   - Tiling is used only as an auxiliary branch for tiny Caries, then merged.
   - Avoids V13's failure mode (large objects fragmented and dropped).

2. **Two-stage pipeline**
   - Stage 1: detect teeth or suspicious regions.
   - Stage 2: classify and segment local crops.

3. **Per-class tuning**
   - Track per-class mAP after each training run.
   - Remember mAP weight ≠ object count: gains on rare tiny Caries barely move the aggregate.

4. **Validation threshold tuning**
   - Tune confidence threshold on validation set.
   - Avoid globally lowering confidence without checking false positives.

5. **K-fold validation**
   - The validation set is small.
   - K-fold can provide a more reliable estimate, especially for rare classes.

6. **Loss modifications (research notes)**
   - See `docs/future_loss_modification_notes.md` (focal / class-weighted BCE / Tversky) —
     unimplemented.

---

## 9. Important Notes

- Use `weights/best.pt` for validation and submission.
- Do not use `weights/last.pt` unless specifically comparing checkpoints.
- Do not judge performance only by final epoch.
- Keep future experiments controlled by changing one major factor at a time.