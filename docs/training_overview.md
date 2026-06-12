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

Current best baseline (V10 is technically highest; improvement over V6 is negligible):

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

### 5.4 Error analysis

The error analysis showed that:

- Caries classes are much harder than larger classes such as Crown and Abrasion.
- Most objects are very small relative to the full image.
- Lowering confidence threshold recovers more predictions but also creates many false positives.
- Train loss keeps decreasing while validation loss often stops improving, suggesting overfitting.
- `best.pt` should be used instead of `last.pt`.

---

## 6. Current Best Baseline

The current best run is V10 (though the improvement over V6 is negligible):

```text
Model:           yolov8s-seg.pt
Image size:      768
Oversampling:    mild rare Caries oversampling
Best checkpoint: weights/best.pt
Best Mask mAP50-95: approximately 0.2341  (V10)
Previous best:   approximately 0.2336       (V6, essentially the same)
```

For practical purposes, either V6 (higher precision) or V10 (higher recall) can be used as the reference baseline.

---

## 7. Recommended Next Experiment

All direct improvements to the full-image YOLO approach have now been tested and did not produce clear gains:

- Image size: `640` → `768` helped; `768` → `896` did not (V7, V8).
- Model size: YOLOv8m did not improve over YOLOv8s (V9).
- Oversampling: both mild (V10) and strong (V7) did not improve Mask mAP50-95.

The recommended next experiment is a **fundamental change to the training strategy**:

```text
Crop or tile-based training
```

Suggested approach:

```text
1. Extract tooth-level or region-level crops from the panoramic images.
2. Train YOLOv8s-seg on these local crops instead of full images.
3. This makes tiny Caries lesions much larger relative to the model input.
```

If crop-based training is not yet feasible, consider starting with per-class mAP analysis and confidence threshold tuning on the validation set as interim diagnostic steps.

---

## 8. Future Optimization Directions

If mild oversampling does not improve validation performance, future work should focus on changing the task formulation rather than simply increasing YOLO size.

Possible directions:

1. **Crop or tile-based training**
   - Train on local dental regions instead of full images.
   - This can make tiny Caries regions larger relative to the model input.

2. **Two-stage pipeline**
   - Stage 1: detect teeth or suspicious regions.
   - Stage 2: classify and segment local crops.

3. **Per-class tuning**
   - Track per-class mAP after each training run.
   - Focus on Caries classes.

4. **Validation threshold tuning**
   - Tune confidence threshold on validation set.
   - Avoid globally lowering confidence without checking false positives.

5. **K-fold validation**
   - The validation set is small.
   - K-fold can provide a more reliable estimate, especially for rare classes.

---

## 9. Important Notes

- Use `weights/best.pt` for validation and submission.
- Do not use `weights/last.pt` unless specifically comparing checkpoints.
- Do not judge performance only by final epoch.
- Keep future experiments controlled by changing one major factor at a time.