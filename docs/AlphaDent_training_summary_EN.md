# AlphaDent YOLO Segmentation Training Summary

This document summarizes the AlphaDent YOLO segmentation experiments completed so far.  
The goal is to record the purpose of each code version, what parameter was changed, why it was changed, whether the result improved, and what conclusions can be used for the next experiment.

The most important development metric is **Mask mAP50-95**, because this is a segmentation task and strict mask quality matters more than only detecting approximate regions.

---

## 1. Overall Experiment Timeline

| Version | Main Purpose | Model | Image Size | Key Training Change | Best Mask mAP50-95 | Result |
|---|---|---:|---:|---|---:|---|
| V5 | Low-resolution YOLOv8s baseline | YOLOv8s-seg | 640 | First stable baseline run | 0.1975 | Valid baseline |
| V6 | Test higher resolution | YOLOv8s-seg | 768 | Increase `imgsz` from 640 to 768 | 0.2336 | Improved; current best baseline |
| V7 | Test high resolution + rare Caries strategy | YOLOv8s-seg | 896 | Higher `imgsz`, rare Caries oversampling, reduced strong augmentation | 0.2260 | Worse than V6 |
| V8 | Controlled image-size-only experiment | YOLOv8s-seg | 896 | Only increase `imgsz` to 896; no oversampling or extra augmentation changes | 0.2260 | Worse than V6 |
| V9 | Test larger model capacity | YOLOv8m-seg | 768 | Change model from YOLOv8s to YOLOv8m; keep `imgsz=768` | 0.2320 | Almost tied with V6, but not better |
| V10 | Test mild rare Caries oversampling | YOLOv8s-seg | 768 | YOLOv8s-seg + imgsz=768 + mild rare Caries oversampling | 0.2341 | Marginally above V6 but within noise; recall improved, precision fell |

Current best practical baseline (V10 is technically highest but improvement over V6 is negligible):

```text
YOLOv8s-seg + imgsz=768 + mild rare Caries oversampling
```

---

## 2. Version-by-Version Analysis

## V5: YOLOv8s-seg, imgsz=640

### Configuration

| Item | Setting |
|---|---|
| Model | `yolov8s-seg.pt` |
| Image size | 640 |
| Batch size | 16 |
| Epochs | 120 |
| Patience | 25 |
| Horizontal flip | `fliplr=0` |
| Main purpose | Build a stable low-resolution baseline |

### Why this setting was used

This was the first stable YOLO segmentation baseline.  
The image size `640` is a common starting point for YOLO training because it is GPU-friendly and allows quick iteration.

### Best validation result

| Metric | Value |
|---|---:|
| Best epoch | 27 |
| Mask Precision | 0.5641 |
| Mask Recall | 0.3798 |
| Mask mAP50 | 0.3589 |
| Mask mAP50-95 | 0.1975 |
| Box Precision | 0.5770 |
| Box Recall | 0.3857 |
| Box mAP50 | 0.3705 |
| Box mAP50-95 | 0.2268 |

### Interpretation

The model learned meaningful patterns, but segmentation quality was still weak.  
The relatively low Mask mAP50-95 suggested that the model could roughly locate some targets, but strict mask quality was poor.

### Conclusion from V5

The pipeline was valid, but `imgsz=640` was likely too small for this task.  
Since many AlphaDent targets, especially Caries regions, are small, increasing image size was a reasonable next step.

---

## V6: YOLOv8s-seg, imgsz=768

### Configuration

| Item | Setting |
|---|---|
| Model | `yolov8s-seg.pt` |
| Image size | 768 |
| Batch size | 16 |
| Epochs | 120 |
| Patience | 25 |
| Main purpose | Test whether higher resolution improves small-target segmentation |

### Why this parameter was changed

The V5 result suggested that the model might not have enough spatial detail to segment small dental findings.  
Increasing image size to `768` was expected to preserve more local information.

### Best validation result

| Metric | Value |
|---|---:|
| Best epoch | 32 |
| Mask Precision | 0.6977 |
| Mask Recall | 0.4053 |
| Mask mAP50 | 0.4125 |
| Mask mAP50-95 | 0.2336 |
| Box Precision | 0.6701 |
| Box Recall | 0.3972 |
| Box mAP50 | 0.4189 |
| Box mAP50-95 | 0.2568 |

### Change from V5

| Metric | V5 img640 | V6 img768 | Change |
|---|---:|---:|---:|
| Mask mAP50-95 | 0.1975 | 0.2336 | +0.0361 |
| Mask mAP50 | 0.3589 | 0.4125 | +0.0536 |
| Mask Precision | 0.5641 | 0.6977 | +0.1336 |
| Mask Recall | 0.3798 | 0.4053 | +0.0255 |

### Interpretation

This was a clear improvement.  
Precision and mAP improved, meaning the predictions became more reliable and the masks matched the ground truth better.

However, recall only improved slightly, so the model still missed many true objects.

### Conclusion from V6

Increasing image size from `640` to `768` was beneficial.  
V6 became the strongest and most stable baseline so far.

---

## Error Analysis After V6

After the `imgsz=768` baseline, an error analysis notebook was used to diagnose the main bottlenecks.

### Key findings

1. **Caries classes were much worse than larger classes.**  
   Large or visually obvious classes such as Crown and Abrasion performed much better than Caries classes.

2. **Small-object issue was severe.**  
   About 78% of validation objects occupied less than 1% of the image area. Caries classes were especially small.

3. **Train-validation gap was clear.**  
   The model performed much better on the training set than on the validation set, suggesting overfitting or limited generalization.

4. **Lower confidence thresholds increased false positives.**  
   Lowering confidence recovered more predictions, but it also produced many incorrect detections.

### Resulting hypothesis

The main bottleneck was not only image size.  
The problem was likely a combination of:

- very small Caries objects,
- class imbalance,
- false positives,
- class confusion,
- limited generalization,
- full-image training being difficult for tiny dental findings.

---

## V7: YOLOv8s-seg, imgsz=896, rare Caries oversampling, reduced augmentation

### Configuration

| Item | Setting |
|---|---|
| Model | `yolov8s-seg.pt` |
| Image size | 896 |
| Epochs | 120 |
| Patience | 25 |
| Oversampling | Enabled for rare Caries images |
| Augmentation | Reduced strong augmentation, such as lower mosaic and no mixup/copy-paste |
| Main purpose | Improve small Caries detection and reduce augmentation damage to small targets |

### Why these parameters were changed

This run was based on the error analysis after V6.

The reasoning was:

- Caries regions were very small, so higher image size might help.
- Rare Caries classes had very few examples, so oversampling might help the model see them more often.
- Strong augmentations might distort small dental lesions, so reducing them might preserve local details.

### Best validation result

| Metric | Value |
|---|---:|
| Best epoch by Mask mAP50-95 | 10 |
| Mask Precision | 0.4760 |
| Mask Recall | 0.4180 |
| Mask mAP50 | 0.3826 |
| Mask mAP50-95 | 0.2260 |
| Box Precision | 0.4930 |
| Box Recall | 0.4397 |
| Box mAP50 | 0.4041 |
| Box mAP50-95 | 0.2539 |

### Change from V6

| Metric | V6 img768 | V7 img896 + oversampling | Change |
|---|---:|---:|---:|
| Mask Precision | 0.6977 | 0.4760 | -0.2217 |
| Mask Recall | 0.4053 | 0.4180 | +0.0127 |
| Mask mAP50 | 0.4125 | 0.3826 | -0.0299 |
| Mask mAP50-95 | 0.2336 | 0.2260 | -0.0076 |

### Interpretation

This run slightly increased recall but sharply reduced precision.  
The model became more willing to predict objects, but many of the extra predictions were likely false positives.

The combined strategy did not improve the final segmentation metric.

### Conclusion from V7

V7 did not prove that `imgsz=896` itself was bad, because multiple variables were changed together.  
However, it showed that the combined strategy of high resolution, rare Caries oversampling, and reduced augmentation was not better than the V6 baseline.

The next step was to split this into a cleaner controlled experiment.

---

## V8: YOLOv8s-seg, imgsz=896 only

### Configuration

| Item | Setting |
|---|---|
| Model | `yolov8s-seg.pt` |
| Image size | 896 |
| Oversampling | Disabled |
| Manual augmentation changes | Removed |
| Epochs | 120 |
| Patience | 25 |
| Main purpose | Isolate the effect of increasing image size from 768 to 896 |

### Why this parameter was changed

V7 changed too many things at once, so it was unclear whether the worse result came from:

- `imgsz=896`,
- oversampling,
- reduced augmentation,
- or the interaction among them.

V8 was designed as a controlled experiment.  
The only intended change from the V6 baseline was image size.

### Best validation result

| Metric | Value |
|---|---:|
| Best epoch | 10 |
| Mask Precision | 0.4760 |
| Mask Recall | 0.4180 |
| Mask mAP50 | 0.3826 |
| Mask mAP50-95 | 0.2260 |
| Box Precision | 0.4930 |
| Box Recall | 0.4397 |
| Box mAP50 | 0.4041 |
| Box mAP50-95 | 0.2539 |

### Change from V6

| Metric | V6 img768 | V8 img896 only | Change |
|---|---:|---:|---:|
| Mask Precision | 0.6977 | 0.4760 | -0.2217 |
| Mask Recall | 0.4053 | 0.4180 | +0.0127 |
| Mask mAP50 | 0.4125 | 0.3826 | -0.0299 |
| Mask mAP50-95 | 0.2336 | 0.2260 | -0.0076 |

### Interpretation

The controlled run showed that increasing `imgsz` from `768` to `896` alone did not improve the result.

The pattern was similar to V7:

- recall slightly increased,
- precision dropped significantly,
- Mask mAP50-95 decreased.

### Conclusion from V8

The improvement from `640` to `768` did not continue from `768` to `896`.  
For the current full-image YOLOv8s training strategy, `imgsz=768` appears to be a better operating point than `imgsz=896`.

Therefore, continuing to `imgsz=1024` is not recommended at this stage.

---

## V9: YOLOv8m-seg, imgsz=768

### Configuration

| Item | Setting |
|---|---|
| Model | `yolov8m-seg.pt` |
| Image size | 768 |
| Epochs | 120 |
| Patience | 25 |
| Main purpose | Test whether larger model capacity improves performance |

### Why this parameter was changed

After V8 showed that increasing image size to `896` did not help, the next hypothesis was that the smaller YOLOv8s model might not have enough capacity to model subtle Caries features.

To test this, the image size was kept at the best known value, `768`, while the model was changed from YOLOv8s-seg to YOLOv8m-seg.

### Best validation result

| Metric | Value |
|---|---:|
| Best epoch | 32 |
| Mask Precision | 0.4059 |
| Mask Recall | 0.4050 |
| Mask mAP50 | 0.3889 |
| Mask mAP50-95 | 0.2320 |
| Box Precision | 0.5597 |
| Box Recall | 0.3838 |
| Box mAP50 | 0.3986 |
| Box mAP50-95 | 0.2537 |

### Change from V6

| Metric | V6 YOLOv8s img768 | V9 YOLOv8m img768 | Change |
|---|---:|---:|---:|
| Mask Precision | 0.6977 | 0.4059 | -0.2918 |
| Mask Recall | 0.4053 | 0.4050 | -0.0003 |
| Mask mAP50 | 0.4125 | 0.3889 | -0.0236 |
| Mask mAP50-95 | 0.2336 | 0.2320 | -0.0016 |
| Box mAP50-95 | 0.2568 | 0.2537 | -0.0031 |

### Interpretation

YOLOv8m-seg did not clearly improve validation performance.  
Mask mAP50-95 was almost tied with the V6 baseline, but precision dropped sharply.

This suggests that simply increasing model capacity does not solve the main bottleneck.  
The larger model may also be more prone to overfitting or unstable predictions on this dataset.

### Conclusion from V9

The best baseline remains:

```text
YOLOv8s-seg + imgsz=768
```

A larger model did not provide a clear benefit under the current full-image training setup.

---

## V10: YOLOv8s-seg, imgsz=768, mild rare Caries oversampling

### Configuration

| Item | Setting |
|---|---|
| Model | `yolov8s-seg.pt` |
| Image size | 768 |
| Batch size | 16 |
| Epochs | 120 (early stopped at epoch 49 via patience) |
| Patience | 25 |
| Oversampling | Mild only — images containing rare Caries classes duplicated once |
| Augmentation | Default/baseline settings |
| Main purpose | Test whether mild rare Caries oversampling improves Caries detection without harming precision |

### Why this setting was used

All prior directions had been exhausted with the full-image YOLO approach:

- Increasing image size to `896` did not help (V7, V8).
- Switching to a larger model did not help (V9).

The one remaining untested hypothesis was **mild** rare Caries oversampling, using the best-known model and image size.  
The strong oversampling in V7 increased recall but sharply reduced precision.  
V10 tested a controlled, milder version: only duplicating images containing rare Caries classes once.

### Best validation result

| Metric | Value |
|---|---:|
| Best epoch | 24 |
| Mask Precision | 0.5074 |
| Mask Recall | 0.4685 |
| Mask mAP50 | 0.4089 |
| Mask mAP50-95 | 0.2341 |
| Box Precision | 0.5150 |
| Box Recall | 0.4737 |
| Box mAP50 | 0.4260 |
| Box mAP50-95 | 0.2569 |

Training stopped at epoch 49 because patience=25 triggered after no improvement from the best epoch 24.

### Change from V6

| Metric | V6 img768 | V10 img768 + mild oversampling | Change |
|---|---:|---:|---:|
| Mask Precision | 0.6977 | 0.5074 | -0.1903 |
| Mask Recall | 0.4053 | 0.4685 | +0.0632 |
| Mask mAP50 | 0.4125 | 0.4089 | -0.0036 |
| Mask mAP50-95 | 0.2336 | 0.2341 | +0.0005 |

### Interpretation

Mild oversampling produced the same pattern as the strong oversampling in V7:

- recall increased,
- precision dropped significantly,
- Mask mAP50-95 was essentially unchanged.

The +0.0005 improvement in Mask mAP50-95 is within noise and cannot be considered a real gain.  
The model traded precision for recall: it predicted more objects, but many extra predictions were likely false positives.

The best epoch arrived relatively early at epoch 24, and training stopped at epoch 49 with no further improvement.  
This indicates the model reached a performance ceiling quickly, consistent with the plateau seen across all previous versions.

### Conclusion from V10

Mild oversampling did not clearly improve Mask mAP50-95 beyond V6.  
Both mild (V10) and strong (V7) oversampling showed the same trade-off: recall rises, precision falls, the strict mask metric stays approximately the same.

This confirms that **oversampling alone cannot break through the current performance ceiling**.  
The full-image YOLO approach appears to be plateaued at approximately 0.23–0.24 Mask mAP50-95.

The next step should not be another oversampling or model-size experiment.  
A **fundamental change to the training strategy** is required.

---

## 3. Cross-Version Conclusions

## 3.1 Image size conclusion

The comparison across image sizes shows:

```text
imgsz=640  ->  Mask mAP50-95 = 0.1975
imgsz=768  ->  Mask mAP50-95 = 0.2336
imgsz=896  ->  Mask mAP50-95 = 0.2260
```

Conclusion:

- Increasing image size from `640` to `768` helped.
- Increasing image size from `768` to `896` did not help.
- `imgsz=768` is currently the best image size among tested settings.

## 3.2 Model-size conclusion

The model-size comparison at `imgsz=768` shows:

```text
YOLOv8s-seg + imgsz=768  ->  Mask mAP50-95 = 0.2336
YOLOv8m-seg + imgsz=768  ->  Mask mAP50-95 = 0.2320
```

Conclusion:

- YOLOv8m-seg did not improve the strict mask metric.
- YOLOv8m-seg had much lower precision.
- Larger model capacity alone is not the main solution.

## 3.3 Oversampling conclusion

Both strong (V7) and mild (V10) oversampling have now been tested.

```text
V7 strong oversampling  ->  Mask mAP50-95 = 0.2260  (recall up, precision down)
V10 mild oversampling   ->  Mask mAP50-95 = 0.2341  (recall up, precision down)
```

Conclusion:

- Both levels of oversampling shift the precision-recall trade-off in the same direction: more recall, less precision.
- Neither mild nor strong oversampling improved Mask mAP50-95 meaningfully beyond the V6 baseline.
- Oversampling alone cannot solve the Caries detection bottleneck.
- Further oversampling experiments are not recommended without first changing the underlying training pipeline.

## 3.4 Overfitting conclusion

Across multiple runs, the pattern was similar:

- training loss kept decreasing,
- validation loss stopped improving or increased,
- `last.pt` was worse than `best.pt`.

Conclusion:

- Always use `best.pt`, not `last.pt`.
- Simply training longer is unlikely to help.
- Future changes should improve generalization rather than extend epochs.

---

## 4. Recommended Next Direction

All direct improvements to the full-image YOLO approach have now been tested:

- Image size: `640` → `768` helped; `768` → `896` did not.
- Model size: YOLOv8m did not improve over YOLOv8s.
- Oversampling: Both mild (V10) and strong (V7) did not improve Mask mAP50-95.

The full-image YOLO approach appears to be plateaued.  
The next step should be a **fundamental change to the training strategy**.

The primary recommended direction is:

```text
Crop or tile-based training
```

### Why this direction

The error analysis after V6 identified that approximately 78% of validation objects occupied less than 1% of the image area.  
Training on full panoramic images means that tiny Caries regions are extremely small relative to the model input.

Crop-based training addresses this directly:

- Train on tooth-level or region-level crops instead of full images.
- This makes tiny Caries lesions much larger relative to the input.
- The model can then develop finer-grained segmentation representations.

### Suggested next configuration

| Item | Suggested Setting |
|---|---|
| Model | `yolov8s-seg.pt` |
| Image size | 640 or 768 (relative to crop size) |
| Training data | Tooth-level or region-level crops from panoramic images |
| Epochs | 100–150 |
| Patience | 25–30 |
| Augmentation | Default/baseline settings |

### Alternative if crop training is too complex to implement first

If crop-based training is not yet feasible, the next controlled experiment could be per-class mAP analysis to understand which specific Caries subtypes drive the bottleneck, followed by threshold tuning on the validation set.

---

## 5. Longer-Term Optimization Ideas

If the mild oversampling experiment does not improve the result, the next major direction should be to change the training formulation rather than simply increasing YOLO size.

Possible future directions:

1. **Crop or tile-based training**
   - Train on tooth-level or region-level crops instead of full panoramic images.
   - This may make tiny Caries lesions much larger relative to the input.

2. **Two-stage pipeline**
   - Stage 1: detect teeth or suspicious regions.
   - Stage 2: segment and classify findings inside local crops.

3. **Class-specific analysis**
   - Track per-class mAP after every important run.
   - Focus especially on Caries 3, 4, 5, and 6.

4. **Threshold tuning**
   - Use validation data to tune confidence thresholds.
   - Avoid simply lowering confidence globally because it increases false positives.

5. **K-fold validation**
   - The validation set is small and rare classes are unstable.
   - K-fold can provide a more reliable estimate.

---

## 6. Current Best Baseline

V10 is technically the highest-scoring run, though the improvement over V6 is negligible (+0.0005).  
Until a new experiment clearly improves the result, the current best baseline should be considered:

```text
Model:          yolov8s-seg.pt
Image size:     768
Oversampling:   mild rare Caries oversampling
Best checkpoint: weights/best.pt
Best Mask mAP50-95: approximately 0.2341  (V10)
Previous best:  approximately 0.2336       (V6, essentially the same)
```

For practical purposes, either V6 or V10 can be used as the baseline.  
V6 has higher precision; V10 has higher recall.