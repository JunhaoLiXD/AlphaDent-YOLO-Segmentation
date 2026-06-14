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
| V11 | Plan D: decouple destructive aug + medical-grade copy-paste | YOLOv8s-seg | 768 | `mosaic=0`, `mixup=0`, `copy_paste=0.2` | 0.2135 | **Clear regression (−0.020)**; disabling mosaic accelerated overfitting |
| V12 | Attack small-object bottleneck at the architecture level | YOLOv8s-seg + P2 head | 768 | Add stride-4 (P2) segment head; aug reverted to clean V6 baseline | 0.2215* | **Did not beat baseline (≈−0.013)**; recall did not improve, so P2 caught no extra small lesions |

\* V12's 0.2215 is a single-epoch spike at ep32; the sustained level is ~0.21, i.e. ≈−0.02 vs the V6/V10 baseline.

Current best practical baseline (V10 is technically highest but improvement over V6 is negligible; neither V11 nor V12 beat it):

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

## V11: YOLOv8s-seg, imgsz=768, Plan D (decoupled aug + copy-paste)

### Configuration

| Item | Setting |
|---|---|
| Model | `yolov8s-seg.pt` |
| Image size | 768 |
| Batch size | 16 |
| Epochs | 60 (template default; CSV ends at epoch 51) |
| Patience | 25 |
| Mosaic | `mosaic=0.0` (disabled) |
| Mixup | `mixup=0.0` (disabled) |
| Copy-paste | `copy_paste=0.2` (enabled) |
| Main purpose | Plan D — stop letting mosaic/mixup destroy tiny lesions, and instead synthesise rare Caries with mask-aware copy-paste |

### Why this setting was used

Every prior direction (image size, model size, oversampling) had plateaued at ~0.23–0.24.
The hypothesis behind Plan D was that the **destructive augmentations** (mosaic downscales
objects; mixup blurs fine mask boundaries) were hurting the very small lesions we care about,
while **copy-paste** is a "medical-grade" augmentation that pastes real lesions with their
masks into new images, boosting rare-Caries exposure without distorting them. The plan was
to decouple the two: turn off the destructive augmentations and turn on copy-paste.
Implemented via `experiments/train_small_object_friendly.py`.

### Best validation result

| Metric | Value |
|---|---:|
| Best epoch (by Mask mAP50-95) | 42 |
| Mask Precision | 0.5656 |
| Mask Recall | 0.4206 |
| Mask mAP50 | 0.3880 |
| Mask mAP50-95 | 0.2135 |
| Box Precision | 0.5747 |
| Box Recall | 0.4288 |
| Box mAP50 | 0.4075 |
| Box mAP50-95 | 0.2372 |

> Note: the run stopped at epoch 51 (best at 42), before patience (25) would have triggered
> at epoch 67 — i.e. the run was cut short. The trend after the peak was already deteriorating,
> so a longer run would not have recovered the gap.

### Change from V6 / V10

| Metric | V6 img768 | V10 + mild oversampling | V11 Plan D | V11 vs best |
|---|---:|---:|---:|---:|
| Mask Precision | 0.6977 | 0.5074 | 0.5656 | — |
| Mask Recall | 0.4053 | 0.4685 | 0.4206 | −0.048 vs V10 |
| Mask mAP50 | 0.4125 | 0.4089 | 0.3880 | −0.021 |
| Mask mAP50-95 | 0.2336 | 0.2341 | 0.2135 | **−0.0206** |
| Box mAP50-95 | 0.2568 | 0.2569 | 0.2372 | −0.020 |

### Interpretation

This is a **clear regression**, not noise (−0.020 Mask mAP50-95 is ~7× our ~0.003 noise band).
Unlike the oversampling runs, this was not a precision/recall trade-off: **both** Mask mAP50
and Mask mAP50-95 dropped together, meaning overall mask quality fell.

The training curves explain why. `train/seg_loss` decreases smoothly throughout, but
`val/seg_loss` bottoms out around epoch 17 (≈2.09) and then climbs steadily to ≈2.44 by
epoch 51 — a textbook overfitting signature, and more pronounced than earlier versions.

The most likely cause is **disabling mosaic entirely**. In YOLO, mosaic is not just an
object-downscaler; it is the strongest source of regularisation and scene diversity. Removing
it let the small dataset overfit faster, and `copy_paste=0.2` (a small number of synthetic
samples) did not compensate. In other words, mosaic's regularisation value outweighed its
small-object downscaling cost — the opposite of the Plan D assumption.

### Conclusion from V11

Plan D as configured **hurt** the strict mask metric. The lesson is not "copy-paste is bad"
but that **the variables were not decoupled**: turning off mosaic and turning on copy-paste at
the same time confounds the result, and the mosaic removal dominated. Copy-paste should be
retested with mosaic **kept on** before any conclusion about copy-paste itself.

---

## V12: YOLOv8s-seg + P2 small-object head, imgsz=768

### Configuration

| Item | Setting |
|---|---|
| Architecture | `yolov8s-seg` + P2 head (4 segment layers, strides 4/8/16/32) |
| Weights | `.load("yolov8s-seg.pt")` — backbone transfers, P2 branch random-init |
| Image size | 768 |
| Batch size | 16 |
| Epochs | 120 (CSV records 57 epochs) |
| Patience | 25 |
| Augmentation | clean V6 baseline: `mosaic=1.0`, `close_mosaic=10`, `mixup=0`, `copy_paste=0` |
| Oversampling | disabled |
| Main purpose | Attack the small-object bottleneck at the **architecture** level — a stride-4 (P2) head gives a 192×192 grid at imgsz=768 (vs 96×96 at P3), so the smallest lesions map to more than one anchor cell |

### Why this setting was used

V11 confirmed that augmentation changes could not break the plateau, and image size,
model size, and oversampling had all been exhausted. The error analysis after V6 showed
~78% of objects occupy <1% of the image area, so V12 targeted that finding directly at the
architecture level: add a high-resolution P2 head. Augmentation was reverted to the clean V6
baseline so the **P2 head is the only change** (single-variable discipline).

### Best validation result

| Metric | Value |
|---|---:|
| Best epoch (by Mask mAP50-95) | 32 |
| Mask Precision | 0.5147 |
| Mask Recall | 0.3934 |
| Mask mAP50 | 0.3939 |
| Mask mAP50-95 | 0.2215 |
| Box Precision | 0.5170 |
| Box Recall | 0.3909 |
| Box mAP50 | 0.4085 |
| Box mAP50-95 | 0.2510 |

> **Important caveat — the best value is a single-epoch spike.** At ep32 every metric jumped
> (Mask mAP50-95 0.1965 → **0.2215** → 0.1946 at ep31/32/33; Box mAP50-95 also spiked to 0.251),
> then fell back the next epoch. This is a lucky checkpoint on a small validation set, not a
> stable level. Over the final epochs (ep50–57) Mask mAP50-95 sits at ~0.20–0.212 and never
> re-reaches the ep32 peak. The honest read of V12's true level is **~0.21**.

### Change from V6 / V10

| Metric | V6 img768 | V10 + mild oversampling | V12 P2 head | V12 vs best |
|---|---:|---:|---:|---:|
| Mask Precision | 0.6977 | 0.5074 | 0.5147 | — |
| Mask Recall | 0.4053 | 0.4685 | 0.3934 | **−0.075 vs V10** |
| Mask mAP50 | 0.4125 | 0.4089 | 0.3939 | −0.015 |
| Mask mAP50-95 | 0.2336 | 0.2341 | 0.2215 | **−0.0126** (spike); ~−0.02 sustained |
| Box mAP50-95 | 0.2568 | 0.2569 | 0.2510 | −0.006 |

### Interpretation

V12 did **not** break the plateau. Even taking the ep32 spike at face value it is ≈−0.013
below the baseline, and the sustained level (~0.21) is ≈−0.02 below it — comparable to V11.

The most informative signal is **recall**. The P2 head exists to recover the tiny lesions
(~78% of objects <1% area); if it worked, recall should rise. Instead Mask recall *fell* to
0.393 — well below V10's 0.468 and below V6's 0.405 — and Mask mAP50 also dropped (0.394 vs
0.41+). So the extra high-resolution head did not detect more small objects; it mostly added
parameters and training difficulty without the intended benefit.

This is not an under-training artefact. The P2 branch starts from a much higher loss (ep1
`seg_loss` ≈4.68 vs ~2.6 for the standard head) because it is randomly initialised and converges
more slowly, but it had caught up by ep32 and produced no new peak in the following 25 epochs.
`val/seg_loss` bottoms around ep26 (~2.26) and then drifts in the 2.30–2.45 band — overfitting
is present but less severe than V11's monotonic climb.

### Conclusion from V12

Adding a P2 small-object head to **full-image** YOLOv8s-seg does not break the ~0.23–0.24
plateau on this dataset, and notably fails to improve recall — the exact metric it was meant to
move. Combined with the image-size, model-size, oversampling, and augmentation results, this
is strong evidence that the bottleneck cannot be fixed by tweaking the full-image model. The
next step must change the **training input** (crop / tile-based training), not the head.

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

## 3.5 Augmentation conclusion (added after V11)

V11 tested "Plan D": disabling the destructive augmentations (`mosaic=0`, `mixup=0`) and
enabling mask-aware `copy_paste=0.2`.

```text
V6  baseline (mosaic on)        ->  Mask mAP50-95 = 0.2336
V11 mosaic off + copy_paste 0.2 ->  Mask mAP50-95 = 0.2135  (clear regression, more overfitting)
```

Conclusion:

- Disabling mosaic on this small dataset removes critical regularisation and accelerates
  overfitting; the loss in regularisation outweighs the benefit of not downscaling small objects.
- The experiment confounded two changes (mosaic off + copy-paste on), so it cannot judge
  copy-paste on its own.
- Copy-paste should only be retested with **mosaic kept on** (or only partially closed via
  `close_mosaic`).

## 3.6 Architecture conclusion (added after V12)

V12 added a stride-4 (P2) small-object segment head to YOLOv8s-seg, with augmentation reverted
to the clean V6 baseline.

```text
V6  baseline (P3/P4/P5 head)        ->  Mask mAP50-95 = 0.2336  (recall 0.405)
V12 + P2 head (stride-4)            ->  Mask mAP50-95 = 0.2215* (recall 0.393)  *single-epoch spike; ~0.21 sustained
```

Conclusion:

- A P2 head does **not** break the ~0.23–0.24 plateau on full-image training.
- Most tellingly, **recall did not improve** (it fell, 0.405/0.468 → 0.393), so the extra
  high-resolution head failed at its one job: detecting more tiny lesions.
- The plateau is a property of the **full-image input**, not of the detection head. Changing
  the head is not enough; the input scale must change.

---

## 4. Recommended Next Direction

All direct improvements to the full-image YOLO approach have now been tested:

- Image size: `640` → `768` helped; `768` → `896` did not.
- Model size: YOLOv8m did not improve over YOLOv8s.
- Oversampling: Both mild (V10) and strong (V7) did not improve Mask mAP50-95.
- Augmentation (V11): disabling mosaic + adding copy-paste regressed the result (−0.020).
- Architecture (V12): a P2 small-object head did not beat the baseline and did not improve recall.

The full-image YOLO approach is plateaued, and V12 confirms the head is not the bottleneck.  
The next step must be a **fundamental change to the training input**.

### Immediate next experiment (V13 — crop / tile-based training)

With image size, model size, oversampling, augmentation, and the detection head all exhausted,
the remaining lever is the **input scale**. Crop / tile-based training is the most on-target
response to the "~78% of objects <1% of image area" finding: by training on tooth-level or
region-level crops, tiny Caries lesions occupy a much larger fraction of the input and the
model can learn finer-grained masks — exactly what the P2 head tried, and failed, to achieve
at the architecture level.

Lower-effort alternative to run first if crop tooling is not ready:

- **Clean copy-paste ablation** — keep `mosaic=1.0`, `mixup=0`, add `copy_paste=0.2–0.3`.
  Isolates copy-paste's real contribution without the mosaic-removal side effect that sank V11.

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
V11 (Plan D) did not beat it — it regressed to 0.2135.  
V12 (P2 head) did not beat it either — best 0.2215 (a single-epoch spike; ~0.21 sustained).  
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