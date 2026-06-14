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
The latest architecture experiment (V12, P2 small-object head) did **not** beat this baseline.

---

## Project Structure

```
AlphaDent/
├── README.md                   # This file
├── .gitignore
├── docs/
│   ├── training_overview.md                  # Workflow and model overview
│   ├── AlphaDent_training_summary_EN.md      # Detailed experiment log (English)
│   └── AlphaDent_training_summary_CN.md      # Detailed experiment log (Chinese)
├── results/
│   ├── version5_results.csv    # Training metrics per epoch, V5
│   ├── version6_results.csv    # V6
│   ├── version7_results.csv    # V7
│   ├── version8_results.csv    # V8
│   ├── version9_results.csv    # V9
│   ├── version10_results.csv   # V10
│   ├── version11_results.csv   # V11 (Plan D, regressed)
│   └── version12_results.csv   # V12 (latest — P2 head, did not beat baseline)
└── src/
    └── 01-yolo-seg-baseline-training-alphadent.ipynb
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

\* V12's 0.2215 is a single-epoch spike (ep32); the sustained level is ~0.21. See the V12 section in the experiment log.

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

### Main bottleneck
Error analysis revealed that ~78% of validation objects occupy less than 1% of the image area.  
Caries classes are significantly harder than larger classes (Crown, Abrasion).  
The full-image YOLO approach appears to be plateaued at approximately 0.23–0.24 Mask mAP50-95.

---

## V12 Result (P2 small-object head) — did not break the plateau

V12 attacked the small-object bottleneck at the **architecture** level: a stride-4 (P2) segment head added to YOLOv8s-seg (192×192 grid at `imgsz=768` vs 96×96 at P3), with augmentation reverted to the clean V6 baseline so the P2 head was the only change.

**Result: best Mask mAP50-95 = 0.2215 @ epoch 32, but that is a single-epoch spike (ep31 = 0.1965, ep33 = 0.1946); the sustained level over the final epochs is ~0.21.** Even taking the spike at face value, this is ≈−0.013 vs the V6/V10 baseline; the sustained level is ≈−0.02. The decisive evidence is that **recall did not improve** (0.393 vs V10's 0.468) and Mask mAP50 also fell (0.394 vs 0.41+) — the extra high-resolution head did not detect more tiny lesions, which was its entire purpose.

**Conclusion: the P2 head does not break the ~0.23–0.24 plateau.** Adding a small-object head to full-image training is not the answer for this dataset.

## Next Experiment — crop / tile-based training

With image size, model size, oversampling, augmentation, and now the P2 head all exhausted on full-image training, the next step is a **fundamental change to the training input**: **crop / tile-based training** (train on local tooth-level crops so tiny lesions occupy a much larger fraction of the input). A cleaner copy-paste ablation (`mosaic=1.0` kept on, add `copy_paste=0.2–0.3`) remains a lower-effort alternative.

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
