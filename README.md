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
│   └── version10_results.csv   # V10 (latest)
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

See [`docs/AlphaDent_training_summary_EN.md`](docs/AlphaDent_training_summary_EN.md) for the full experiment log with per-version analysis, interpretation, and conclusions.

---

## Key Findings

### What helped
- Increasing image size from `640` → `768` gave the biggest single improvement (+0.036 Mask mAP50-95).

### What did not help
- Image size `768` → `896` decreased performance (V7, V8).
- Switching to the larger YOLOv8m model did not improve the result (V9).
- Both mild and strong rare Caries oversampling traded precision for recall without improving Mask mAP50-95 (V7, V10).

### Main bottleneck
Error analysis revealed that ~78% of validation objects occupy less than 1% of the image area.  
Caries classes are significantly harder than larger classes (Crown, Abrasion).  
The full-image YOLO approach appears to be plateaued at approximately 0.23–0.24 Mask mAP50-95.

---

## Recommended Next Direction

Since all straightforward improvements to full-image YOLO training have been exhausted, the next step is a **fundamental change to the training strategy**:

**Crop / tile-based training** — train on local tooth-level crops instead of full panoramic images, so that tiny Caries lesions occupy a much larger fraction of the model input.

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
