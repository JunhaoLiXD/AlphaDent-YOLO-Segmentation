# Small-class semantic-segmentation hybrid — research notes

> **Status (2026-06-26): RUN & FAILED (V16, no-go).** `results/version16_results.csv`: supported-small
> (caries 1/2/3/5) semseg AP = **0.032 vs V6 0.081** (≈−60%); hybrid aggregate **0.1855 vs V6 0.2099
> (−0.024)**. Failure = **two deficits multiply** — (1) the 512×1024 resize starves tiny caries of pixels,
> and (2) the semantic→instance conversion is structurally weak (connected components conflate instances;
> mean-prob is not a ranking score; multiclass argmax makes subtypes compete per-pixel). Escaping the box
> is necessary but not sufficient. **Successor = route B (`src/12`, `docs/instance_seg_small_hybrid_notes.md`)**
> replaces the conversion with a learned center+offset instance head. V6+V10 ensemble (LB 0.31753) stays
> production. *(Original build status below kept as the design record.)*

## Why this line

All prior small-object work refined **after** a box (two-stage Stage-2, MedSAM mask swap) or tried to
tighten the box at training time (V15 NWD). They all hit the same wall: V6's tiny Caries boxes at
conf≈0.05 are **loose**, so the ROI/mask is mis-framed and the GT-box oracle gains (+0.11–0.22 on small
Caries) never transfer to the real pipeline. **Semantic segmentation classifies pixels directly and
never goes through a box** — it is the one lever that structurally escapes the box-quality wall on the
small classes.

It does **not** help the large classes (Abrasion/Crown/Filling): those carry the per-class-averaged mAP,
are near-saturated, and their YOLO boxes are already good. So the design is a **hybrid**:

- **LARGE → YOLO (V6)** — unchanged, never disturbed.
- **SMALL (caries) → semantic segmentation** — `U-Net(resnet18, imagenet)`, multiclass per-pixel.

## The metric subtlety (important)

The competition metric is **instance-seg Mask mAP50-95**. Semantic segmentation produces a per-pixel
class map, not instances + scores. So the pipeline must convert: per-pixel argmax → per-class
**connected components** → one instance each → polygon (largest external contour) + **confidence = mean
class probability over the component**. This conversion is itself a new error source (touching same-class
lesions merge; the confidence is a proxy) — a baseline limitation to revisit if the signal is positive.

## `src/11` baseline (built)

- Self-contained Kaggle notebook (20 cells). Inputs: training dataset (`yolo_seg_train.yaml`, train+val
  images **and** labels) + the **V6** detector.
- Trains a multiclass `U-Net(resnet18, imagenet)` over {background, caries...} at a fixed `512×1024`
  resize (panoramic ~2:1; normalized polygon coords are resize-invariant, so instances map straight back
  to full-image space). CE loss with background down-weighted (`BG_WEIGHT=0.2`) for the severe imbalance;
  checkpoint by **val foreground mIoU**.
- Inference → small-class instances (above). Large-class instances from V6 (large classes only).
- Scored with the **same comparable Mask mAP as src/03/04/09** (local-frame mask-IoU, 10 IoU thr, 101-pt
  AP) so the delta vs V6 is a true signal.

### Knobs (§2)

`LARGE_NAMES` (routing), `ENCODER`/`ENCODER_WTS`, `IMG_H/IMG_W`, `EPOCHS`, `BATCH_SIZE`, `LR`,
`BG_WEIGHT`, `MIN_COMPONENT_PX` (instance noise filter).

## Pre-registered reading

- **Headline = supported-small semseg AP vs V6 supported-small AP** — Caries **1/2/3/5 only** (Caries 4
  n=4 / Caries 6 n=5 are noise). Does boxless segmentation beat the loose-box detector on the small
  classes (beyond ~0.003)?
- Secondary = **hybrid aggregate (9 classes) vs V6 0.2099** — conservative; small classes are low-weight,
  so even a big small-class win moves the aggregate only modestly (the mAP-weight lesson).
- **Go**: headline clearly positive → refine (higher resolution, Dice/focal loss, better instance split,
  semseg TTA) and build a test-set submission path. **No-go**: flat/negative even as a baseline → the
  small-class headroom is genuinely capped; stop and keep the 2-model ensemble (LB 0.31753) as production.

## Outputs

`semseg_hybrid_baseline.csv` (per-class semseg-small AP + hybrid AP + V6 ref), `semseg_small_baseline.pt`
(the trained small-class segmenter, for follow-ups).
