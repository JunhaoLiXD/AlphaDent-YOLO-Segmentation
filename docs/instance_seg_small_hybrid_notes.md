# Boxless INSTANCE segmentation (center+offset) for the small classes — research notes

> **Status (2026-06-30): line CLOSED / NO-GO after a FAIR re-run.** The v17 FIX worked — the instance
> machinery is validated (center-recall 0.82, 526 instances over 83 val images, only 1 image with 0
> peaks) — but every supported-small instance AP is still **0.000** because the underlying semantic mask
> quality is far too low (val fg-mIoU ≈ 0.02) to clear the strict mask-IoU≥0.5 bar. So route B is now
> *fairly* tested: **the grouping+score machinery was NOT the bottleneck; the pixel signal (resolution)
> is.** Same wall as src/11. Closing the line; the **2-model V6+V10 ensemble + multi-view TTA (LB
> 0.31753) stays production.** Full result below.

## RESULT — V17 re-run (2026-06-30): machinery validated, capped by pixel signal → NO-GO

`results/version17_results.csv` (re-run, overwrites the failed V1 all-0 table). Decode diagnostics
(from the run log) confirm the v17 FIX fully took effect:

- **Instance machinery works:** training reached **center-recall 0.82** (`loaded best fg-mIoU 0.0189 +
  c-rec 0.8178, combo 0.8367`); decode produced **526 instances over 83 images**, center-heatmap max
  mean 0.197 / max 0.415, **11.7 peaks/image, only 1/83 images with 0 peaks**. This is *not* the V1
  zero-instance failure — center+offset grouping + the learned peak score behave as designed.
- **But every caries `inst_small_AP = 0.000`, `hybrid_AP = 0.000`; hybrid aggregate ≈ 0.171** (< V6
  0.2099). Large classes intact via V6 routing (Abrasion 0.637 / Filling 0.269 / Crown 0.636).
- **Root cause = pixel/mask quality, not the instance machinery.** `val fg-mIoU` stayed ≈ **0.008–0.032**
  all run (best epoch 0.0189) — the predicted caries pixels overlap GT at only ~2%. mAP50-95 needs mask
  IoU ≥ 0.5 for a TP; none of the 526 instances clear it → AP collapses to 0 at every threshold. The
  512×1024 resize starves tiny caries of pixels (the resolution confounder src/11 already flagged).

This lands on the pre-registered **No-go** branch ("instance/score machinery was not the bottleneck →
resolution"). Note one confound left un-chased: the v17 checkpoint metric `combo = fg_miou + c_rec` is
numerically **dominated by c-rec** (0.5–0.8 vs fg-mIoU 0.01–0.03), so it selected for center recall and
picked a *low*-fg-mIoU epoch (0.0189 when ep39 had 0.0317). Re-weighting it could lift fg-mIoU back to
~0.03 ≈ src/11 — still flat / no-go — so it was judged not worth a re-train. **Line closed.**

> **Original status (2026-06-26): route B baseline BUILT (`src/12`), not yet run.** Direct successor to
> the semantic-seg hybrid (`src/11`, V16) which FAILED (supported-small AP 0.032 vs V6 0.081). This fixes
> the two structural holes that sank src/11 — connected-components instances and mean-prob confidence —
> with learned machinery, holding everything else identical. (V1 ran all-0 = a decode/checkpoint bug, not
> a fair test; the v17 FIX above made the re-run fair → the NO-GO is now real.)

## Why this line (what src/11 got wrong)

src/11 proved boxless segmentation *can* run, but lost to V6 on every supported-small class. The losses
were **not** mainly the pixels — they were the conversion from a per-pixel map to *instances + scores*:

1. **Connected components ≠ instances.** Same-class touching caries merge into one component (recall
   loss); a fragmented mask splits into two (a false-positive instance, precision loss). Caries cluster,
   so both happen often.
2. **Mean class probability is not a ranking score.** mAP sorts all predictions by confidence; a
   mean-prob proxy mis-orders TPs vs FPs (and is biased high for tiny noise blobs) → AP craters even
   when the pixels are fine.

(A third hole — multiclass argmax subtype pixel-competition — and the **resolution** confounder are real
too, but are deferred to *separate* single-variable follow-ups so this run isolates holes 1+2.)

## Route B = proper boxless instance segmentation

Add two **learned** heads on the shared U-Net and decode instances Panoptic-DeepLab-style:

- **Center heatmap** (1 ch): a gaussian peak at each instance centroid. Peak value = the instance's
  **learned confidence** → fixes hole 2.
- **Offset-to-center** (2 ch): every foreground pixel regresses a vector to its instance's centroid. At
  decode, each fg pixel votes `(x+dx, y+dy)` and is assigned to the nearest detected center → groups
  pixels into instances *without* connectivity → fixes hole 1 (touching lesions split by voting to
  different centers).
- Semantic head (N_SEM ch): unchanged from src/11; gives each instance its class by majority vote.

## `src/12` baseline (built)

- Self-contained Kaggle notebook (20 cells). Inputs: training dataset (`yolo_seg_train.yaml`, train+val
  images **and** labels) + the **V6** detector.
- One `U-Net(resnet18, imagenet)` with `classes = N_SEM + 3`, split into semantic / center / offset.
- Losses: weighted CE (semantic, `BG_WEIGHT=0.2`) + **CenterNet penalty-reduced focal** (center heatmap)
  + masked **L1** (offset). Targets built from the GT polygons (per-instance rasterize → centroid →
  gaussian + offset field); hflip applied to polygons first so targets stay consistent.
- Decode (§6): max-pool-NMS peaks → offset-voting assignment → per-instance polygon + peak-score.
- Large classes → V6 (unchanged). Scored with the **same comparable Mask mAP as src/03/04/09/11**.

### Single-variable discipline (held identical to src/11)

Multiclass semantic head, fixed `512×1024` resize, `BG_WEIGHT=0.2`, the comparable metric, LARGE→V6.
**The only change vs src/11 is the instance extraction + the score.** So the headline delta is
attributable to route B's core mechanism, not to resolution / loss / routing.

### Decisions flagged to the user at build time

- **Grouping = center+offset** (pure-PyTorch extension of the existing U-Net, learned score for free),
  not StarDist (stronger domain fit for small convex blobs, but adds a dependency / reimpl) — StarDist is
  the fallback if center+offset under-splits.
- **Center loss = CenterNet penalty-reduced focal**, not plain MSE (the heatmap is sparse → MSE collapses
  to zero).
- **Checkpoint metric = val fg-mIoU** (semantic proxy, same as src/11, for a clean comparison). It does
  **not** measure instance quality — open limitation; a center-detection AP proxy is the natural upgrade
  if the result lands borderline.

### Knobs (§2)

`HEATMAP_SIGMA`, `OFFSET_NORM`, `PEAK_THRESH`, `PEAK_NMS_KERNEL`, `LAMBDA_CENTER`, `LAMBDA_OFFSET`,
`MIN_COMPONENT_PX` (+ all the src/11 knobs).

## Pre-registered reading

- **Headline = supported-small instance-seg AP** (caries 1/2/3/5) vs **both** src/11 semseg (0.032) and
  V6 (0.081). The §8 table prints `inst − semseg` and `inst − V6` directly.
- **Go**: `inst` clearly beats src/11 *and* V6 (>~0.003) → the grouping+score fix was the bottleneck;
  refine (binary+subtype, higher res/tiling, center-AP checkpoint, TTA) + submission path.
- **Partial**: `inst` beats src/11 but not V6 → machinery helped, pixel signal (resolution) still caps →
  next lever is resolution/tiling.
- **No-go**: `inst` flat vs src/11 → the instance/score machinery was not the bottleneck → resolution, or
  stop and keep the 2-model ensemble (LB 0.31753).
- Secondary = hybrid aggregate (9 classes) vs V6 0.2099 — conservative (small classes are low-weight).

## Outputs

`instance_seg_hybrid_baseline.csv` (per-class inst AP + hybrid AP + src/11 & V6 refs),
`instance_seg_small_baseline.pt` (the trained center+offset segmenter). Save the CSV as
`results/versionN_results.csv` for the durable record.
