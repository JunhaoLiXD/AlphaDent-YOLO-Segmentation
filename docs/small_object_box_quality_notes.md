# Small-Object Box-Quality Research Notes — fix the loose tiny boxes at the loss level (NWD)

> **Status (2026-06-23): V15 BUILT in `src/08-yolo-seg-nwd-training.ipynb`, not yet trained.**
> Single-variable vs V6 (full-image, `imgsz=768`, clean aug, stock `yolov8s-seg`) — the ONLY change
> is an **NWD-blended box regression loss**. Versioning note: `results/version14_results.csv` is the
> MedSAM Phase-0 eval table, so this *training* experiment is **V15** → `results/version15_results.csv`.

---

## Why this direction — the wall both closed lines hit was box quality

Two research lines are now closed, and both failed for the **same** reason:

- **Two-stage detect-then-refine** (`docs/small_object_research_notes.md`, Phase 1c) — NO-GO.
- **MedSAM mask refinement** (`docs/medsam_refine_research_notes.md`, Phase 0) — NO-GO.

In **both**, the GT-box **oracle** validated a large ceiling for the small Caries classes
(+0.11–0.22 Mask AP; MedSAM oracle even rescued tiny Caries to 0.13–0.25), but the **real pipeline
never reached it** — because V6's *small* boxes are **loose**. The diagnosis each time:

> Stage 1 must run at conf≈0.05 to recall the small Caries (Phase 1a), but at that confidence the
> boxes are loosely localized (recall@IoU0.5 ≪ recall@IoU0.3), so the ROI is mis-framed and the
> refiner / SAM gets a bad crop. **The binding constraint is box localization quality for tiny
> objects** — and "improve tiny-object localization" is exactly the plateaued problem (V11/V12/V13).

So instead of refining *after* a loose box, **fix the box itself at training time.** If the small
boxes tighten, the previously-validated oracle headroom becomes reachable — and the dev metric
(per-class-averaged Mask mAP50-95) benefits directly: 4 supported Caries classes (1/2/3/5) moving
even halfway to the oracle is worth ≈ **+0.04** on the 9-class average, larger than any single tweak
since V6.

---

## Root cause — IoU is the wrong metric for tiny boxes

This is the consensus across the recent small-object-detection literature (see References):

1. **Regression instability.** For an 8 px lesion, a 2 px shift can drop IoU from ~0.6 to ~0.2,
   while the same shift is negligible for a large box. The CIoU gradient for tiny boxes is therefore
   erratic → the boxes never tighten. This is exactly the recall@IoU0.5 ≪ recall@IoU0.3 gap Phase 1a
   measured.
2. **Label-assignment starvation.** IoU-threshold assignment gives tiny GTs **< 1 positive sample on
   average** (AI-TOD), so the detector gets almost no supervision for them.

**NWD (Normalized Gaussian Wasserstein Distance)** models each box as a 2-D Gaussian
(`μ = center`, `Σ = diag((w/2)², (h/2)²)`) and measures a normalized Wasserstein distance between
them. It is **smooth** under small localization error and does not collapse for tiny boxes, so it
fixes both problems when it replaces / blends with IoU.

---

## The V15 change — NWD blended into the box regression loss only

```
box_loss = λ · (1 − CIoU) + (1 − λ) · (1 − NWD)
```

- Large boxes keep being driven by **CIoU**; small boxes get the **stable NWD** signal.
- **Only the regression loss changes.** The DFL term, the assigner, NMS, model, data, input scale,
  and augmentation are all the **clean V6 baseline** → strict single variable, and the in-training
  Mask mAP50-95 is **directly comparable to V6 ≈0.234** (unlike V13's tiled-val number).
- This is the *minimal, lowest-risk* NWD integration. A stronger variant (future) also blends NWD
  into the `TaskAlignedAssigner` metric and NMS — deferred to keep V15 a clean one-variable run.

### NWD similarity (xyxy, stride-normalized units, as BboxLoss receives them)

```
W2² = (cx_p − cx_t)² + (cy_p − cy_t)² + ((w_p − w_t)² + (h_p − h_t)²) / 4
NWD = exp( − sqrt(W2² + eps) / C )
```

---

## Implementation — `src/08-yolo-seg-nwd-training.ipynb` (BUILT, not trained)

- **New training notebook** (per the user's preference — do NOT rewire `src/01`; one concern per
  notebook, matching `src/04`–`src/07`). Kaggle-self-contained; reuses `src/01`'s proven dataset-YAML
  location / path-check / train / output scaffolding for the **full-image V6 data** (writes a runtime
  YAML with absolute train/val paths; this run is NOT tiled).
- **The patch (section 7).** Monkey-patches `ultralytics.utils.loss.BboxLoss.forward` at the **class**
  level so the loss object the trainer builds later (inside `v8SegmentationLoss`) picks it up. It
  recomputes only the regression term (CIoU↔NWD blend) and **delegates the DFL term to the stock
  forward**, so it survives Ultralytics version drift.
- **Inputs (Kaggle):** only the original full-image AlphaDent dataset (`yolo_seg_train.yaml`). No V6
  weights needed — V15 trains from `yolov8s-seg.pt` with the V6 recipe + the new loss.

### Knobs (section 6 — the only cell to edit when sweeping)

| Knob | Meaning | Default |
|---|---|---|
| `NWD_ENABLE` | master switch; `False` = pure V6 CIoU baseline (parity check) | `True` |
| `NWD_IOU_RATIO` (λ) | weight on the CIoU term; NWD weight = `1 − λ`. `1.0` = baseline | `0.5` |
| `NWD_CONSTANT` (C) | Wasserstein normalization scale, **stride-normalized units**. **The key knob.** Too large → NWD≈1 (no effect); too small → NWD saturates to 0 (no gradient) | `5.0` |

### Gotcha fixed during bring-up — `BboxLoss.forward` signature drift
Ultralytics ≥ 8.3 appended **`imgsz, stride`** to `BboxLoss.forward` (now `self` + 9 = 10 positional
args). The first patch hard-coded 7 args → `TypeError: forward() takes 8 positional arguments but 10
were given`. Fix: the patched `forward` absorbs trailing args with **`*extra`** and passes them through
to `_ORIG_BBOX_FORWARD` for the DFL term — version-robust, no hard-coded arg count.

---

## Evaluation discipline (pre-registered)

1. **Leading indicator (box quality):** re-run `src/05` Phase-1a with V15's `best.pt` and compare
   the **small-Caries localization recall@IoU0.5** vs V6. This is the most direct "did the boxes get
   tighter" proxy and should move **before** the aggregate mAP.
2. **Headline:** best Mask `metrics/mAP50-95(M)` vs **V6 0.234** (noise band ≈ 0.003).
3. **Per-class:** Caries 1/2/3/5 (the supported small classes) should rise, AND the large classes
   (Abrasion/Crown/Filling) must **not** regress.
4. **C is unit-sensitive — judge NWD only after a small sweep, not from a single C.**

---

## Recommended order

1. *(optional, rigorous)* one **parity run** `NWD_ENABLE=False` to confirm `src/08` reproduces V6
   ≈0.234 (the notebook is new scaffolding — rules out config drift before trusting NWD deltas).
2. **Default NWD** run (λ=0.5, C=5.0).
3. **Sweep `NWD_CONSTANT` {3, 5, 8}** (one variable), then **`NWD_IOU_RATIO` {0.5, 0.7}**. Use the
   `src/05` recall@IoU0.5 leading indicator to prune, don't blind-sweep full mAP.
4. If V15 clears V6 + 0.003 (and large classes hold): stack the **assigner-level NWD** or **RFLA**
   (one variable each), and/or feed the tighter boxes back into the **MedSAM refine (`src/07`)** to
   revisit that oracle ceiling.

---

## Honest caveats

- NWD attacks **box localization**, not class weight. The **dev metric** (per-class average) benefits
  if the 4 Caries classes improve (≈+0.04 from the oracle estimate), but the **competition aggregate**
  weights small classes lightly — set expectations accordingly.
- The minimal V15 patches only the loss, not the assigner; if the loss-only change underwhelms, the
  assigner-level NWD / RFLA is the next lever, not a reason to abandon NWD.

---

## References (recent literature)

- Wang et al., *A Normalized Gaussian Wasserstein Distance for Tiny Object Detection*, arXiv
  2110.13389 — the NWD metric (label assignment + NMS + loss).
- Xu et al., *Detecting tiny objects in aerial images: a normalized Wasserstein distance and a new
  benchmark (NWD-RKA)*, ISPRS 2022 / arXiv 2206.13996 — ranking-based assignment with NWD.
- Xu et al., *RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection*,
  ECCV 2022 / arXiv 2208.08738 — assigner-level fix (the stronger follow-up lever).
- *Small Object Detection: A Comprehensive Survey*, arXiv 2503.20516 (2025); *Advancements in
  Small-Object Detection 2023–2025*, MDPI Appl. Sci. 15(22):11882 — surveys.
- Query-based bets (longer-term): DINO-DETR, Co-DETR, DQ-DETR (arXiv 2404.03507), Dome-DETR
  (arXiv 2505.05741); MaskDINO for instance segmentation.

## Cross-references
- The closed lines this builds on: `docs/small_object_research_notes.md` (two-stage),
  `docs/medsam_refine_research_notes.md` (MedSAM), README, `docs/AlphaDent_training_summary_EN.md`.
- Box-quality leading indicator harness: `src/05-stage1-recall-and-transfer.ipynb` (Phase 1a recall).
