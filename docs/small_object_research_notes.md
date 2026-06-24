# Small-Object Research Notes — Two-Stage Detect-then-Refine

> **Status (2026-06-18): LINE CLOSED — NO-GO.** Phase 0 (oracle) validated the ceiling, but
> Phase 1c (retrain on real V6 boxes + background class, `src/06`) **FAILED**: every pipeline
> variant scores **below V6 0.2099** even at the perfect-FP ceiling, because the oracle's gains came
> from *perfect boxes* that a real Stage 1 cannot produce (see "Phase 1c result" below). **V6 (≈0.234)
> remains the production/submission model.** Phases: Phase 0 = `src/04`, Phase 1a/1b = `src/05`,
> Phase 1c = `src/06`. Pre-registered evaluation rules are below; they were honored at every step.
>
> **File note (2026-06-24 cleanup):** this line is closed, so its run outputs and helper tools were
> deleted — the `stage2/` folder (`stage2_results.csv`, `stage2_history.csv`, `phase1a_recall.csv`,
> `phase1b_pipeline.csv`, `phase1c_pipeline.csv`, `stage2_p1c_history.csv`) and `stage2_best.pt`, plus
> `tools/tile_yolo_seg.py` (the `untile_polygon` helper cited below) and `tools/infer_sahi_yolo_seg.py`.
> Any such file paths below are **historical** — the numbers and conclusions are kept here as the record.

---

## Motivation

The full-image YOLO approach is plateaued at ~0.234 Mask mAP50-95 and every structural attempt to
break it has failed:

- V12 (P2 small-object head) — no gain, recall did **not** improve.
- V13 (crop/tile training) — −0.11, the worst result; tiling fragments and discards the large
  objects that carry most of the per-class-averaged mAP.

**Key reframing from V13 (must stay front of mind): mAP weight ≠ object count.** The "~78% of
objects occupy <1% of the image" figure is the *object-count* distribution. mAP is averaged
per class and is carried by the large/common classes (Abrasion ~0.65, Crown ~0.63), not by the
rare tiny Caries (low AP, single-digit support). So:

- A small-object method should be expected to lift **small-Caries recall / AP**, **not** to break
  the aggregate plateau.
- Any method that sacrifices the large classes (as tiling did) backfires.

The two-stage idea below is the "smarter tiling": it tries to give tiny lesions their pixels back
**without** touching the large-object path.

---

## Phase 0 result (2026-06-17, oracle = perfect GT boxes, 30 epochs)

Stage 2 = U-Net + ImageNet-pretrained ResNet18 encoder + classification head (class + fine mask),
trained on GT-box ROIs (`PAD_MODE=relative, PAD_FACTOR=1.5, ROI_INPUT=224`). Raw data in
`stage2/stage2_results.csv` + `stage2/stage2_history.csv`. Per-class Mask AP vs the documented
V6 (src/03, same metric):

| class | n_gt | Stage2 (oracle) | V6 | delta |
|---|---:|---:|---:|---:|
| Abrasion | 408 | 0.862 | 0.647 | +0.215 |
| Filling | 186 | 0.454 | 0.280 | +0.174 |
| Crown | 19 | 0.467 | 0.631 | **−0.165** |
| Caries 1 | 62 | 0.234 | 0.120 | **+0.115** |
| Caries 2 | 73 | 0.259 | 0.085 | **+0.175** |
| Caries 3 | 33 | 0.202 | 0.012 | **+0.191** |
| Caries 4 | 4 | 0.000 | 0.000 | 0 (noise) |
| Caries 5 | 81 | 0.329 | 0.110 | **+0.220** |
| Caries 6 | 5 | 0.000 | 0.005 | ~0 (noise) |

- **Oracle mAP50-95 = 0.312**, **Hybrid (large→V6, small→Stage2) = 0.331**, vs V6 0.210.
- Training converged by ~ep12, best val ROI mask-IoU 0.813 @ ep21 (mild overfit after; best kept).

**Verdict: the direction is validated at the ceiling level.** The pre-registered signal — small
Caries *with adequate support* (1/2/3/5, n=62/73/33/81) clearly beating V6 beyond the noise band —
is met (+0.11 to +0.22). Crown regressing (−0.165) confirms large objects must route to V6
(the Hybrid takes the per-class max).

**Two honest caveats (these define Phase 1):**
1. **Oracle = perfect recall.** GT boxes give recall = 1 for every class; part of the small-Caries
   gain is "perfect localization", not just "better refinement". The real ceiling depends on how
   many small-Caries boxes a *real* Stage-1 detector actually finds → the first Phase-1 measurement.
2. **Small classes are low-weight.** Per the mAP-weight reframing, doubling small-Caries AP moves the
   aggregate competition mAP only modestly. The value is better small-lesion detection + a modest
   hybrid lift, **not** a jump to ~0.31 in the real pipeline.

### Background / no-lesion handling (decided for Phase 1)

The Phase-0 model has **no background class** — its classifier must pick one of the 9 lesion
classes, so it can only *re-label*, never *reject*. That is fine for the oracle (every GT box has a
lesion) but **not** for Phase 1: a recall-tuned Stage 1 emits many false-positive boxes with no
lesion, and a model that cannot say "background" will hallucinate a lesion on each → precision
collapse. **Phase 1 therefore adds a background class** (classifier `num_classes + 1`, background
ROIs target an empty mask), trained on hard negatives = Stage-1 boxes that don't match any GT.

---

## Phase 1a/1b result (2026-06-17, real Stage 1 = V6, `src/05`)

Raw data in `stage2/phase1a_recall.csv` + `stage2/phase1b_pipeline.csv`.

**Phase 1a — V6 localization recall (the gate): PASSED.** Class-agnostic box recall at conf=0.05:

| class | n_gt | recall@IoU0.3 | recall@IoU0.5 |
|---|---:|---:|---:|
| Caries 1 | 62 | 0.89 | 0.74 |
| Caries 2 | 73 | 0.73 | 0.62 |
| Caries 3 | 33 | 0.58 | 0.36 |
| Caries 5 | 81 | 0.80 | 0.72 |

Recall **collapses at higher conf** (Caries 2/3/5 lose 40–60% going 0.05→0.25), so **Stage 1 must run
at conf≈0.05.** The boxes for the supported small Caries exist → Stage 1 recall is not the bottleneck.

**Phase 1b — transfer check (V6 boxes → Phase-0 `stage2_best.pt`): WEAK.** Aggregate Mask mAP:
`full@0.05 = 0.182` (below V6 0.210 — no background class to reject FPs), `TPonly@0.05 = 0.218`
(perfect-FP-rejection upper bound, ≈V6 and driven by Abrasion, not Caries). The oracle's big Caries
gains did **not** transfer: TPonly Caries 1/2 ≈ flat vs V6, Caries 3 +0.03, Caries 5 +0.05.

**Verdict:** the pre-registered "TPonly clearly beats V6 on small Caries" bar was **not cleanly met**,
but the collapse is attributable to two *fixable* things — the **GT→V6 box-framing domain gap** (the
Phase-0 model only ever saw tight GT boxes) and the **missing background class**. Both are exactly what
Phase 1c changes, so Phase 1c is run to test whether the gap is the cause (decision taken with the user,
who also fixed Stage-1 conf at 0.05). Caveat retained: small Caries are low-weight, so even a clean win
moves the aggregate modestly — the headline is the **hybrid (large→V6)** number vs 0.2099.

---

## Phase 1c result (2026-06-18, retrain on real V6 boxes + background class, `src/06`) — FAILED, NO-GO

Stage 2 retrained on V6's TRAIN-split predicted boxes at conf=0.05 (IoU≥0.5→fg, <0.3→bg, [0.3,0.5)
ignored; bg ~3:1) with a background class (`nc+1`), warm-started from `stage2_best.pt`. 30 epochs.
Raw data: `stage2/phase1c_pipeline.csv` + `stage2/stage2_p1c_history.csv`.

**Training (`stage2_p1c_history.csv`):** `cls` loss 0.77→0.05 (converged, no crash), but the proxy
metrics sit **below the Phase-0 oracle** — `val_acc` ~0.70–0.73 (9 classes + bg), `val_fg_mask_iou`
~0.79 (oracle reached 0.813). Lower-quality input (real V6 boxes vs tight GT boxes) caps Stage 2's
refinement from the start.

**Aggregate Mask mAP50-95 (9 classes, same metric, V6 = 0.2099):**

| variant | what it is | aggregate mAP | vs V6 |
|---|---|---:|---:|
| `full@0.05` | all V6 boxes incl. FP (headline) | 0.157 | −0.053 |
| `full@0.25` | higher conf | 0.146 | −0.063 |
| `TPonly@0.05` | only boxes matching a GT = perfect FP rejection (ceiling) | 0.178 | −0.032 |
| hybrid (derived) | large→V6 + TPonly Caries (ceiling) | ≈0.203 | −0.007 |
| hybrid (derived) | large→V6 + full@0.05 Caries (realistic) | ≈0.196 | −0.014 |

**Every variant is below V6, including the perfect-FP-rejection ceiling.** The hybrid rows are derived
from the per-class table (V6's large classes Abrasion/Filling/Crown + Stage-2 Caries); both fall inside
or below the noise band relative to V6, so the **go/no-go (hybrid > V6 + 0.003) is NOT met → NO-GO.**

**The decisive failure — the oracle's Caries gains evaporated on real boxes, even at the ceiling:**

| class | Phase 0 oracle | Phase 1c `TPonly@0.05` | V6 |
|---|---:|---:|---:|
| Caries 1 | 0.234 | 0.079 | **0.120** |
| Caries 2 | 0.259 | 0.061 | **0.085** |
| Caries 3 | 0.202 | 0.018 | 0.012 |
| Caries 5 | 0.329 | 0.107 | **0.110** |

The +0.11..+0.22 oracle headroom is **gone** — Stage 2 on real Caries boxes only matches or trails V6.
Crown also collapsed (`TPonly` 0.368 vs V6 0.631), but Crown is "large" so the hybrid routes it to V6;
it is not the deciding factor.

**Diagnosis — the entire oracle→real gap is Stage-1 box quality, not Stage-2 capability.** `TPonly`
removes false positives completely and still ≈V6, so the missing-background-class issue is *not* the
binding constraint (the bg class did help: `full` 0.157 → `TPonly` 0.178 shows FPs cost ~0.02). The
binding constraint is the **recall-vs-localization tension**: Phase 1a showed Stage 1 must run at
conf≈0.05 to recall small Caries, but those boxes are loose (recall@IoU0.5 well below @IoU0.3), so the
ROI is mis-framed (off-center / wrong scale / clipped) versus the tight GT boxes the oracle enjoyed.
Reaching the oracle needs near-perfect boxes; a real detector at this object size cannot give them, and
making the detector better at tiny-object localization IS the plateaued problem (V11/V12/V13).

**Conclusion: the two-stage detect-then-refine line is CLOSED.** The ceiling is real but unreachable
with a real Stage 1. V6 (≈0.234) remains the production model. Next direction pivots off small objects
to all-class / capacity levers (TTA, V6+V10 ensemble, larger backbone) — see the recommended-order
update below and `docs/AlphaDent_training_summary_EN.md`. **Optional closure diagnostic** (not yet run):
bin V6 TP boxes by IoU-with-GT and plot Stage-2 Caries AP per bin — it should climb toward the oracle
only in the IoU≳0.8 bin, formally proving box quality is the sole lever.

---

## The core idea — detect at a reasonable resolution, refine at native resolution

Tiny lesions are only ~10–20 px after a panoramic image is downscaled to the detector's input, so
the detector has no signal (this is exactly why V12's P2 head and V13's tiling did not recover
recall). Instead of changing the detector's input globally, refine **locally**:

1. **Stage 1 (existing model):** run the current YOLO (V6) at a reasonable resolution to **find
   locations** (boxes). Tune Stage 1 for **recall**, not precision — push conf low and let Stage 2
   filter false positives. Optionally SAHI-assist the small-object path; the large-object path stays
   full-image.
2. **Map boxes back to the original full-resolution image.**
3. **Route by box size:**
   - **Large boxes** (Abrasion, Crown, large Filling) → **trust Stage 1 directly.** Do *not* crop +
     resize them — they are already accurate full-image, and cropping/shrinking only loses context.
     This is the explicit guard against repeating the V13 large-object collapse.
   - **Small boxes** → crop the ROI from the **original high-resolution** image (with context
     padding, see below), upscale to a fixed input size, and send to Stage 2.
4. **Stage 2 (new, trained):** on the upscaled ROI, do **classification + fine segmentation**
   (see "Classify vs segment" — classification alone is not enough for this metric).
5. **Map Stage 2 masks back to full-image coordinates, merge (NMS), evaluate.** The coordinate
   bookkeeping is the same family as `tools/tile_yolo_seg.py::untile_polygon`.

---

## Phase 0 FIRST — the oracle upper-bound experiment

**Do this before building the real pipeline.** It is the diagnostic step V12/V13 lacked.

- Use the **validation-set ground-truth boxes as a "perfect Stage 1"** (NOT the real model).
- Crop ROIs from the original image, train/run Stage 2 on them, and measure the small-Caries gain.
- **Purpose — decouple two failure modes:**
  - if even with perfect boxes Stage 2 cannot help small Caries → the whole pipeline is not worth
    building (the real Stage 1 will only be worse);
  - if it helps → the hard remaining problem is "can the real Stage 1 actually find the small
    boxes" (the recall ceiling), and that becomes the next thing to attack.
- Because Phase 0 uses GT boxes, its result is an **upper bound**. The real pipeline (Phase 1) will
  be lower once Stage 1 recall is factored in — so any "continue / stop" bar must be set with that
  haircut in mind.

**Phase 1** is the real pipeline (existing model as Stage 1 → trained Stage 2), built only if
Phase 0 clears the bar.

---

## Design decisions already made in discussion

### Stage 2 model — pretraining matters more than architecture family
- **First baseline: an ImageNet-pretrained lightweight CNN** (e.g. ResNet18 / EfficientNet-B0).
- Rationale: ViT/Transformers are data-hungry; on this small dataset a *from-scratch* ViT will
  likely lose to a pretrained lightweight CNN. The win we are testing for is "high-res ROI +
  context", not a specific architecture. Prove the gain exists with a pretrained CNN first; only
  then compare CNN vs Transformer (and if Transformer, use a *pretrained* small one like DeiT-tiny).

### ROI cropping — GT crop + padding ablation (context vs resolution)
- Test whether small-lesion recognition depends on **surrounding context** by cropping
  `box + padding` at several settings and comparing.
- **Hidden trade-off to keep explicit:** if `box+padding` is resized to a fixed input (e.g. 224²),
  larger padding makes the lesion occupy a *smaller* fraction of the input — padding competes with
  the very resolution gain we are after. The ablation is really a **context-vs-effective-resolution
  curve**, not "more is better".
- Variants to run:
  - **tight (0.0× padding)** — required baseline, shows whether context helps at all;
  - **relative padding 1.5× / 2.0×** — note relative padding adds a huge background patch to large
    boxes and almost nothing to tiny boxes;
  - **absolute margin** (e.g. +32 / +64 px outside the box) — fairer for tiny objects than relative.
- Handle **aspect ratio**: crop a square around the box center or letterbox; avoid distorting the
  lesion by squashing a non-square crop into a square input.

### Classify vs segment — this is a segmentation competition
- The metric is **Mask mAP50-95**. If Stage 2 only **classifies**, the masks must come from
  somewhere else (e.g. Stage 1's YOLO mask) and mask quality is unchanged → little gain on the
  metric. Leaning toward **Stage 2 = classification + fine segmentation** on the high-res ROI so
  the small-lesion *mask* actually improves. Confirm during Phase 0.

---

## Evaluation discipline (pre-registered)

These rules are fixed *before* running so a noisy number cannot retroactively justify "continue".

1. **Decision metric = the comparable full-image metric** (the `src/03-alphadent-val-map-eval.ipynb`
   self-contained mask-mAP, or at least per-class AP on the small Caries). **NOT** ROI-level
   classification accuracy — ROI accuracy is the same inflated, non-comparable trap as V13's
   tiled-val mAP (objects fill the crop → easy task).
2. **Only judge on small classes with adequate support.** Caries 1 (n=62), Caries 2 (n=73),
   Caries 5 (n=81) are usable. Caries 4 (n=4) and Caries 6 (n=5) are statistically meaningless on
   their own — one object is 20–25% — so look at them only as a trend, never as the deciding signal.
   (Recall the project noise band: Mask mAP differences under ~0.003 are noise.)
3. **Define the threshold precisely when set:** absolute vs relative AP change, on which class set,
   on the comparable metric. *(Left open for now — per the decision to run first and set the bar
   after seeing Phase 0 numbers. Remember Phase 0 is an upper bound, so the real-pipeline bar should
   sit above any noise/Stage-1-haircut.)*
4. **Large classes must not regress.** Track Abrasion/Crown/Filling AP every run; the whole point of
   routing-by-size is to leave them untouched.

---

## Risks / pitfalls

- **Stage 1 recall is the ceiling.** Stage 2 can only refine what Stage 1 found; the small-box
  recall of the existing detector is exactly the weak point. Tune Stage 1 for recall; consider
  SAHI-assist on the small path.
- **ROI-accuracy trap** — see Evaluation rule 1. Direct parallel to V13.
- **Tiny-support noise** — see Evaluation rule 2.
- **Error propagation** — Stage 1 localization offset gives Stage 2 a misaligned crop; the context
  margin (padding) also helps absorb this.
- **Mask coordinate bookkeeping** — ROI→full-image mask mapping must be exact (reuse the
  `untile_polygon` round-trip discipline; it is unit-tested in `tools/tile_yolo_seg.py`).
- **Transformers are data-hungry** — hence the pretrained-lightweight-CNN-first decision.
- **Expectation management** — by the mAP-weight reframing, success means better small-Caries
  recall/AP and clinical usefulness, **not** a big jump in aggregate mAP.

---

## Recommended order

1. **Phase 0 — oracle upper bound** → DONE (`src/04-stage2-oracle-roi.ipynb`, 30 epochs). Direction
   validated: small Caries 1/2/3/5 beat V6 by +0.11..+0.22 at the oracle ceiling. See "Phase 0
   result" above. Outputs archived in `stage2/`.

2. **Phase 1a — measure the real Stage-1 ceiling FIRST (cheap, no retraining).** → implemented in
   **`src/05-stage1-recall-and-transfer.ipynb`**. Run V6 as Stage 1 on val at a low conf threshold
   and measure **per-class RECALL** (not AP), especially the small Caries (box-IoU 0.3 and 0.5). The
   oracle assumed recall = 1; this tells you how much of the +0.11..+0.22 headroom a real detector
   can actually deliver. If V6 cannot localize the small Caries boxes at all, Phase 1 is capped
   regardless of Stage 2.

3. **Phase 1b — (quick) transfer check.** → also in **`src/05`**. Feed V6's predicted boxes into the
   *current* `stage2_best.pt` as-is and score the full V6→Stage2 pipeline (`full` = all boxes;
   `TP-only` = boxes matching a GT = perfect FP rejection). Expect `full` to UNDERPERFORM V6 — the
   model saw only tight, lesion-bearing GT boxes and has **no background class**, so it cannot reject
   V6's false positives and suffers box-geometry domain shift. The informative number is `TP-only`:
   if small-Caries AP there stays well above V6, Phase 1c is justified. Diagnostic, **not** a
   go/no-go test on its own.

4. **Phase 1c — the real Phase 1: retrain Stage 2 on Stage-1 boxes + a background class.** → TRAINED
   2026-06-18 in **`src/06-stage2-phase1c-real-boxes.ipynb`** → **FAILED/NO-GO** (every variant < V6
   0.2099; see the "Phase 1c result" section above). Implemented decisions:
   - ROI training data from **V6's predicted boxes on the TRAIN split** at **conf=0.05**: box-IoU **≥0.5**
     to a GT → that lesion class + the GT mask (rasterized in the V6-box ROI frame); IoU **<0.3** →
     **background** class + empty mask; the **[0.3, 0.5) band is IGNORED** (ambiguous). Background
     subsampled to **~3:1** vs foreground.
   - **Background class** added (`num_classes`, i.e. `nc+1` outputs); warm-started from `stage2_best.pt`
     (encoder/decoder/seg head copied; classifier head grown nc→nc+1, overlapping rows copied).
     Seg loss applied to foreground only; CE (incl. background) to all.
   - Evaluates the full V6→Stage2 pipeline on val with the comparable metric (`full@0.05` = headline,
     `TPonly@0.05` = refinement ceiling) + the **hybrid (large→V6, small→Stage2)** routing.
   - **Go/no-go (set with user):** hybrid mAP > V6 (0.2099) beyond the ~0.003 noise band → integrate
     into a submission; otherwise Stage 2 stays a research result and V6 remains the submission model.
   - **OUTCOME (2026-06-18): FAILED → NO-GO.** All variants below V6 even at the perfect-FP ceiling
     (`full@0.05`=0.157, `TPonly@0.05`=0.178, hybrid≈0.203); the oracle's Caries gains evaporated on
     real boxes. The gap is **Stage-1 box quality** (recall-vs-localization tension at conf≈0.05), not
     Stage-2 capability. **Two-stage line closed; V6 stays production.** See "Phase 1c result" above.

5. **(Parallel, low effort) SAHI two-path inference** with V6 (slices for small + full image for
   large + NMS) — a cheap alternative probe of the small-object recall side; see
   `docs/AlphaDent_training_summary_EN.md` §4 and `tools/infer_sahi_yolo_seg.py`.

6. **Padding ablation** (`PAD_FACTOR` 0.0 / 1.5 / 2.0, or `PAD_MODE=absolute`) can be run on the
   oracle (cheap) to pick the ROI framing before investing in Phase 1c.

---

## Cross-references

- V12 (P2 head) and V13 (tiling) analyses, and the "mAP weight ≠ object count" reframing:
  `docs/AlphaDent_training_summary_EN.md` (§3.6, §3.7, V13 section) and the `_CN` mirror.
- Comparable full-image evaluation harness: `src/03-alphadent-val-map-eval.ipynb`.
- Tiling geometry / coordinate round-trip to reuse for ROI↔full-image mapping:
  `tools/tile_yolo_seg.py`.
- Loss-level ideas (focal / class-weighted BCE / Tversky) that target the same rare small classes:
  `docs/future_loss_modification_notes.md`.
- SAHI slice+full-image inference background (external): Akyon et al., "Slicing Aided Hyper
  Inference and Fine-tuning for Small Object Detection", arXiv:2202.06934.
