# MedSAM Mask-Refinement Research Notes — keep V6 localization, replace the mask

> **Status (2026-06-23): Phase 0 RUN — NO-GO for a zero-shot blanket mask swap.** Results in
> `results/version14_results.csv` (renamed from `medsam_phase0_results.csv`; the summary JSON was
> discarded). The real pipeline `v6box_medsam@0.05` = **0.182 aggregate / 0.499 large**, vs
> `v6_native@0.05` = **0.197 / 0.494** — the large-class gain (+0.005) is inside the noise band AND
> the aggregate regressed (−0.015), so the pre-registered go/no-go **fails**. The gain is real but
> concentrated in **Abrasion alone** (+0.047); Crown regressed and the small Caries collapsed.
> Diagnosis confirms the project-wide wall: **box quality**, not MedSAM mask quality — the GT-box
> oracle reaches **0.357 aggregate / 0.693 large** (highest oracle in the project, and it even
> rescues the small Caries), but real V6 Caries boxes are too loose for SAM. See the Phase 0 Result
> section below. Decision pending: optional decoder-only fine-tune (helps the domain gap, not the
> box gap) vs close the line and pivot to all-class levers (TTA / V6+V10 ensemble / larger backbone).
> **Update:** the all-class pivot won — V6+V10 ensemble + hflip TTA reached **public LB 0.31189** (vs
> single V6 0.27047). See README §"V6+V10 ensemble" and the EN/CN logs §7.
>
> **File note (2026-06-24 cleanup):** helper files cited below were removed when the related lines
> closed — `tools/tile_yolo_seg.py` (the `untile_polygon` placement discipline) and the `stage2/`
> folder. Such paths below are **historical**; `results/version14_results.csv` (this Phase 0's table)
> is kept.

---

## Motivation — attack mask IoU on the LARGE classes, not small-object recall

The full-image YOLOv8s-seg approach is plateaued at ~0.234 Mask mAP50-95. Two things are now
established (see `docs/small_object_research_notes.md` and the V13 reframing):

1. **mAP weight ≠ object count.** mAP is per-class-averaged and carried by the large/common classes
   (V6: Abrasion ~0.65, Crown ~0.63, Filling ~0.28), **not** the rare tiny Caries. The score is
   bottlenecked by the large classes being near saturation.
2. **The two-stage detect-then-refine line is CLOSED.** Phase 1c failed because it tried to fix
   *small-Caries localization* (low weight) and depended on Stage-1 **box quality**, which a real
   detector cannot deliver for tiny objects.

This note targets a *different* lever, the one that actually carries the metric:

> **YOLO-seg's prototype masks are coarse** — the mask proto runs at `imgsz/4` (768 → 192×192).
> `mAP50-95` is IoU-strict (averaged over IoU 0.50…0.95). The large classes are **localized well by
> V6** (their boxes are accurate), so any IoU left on the table is in the **mask boundary quality**,
> not the box. Replacing the coarse YOLO mask with a high-quality mask from a promptable segmenter
> (**MedSAM**) could lift the large-class IoU in the strict 0.7–0.95 band → directly move the mAP.

**Why this is not Phase 1c again (the key distinction):**

| | Phase 1c (FAILED) | MedSAM refine (this note) |
|---|---|---|
| What it tries to fix | small-Caries **localization** | large-class **mask IoU** |
| mAP weight of the target | low (rare Caries) | **high (Abrasion/Crown/Filling)** |
| Depends on box **quality**? | yes — and tiny-object boxes are bad → it died | **no** — large-class boxes are already good, and SAM is robust to loose box prompts |
| Needs training? | yes (retrain Stage 2) | **no** (Phase 0 is zero-training) |

We are refining the masks where the boxes are *trustworthy*, which is exactly where Phase 1c was not.

---

## The pipeline — V6 does localization + class + score; MedSAM does the mask only

V6 is kept **frozen and unchanged**. Each detection becomes one MedSAM call:

| Component | Source | Trained? |
|---|---|---|
| Box (location) | V6 (`version6_best.pt`) | no — as-is |
| Class (`class_id`) | V6 | no — as-is |
| Confidence (mAP ranking) | **V6's box confidence, reused directly** | no — no separate scorer needed |
| Fine instance mask | **MedSAM**, prompted by the V6 box | no — released pretrained weights |

1. Run V6 on the (val) image → boxes + classes + confidences (the V6 mask is **discarded**).
2. For each box, prompt MedSAM with that box → one binary instance mask.
3. Place the mask back in full-image coordinates (reuse the ROI↔full-image bookkeeping discipline
   from `tools/tile_yolo_seg.py::untile_polygon` / the `src/04` crop-offset placement).
4. Keep V6's `class_id` and `confidence` for that instance; **swap in the MedSAM mask**.
5. Evaluate with the comparable `src/03` self-contained mask-mAP, per class, vs V6 re-scored on the
   same images (V6 = 0.2099 in that metric).

Two clean structural wins over the semantic-segmentation alternative:
- **No instance-separation step** — one box prompt → one instance mask.
- **No per-instance scorer to train** — V6's confidence already ranks the instances for mAP.

---

## Phase 0 FIRST — zero-training mask swap (the diagnostic)

**Build this first. It requires no training of anything.** Inputs: `version6_best.pt` + a downloaded
MedSAM checkpoint (ViT-B). It answers the single question that gates the whole idea:

> Holding V6's boxes/classes/scores fixed, does swapping the YOLO mask for a MedSAM mask **raise the
> Mask mAP50-95 of the large classes** (Abrasion/Crown/Filling) on the comparable metric?

- **Upper-bound variant (oracle prompt):** prompt MedSAM with the **GT boxes** to see the ceiling of
  "perfect localization + MedSAM mask" — decouples MedSAM mask quality from V6 box quality, the same
  oracle discipline as Stage-2 Phase 0.
- **Real variant:** prompt MedSAM with **V6's predicted boxes** (the real pipeline number).
- Report both, per class, against V6 re-scored.

If even the GT-box (oracle) MedSAM masks do **not** beat V6 on the large classes, the domain gap is
fatal and we stop (or go straight to the optional fine-tune below). If they do, the real-box variant
tells us how much survives, and Phase 0 has earned a Phase 1.

---

## Design knobs (all Phase-0, none require training)

1. **Prompt type = box.** V6 outputs boxes; box prompts are more stable than points and tolerate
   loose framing. (Optionally add the box-center point as an extra positive prompt — ablation.)
2. **Full image vs ROI crop.** SAM internally resizes the input to 1024², so a tiny lesion in a full
   panoramic image has almost no effective resolution. **Recommended: crop the box ROI (with context
   padding) from the original-resolution image and run MedSAM on the crop**, then map the mask back —
   the same context-vs-effective-resolution trade-off as the Stage-2 padding ablation
   (`PAD_FACTOR` 1.0 / 1.5 / 2.0). For large boxes this matters less; for small ones it is decisive.
3. **MedSAM multimask + selection.** SAM can emit several mask hypotheses per prompt; pick by SAM's
   own predicted IoU, or by max IoU with the discarded YOLO mask as a sanity anchor (diagnostic only).
4. **Mask post-processing.** Keep the largest connected component per prompt; optionally fill holes.
   Do **not** merge across prompts (each prompt is one instance).
5. **Routing (optional, mirrors the V13 guard).** If MedSAM helps large boxes but hurts a class,
   route that class back to the V6 mask — a per-class max, decided *after* seeing Phase 0 numbers,
   never tuned on the test set.

---

## Evaluation discipline (pre-registered, same rules as the Stage-2 notes)

1. **Decision metric = the comparable `src/03` full-image mask-mAP**, per class, vs V6 re-scored on
   the *same* images (V6 = 0.2099 there). Never MedSAM's own IoU or ROI-level numbers.
2. **The headline is the LARGE classes** (Abrasion/Crown/Filling) — that is where the mAP weight and
   the hypothesis live. Small Caries are reported but are low-weight and noisy (Caries 4 n=4 /
   Caries 6 n=5 are noise; only Caries 1/2/5 n≈62/73/81 are usable as a trend).
3. **Noise band:** treat Mask mAP50-95 differences under ~0.003 as noise.
4. **Go/no-go (set before running):** the **real-box** MedSAM pipeline mask-mAP must beat V6 0.2099
   **beyond the 0.003 band** to be worth a submission. The GT-box (oracle) number is an upper bound
   and informs the decision but does not by itself justify "continue".

---

## Risks / pitfalls

- **Domain gap.** MedSAM is trained mostly on CT/MRI/endoscopy/pathology; dental panoramic X-ray is
  partially out-of-distribution. Zero-shot masks may underperform — this is exactly what Phase 0
  measures, and the optional fine-tune below is the fallback.
- **Loose/over-confident SAM masks.** SAM tends to segment the most salient object in the box; for a
  box that contains tooth + lesion it may segment the whole tooth, not the finding. Watch the large
  classes (Crown especially) for this; the ROI crop + multimask selection mitigate it.
- **Coordinate bookkeeping.** ROI→full-image mask mapping must be exact (reuse the unit-tested
  `untile_polygon` round-trip discipline).
- **Per-instance score is borrowed, not re-estimated.** Keeping V6's confidence is the right default,
  but if MedSAM occasionally produces a great mask on a low-conf box (or vice-versa), the ranking is
  suboptimal. Acceptable for Phase 0; revisit only if it clearly costs mAP.
- **Expectation management.** The realistic prize is a modest lift on the large classes' strict-IoU
  AP — not a jump. But unlike the small-Caries effort, this is where the points actually are.

---

## Optional Phase 1 — lightweight MedSAM fine-tune (only if Phase 0's domain gap is the blocker)

If Phase 0 shows MedSAM masks are good in shape but systematically off on dental X-ray (a domain gap,
not a localization failure), fine-tune **only the mask decoder** (or LoRA adapters) on the train split
with (box prompt → GT mask) pairs. Encoder frozen, cheap, small-data-safe. Re-run the Phase-0 eval.
This is the *only* training step in the whole plan, and it is conditional on Phase 0.

---

## Phase 0 notebook — RUN, NO-GO (`src/07-medsam-mask-refine.ipynb`, 23 cells)

- **Inputs (Kaggle):** `version6_best.pt` (auto-detected by the `version6` keyword, MedSAM/stage2
  excluded so they never collide) + a MedSAM ViT-B checkpoint (`medsam_vit_b.pth`, auto-detected by
  the `medsam`/`sam_vit_b` keyword). `MANUAL_V6_PATH` / `MANUAL_MEDSAM_PATH` overrides in §4.
- Kaggle-self-contained, **evaluation-only / zero-training**; new notebook (not editing 01–06).
- MedSAM loaded into the `segment_anything` `vit_b` registry; inference = min-max-normalise the 1024²
  image → `image_encoder` → box-prompted `mask_decoder` (`multimask_output=False`). Two paths via the
  `USE_ROI_CROP` knob: ROI-crop per box (default, restores effective resolution for small lesions) or
  full-image (one encode per image).
- **Clean apples-to-apples:** every variant is scored with the SAME in-notebook matcher (reused from
  src/04: `gt_local_mask` → `iou_local` → 10 IoU thr → `ap_101`). Variants: `v6_native@{0.05,0.25}`
  (V6 boxes + V6 masks = the in-notebook baseline), `v6box_medsam@{0.05,0.25}` (V6 boxes + MedSAM
  masks = the real pipeline), `TPonly_*@0.05` (perfect FP rejection), `oracle_medsam` (GT boxes +
  MedSAM masks = ceiling). The headline is `v6box_medsam` − `v6_native` (mask is the only change),
  reported per class with a large-class (Abrasion/Crown/Filling) aggregate.
- Saves `medsam_phase0_results.csv` + `medsam_phase0_summary.json` to `/kaggle/working`; archive into
  the repo `stage2/` (or a new `medsam/`) folder alongside the other results.
- **Go/no-go:** `v6box_medsam@0.05` beats `v6_native@0.05` on the large classes beyond the ~0.003
  band (and no aggregate regression) → pursue a submission path; else consider the optional
  decoder-only fine-tune below, or stop.

---

## Phase 0 RESULT (run 2026-06-23) — NO-GO for a zero-shot blanket swap

Run config: `vit_b`, `USE_ROI_CROP=True`, `PAD_MODE=relative`, `PAD_FACTOR=1.5`, capture conf 0.05.
Outputs in `results/version14_results.csv` (per-class AP for every variant; the summary JSON was
discarded after the headline numbers were lifted into this note). All variants scored with the SAME
in-notebook matcher, so `v6box_medsam` − `v6_native` is a pure mask-only delta.

**Aggregate and large-class (Abrasion/Crown/Filling) Mask mAP50-95:**

| Variant | Aggregate (9 cls) | Large (Abr/Crown/Fill) |
|---|---:|---:|
| `v6_native@0.05` (baseline) | 0.1970 | 0.4938 |
| `v6box_medsam@0.05` (real pipeline) | **0.1822** (−0.0148) | **0.4989** (+0.0051) |
| `TPonly_v6_native@0.05` | 0.2365 | 0.5687 |
| `TPonly_v6box_medsam@0.05` (perfect-FP ceiling) | 0.2140 | 0.5740 (+0.0053) |
| `oracle_medsam` (GT boxes = ceiling) | **0.3568** | **0.6928** |

**Per-class, real pipeline (`v6_native@0.05` → `v6box_medsam@0.05`):**

| Class | n_gt | native | medsam | Δ |
|---|---:|---:|---:|---:|
| Abrasion | 408 | 0.618 | **0.665** | **+0.047** ✅ |
| Filling | 186 | 0.260 | 0.263 | +0.003 (flat) |
| Crown | 19 | 0.604 | **0.569** | −0.035 ❌ |
| Caries 1 | 62 | 0.108 | **0.017** | −0.091 ❌ |
| Caries 2 | 73 | 0.080 | **0.017** | −0.064 ❌ |
| Caries 3 | 33 | 0.013 | 0.003 | −0.010 |
| Caries 5 | 81 | 0.091 | 0.106 | +0.015 |

(Caries 4 n=4 / Caries 6 n=5 are 0/noise in every real variant.)

**Verdict — the pre-registered go/no-go fails on both clauses.** The large-class gain (+0.0051) is
inside the ~0.003 noise band, AND the aggregate regressed (−0.0148). So a blanket "swap every YOLO
mask for a MedSAM mask" is **NO-GO**.

**What the result actually says (more useful than the binary):**

1. **The win is real but concentrated in Abrasion alone.** Abrasion (the largest class, n=408)
   gains +0.047 native / +0.057 TPonly — MedSAM genuinely produces sharper masks there. But Filling
   is flat and Crown *regresses* (−0.035, n=19 noisy), so "MedSAM helps large classes" overstates
   it: it helps *Abrasion*. A selective Abrasion-only hybrid (MedSAM for Abrasion, V6 for everything
   else) lifts the aggregate to ≈0.202 — still inside the noise band, because a per-class-averaged
   metric only gives one big class 1/9 weight. Not submission-worthy on its own.
2. **MedSAM destroys the small Caries masks** (Caries 1/2 collapse from ~0.1 to ~0.017). On a loose
   small box, SAM segments the most salient object — the **whole tooth**, not the lesion — so IoU
   craters. This is the predicted "SAM segments the whole tooth" risk, realised.
3. **The bottleneck is box quality, not MedSAM mask quality — same wall as the two-stage line.** The
   GT-box oracle reaches **0.357 aggregate / 0.693 large** (the highest oracle in the project) and,
   critically, *rescues the small Caries* (oracle Caries 1/2/3/5 = 0.089/0.130/0.185/0.252, vs
   ~0.01–0.02 on real boxes). So with a perfect box MedSAM segments even tiny lesions well; the
   collapse is 100 % attributable to V6's loose Caries boxes at conf≈0.05 (the recall-vs-localization
   tension from Phase 1a). Large lesions (Abrasion) have accurate-enough boxes → the real pipeline
   keeps most of the oracle gain there; small lesions do not.

**Decision (pending):**
- **Optional Phase 1 (decoder-only / LoRA fine-tune on dental):** could push Abrasion further and
  may narrow a domain gap, but it does **not** fix the binding constraint for the small classes (that
  is a *box-framing* gap, not a *mask-domain* gap — the oracle already shows MedSAM masks the lesion
  correctly when the box is right). Expect it to help the large classes only.
- **Close the line and pivot** to the all-class / capacity levers (consistent with the mAP-weight
  reframing): inference-time TTA + a V6+V10 ensemble (zero training), then a larger backbone
  (yolov8m/l-seg @ imgsz=768) as a single-variable run.
- Either way, **V6 (≈0.234) remains the production/submission model.**

---

## Cross-references

- The closed two-stage line and the mAP-weight reframing this builds on:
  `docs/small_object_research_notes.md`, README, and `docs/AlphaDent_training_summary_EN.md` §3.7.
- Comparable full-image mask-mAP harness to reuse: `src/03-alphadent-val-map-eval.ipynb`.
- ROI crop + mask placement discipline to reuse: `src/04-stage2-oracle-roi.ipynb` and
  `tools/tile_yolo_seg.py::untile_polygon`.
- MedSAM (external): Ma et al., "Segment Anything in Medical Images", Nature Communications 2024;
  base model SAM — Kirillov et al., "Segment Anything", ICCV 2023.
