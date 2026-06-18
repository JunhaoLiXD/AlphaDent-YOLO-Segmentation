# CLAUDE.md

Guidance for working in this repository.

## Project overview

YOLOv8 **instance segmentation** for the Kaggle competition *AlphaDent: Teeth Marking* —
detect and segment dental findings (Caries 1–6, Crown, Abrasion, etc.) on panoramic X-rays.
The development metric that matters is **Mask mAP50-95** (`metrics/mAP50-95(M)` in the CSVs).

Current best: **~0.234** Mask mAP50-95 (V6 imgsz=768, and V10 ≈ tied) — use V6/V10 for submissions.
The full-image approach is plateaued at ~0.23–0.24. V11 (Plan D, −0.020), V12 (P2 head, no gain),
and V13 (crop/tile training, **−0.11** — the worst result) all failed to beat the baseline. Key
reframing from V13: **mAP weight ≠ object count.** The "~78% of objects occupy <1% of the image"
finding is the object-*count* distribution, but mAP is per-class-averaged and carried by the
large/common classes (Abrasion, Crown), not the rare tiny Caries. So the score is bottlenecked by
the large classes being near saturation, *not* primarily by tiny lesions — and any small-object
effort (tiling, P2) that sacrifices the large classes backfires.

## Repository structure

```
configs/      Experimental model architectures (e.g. P2 head) — NOT yet trained
docs/         Experiment log + workflow + research notes (the project's written memory)
experiments/  Standalone training templates for a specific run (e.g. V11 copy-paste)
results/      versionN_results.csv — per-epoch Ultralytics metrics for each run
src/          Training notebook (01) + Kaggle submission notebook (02)
tools/        Validation / diagnostic scripts compared against the native baseline
```

Not in git (see `.gitignore`): datasets, images, `*.pt` weights, `runs/`.

## Important notebooks / scripts

- `src/01-yolo-seg-baseline-training-alphadent.ipynb` — trains YOLOv8-seg; produces `weights/best.pt`, `weights/last.pt`, `results.csv`.
- `src/02-alphadent-yolo-seg-submission.ipynb` — inference only; loads a checkpoint, runs on test images, writes `submission.csv` (format `id,patient_id,class_id,confidence,poly`).
- `src/03-alphadent-val-map-eval.ipynb` — evaluation only; comparable full-image (tiled+merged) Mask mAP, V13 vs V6 re-scored with the same self-contained mask-mAP code.
- `src/04-stage2-oracle-roi.ipynb` — Phase 0 oracle for the two-stage detect-then-refine plan (`docs/small_object_research_notes.md`): GT boxes as a perfect Stage 1 → high-res ROI → U-Net+pretrained-ResNet18 Stage 2 (class + fine mask) → comparable full-image Mask mAP vs documented V6. **Run; direction validated** (outputs in `stage2/`).
- `src/05-stage1-recall-and-transfer.ipynb` — Phase 1a/1b: runs V6 as a real Stage 1 and measures per-class **localization recall** (the gate, PASSED), then the transfer check (feed V6 boxes into `stage2_best.pt`, full + TP-only pipeline Mask mAP — WEAK). Needs the V6 detector + `stage2_best.pt` as Kaggle inputs. Results in `stage2/phase1a_recall.csv` / `phase1b_pipeline.csv`.
- `src/06-stage2-phase1c-real-boxes.ipynb` — Phase 1c (BUILT, not yet trained): retrain Stage 2 on **V6's predicted TRAIN boxes at conf=0.05** (IoU≥0.5→fg, <0.3→bg, [0.3,0.5)→ignore; bg subsampled ~3:1) with a **background class** (`nc+1`), warm-started from `stage2_best.pt`; evals full V6→Stage2 pipeline (`full@0.05` headline) + hybrid (large→V6) vs V6 0.2099. Same V6+`stage2_best.pt` Kaggle inputs as `src/05`.
- `tools/val_native_yolo_seg.py` — Exp 1A, the canonical mAP baseline every experiment is compared against.
- `tools/make_clahe_yolo_dataset.py` (1B), `tools/infer_sahi_yolo_seg.py` (1C, visual only — no mAP), `tools/sweep_yolo_conf.py` (1D, submission threshold).
- `tools/tile_yolo_seg.py` — V13 canonical tiling library (forward: build tiled dataset; reverse: `untile_polygon` + `merge_detections`). Mirrored inline into `src/01` (build+train) and `src/02` (tiled inference+submit) so the notebooks stay Kaggle-self-contained; keep the inline copies in sync with this file.
- `experiments/train_small_object_friendly.py` — V11 "Plan D" template (mosaic=0, mixup=0, copy_paste=0.2). Defaults to a dry-run printout; do not auto-train.
- `configs/yolov8s-seg-p2.yaml` — experimental stride-4 head; review before training.

## Training / inference workflow

1. Train (`src/01`) at `imgsz=768`, change **one** major factor vs the previous best.
2. Save the run's `results.csv` as `results/versionN_results.csv`.
3. Diagnose with `tools/val_native_yolo_seg.py` (always evaluate `best.pt`, never `last.pt`).
4. Build the submission with `src/02`; tune the confidence threshold with `sweep_yolo_conf.py`.

## Experiment / versioning conventions

- One controlled change per run; V7 (multiple changes at once) showed why this matters.
- Every meaningful run gets a `results/versionN_results.csv` AND a version section in the experiment log.
- The validation set is small: treat Mask mAP50-95 differences under ~0.003 as noise, not a result.
- Report precision **and** recall — oversampling/aug changes mostly shift the P/R trade-off without moving mAP.

## Rules for modifying notebooks

- Keep `src/01` and `src/02` clearly separated (training vs inference); don't merge concerns.
- Don't hardcode local absolute paths; the notebooks auto-detect under `/kaggle/input`.
- When changing training hyperparameters, update the relevant markdown cell so it stays accurate.

## Rules for weights, results, and large files

- Never commit `*.pt`/`*.pth`, datasets, images, or `runs/` — they are gitignored on purpose.
- Do not edit or regenerate Kaggle output folders or model checkpoints unless explicitly asked.
- `results/*.csv` are the durable record of each run — append new ones, don't rewrite old ones.

## Documentation rules

- **The experiment log lives in `docs/AlphaDent_training_summary_EN.md` (and the `_CN` mirror).**
  After any meaningful parameter / result / code change, add or update a version section there:
  what changed, why, the best-epoch metrics, the delta vs the prior best, and the conclusion.
- Keep `README.md`'s history table and "current best" in sync with the experiment log.
- Use `docs/future_loss_modification_notes.md` for unimplemented loss ideas (focal / class-weighted BCE / Tversky); mark them clearly as research notes until coded.
- Use `docs/small_object_research_notes.md` for the unimplemented two-stage detect-then-refine plan (YOLO Stage 1 → high-res ROI → trained Stage 2); also research notes until coded.

## Reminders

- Update the experiment log after every meaningful change — it is how this project remembers.
- Avoid editing large output folders, checkpoints, datasets, or generated Kaggle outputs unless explicitly asked.
- Evaluate and submit with `best.pt`, not `last.pt`.

## Project status log

### 2026-06-14 — V11 (Plan D) analysed & documented
- **Action**: Read `results/version11_results.csv` (51 epochs) and logged the result into
  `README.md`, `docs/AlphaDent_training_summary_EN.md`, and `docs/AlphaDent_training_summary_CN.md`.
- **Plan D config**: `mosaic=0`, `mixup=0`, `copy_paste=0.2` (via `experiments/train_small_object_friendly.py`).
- **Result**: best Mask mAP50-95 = **0.2135 @ epoch 42** → a clear **−0.020 regression** vs the
  V6/V10 baseline (~0.234). Both Mask mAP50 and mAP50-95 fell together (not a P/R trade-off).
- **Judgment**: the regression is driven by **disabling mosaic** — `val/seg_loss` bottoms at
  ~epoch 17 then climbs to epoch 51, i.e. overfitting got worse. Mosaic's regularisation value
  outweighed its small-object downscaling cost; `copy_paste=0.2` did not compensate. The run was
  also cut short at epoch 51 (best 42, patience would trigger at 67), but the post-peak trend was
  already deteriorating.
- **Caveat**: Plan D confounded two changes (mosaic off + copy-paste on); it does **not** prove
  copy-paste is bad.
- **Next experiment (V12)**: clean copy-paste ablation with `mosaic=1.0` kept on (`mixup=0`,
  `copy_paste=0.2–0.3`, optionally `close_mosaic=10`) to isolate copy-paste; or move to
  crop/tile-based training for the small-object bottleneck.

### 2026-06-14 — V12 (P2 head) set up in notebook 01
- **Decision**: user chose the **P2 small-object head** for V12 (their own idea), over the
  copy-paste ablation / crop-tile options. Rationale: attack the "~78% objects <1% area"
  bottleneck at the architecture level (stride-4 head → 192×192 grid at imgsz=768).
- **Action**: rewired `src/01-yolo-seg-baseline-training-alphadent.ipynb` from V11 → V12:
  - cell-0 / cell-20 / cell-25 markdown rewritten for V12 (why P2, risks, next-step focus);
  - cell-15 config: added `USE_P2_HEAD=True`, **reverted augmentation to clean V6 baseline**
    (`mosaic=1.0`, `close_mosaic=10`, `mixup=0`, `copy_paste=0`), oversampling stays off,
    `RUN_NAME` now tags `p2head`;
  - cell-21 training: writes the P2 architecture YAML **inline** to `/kaggle/working`
    (`nc` injected from dataset; self-contained — repo `configs/` not on Kaggle), builds
    `YOLO(p2_yaml).load("yolov8s-seg.pt")`, then trains with baseline aug.
- **Single-variable discipline**: architecture is the ONLY change vs V6; V11's aug changes
  are explicitly undone so V12 is not confounded.
- **Status**: implemented, **not yet trained**. Also updated README + EN/CN experiment logs.
- **When results land**: expect `results/version12_results.csv`; compare Mask mAP50-95 vs the
  ~0.234 plateau, and watch precision + `val/seg_loss` for the P2 overfitting/FP risks.
- **P2-seg gotchas fixed during bring-up** (both in notebook cell-21 + `configs/yolov8s-seg-p2.yaml`):
  1. model-YAML `scales` must be list form `s: [depth, width, max_channels]`, not a dict
     (dict → `TypeError: '<' not supported between instances of 'str' and 'int'`).
  2. `Segment` builds its mask Proto from `ch[0]`; the proto must sit on a **stride-8 (P3)**
     feature so pred masks are `imgsz/4` and match GT (`mask_ratio=4`). Putting P2 first →
     pred masks 4× too large → `mask_iou` matmul fails `(Nx27200 and 108800x300)` in
     segment/val.py. Fix: Segment inputs `[21, 24, 27, 18]` (P3,P4,P5,P2 → proto on P3).
     Re-run the YAML-writing cell (not just the train cell) for the change to take effect.

### 2026-06-14 — V12 (P2 head) trained, analysed & documented
- **Action**: Read `results/version12_results.csv` (57 epochs) and logged the result into
  `README.md`, `docs/AlphaDent_training_summary_EN.md`, and `docs/AlphaDent_training_summary_CN.md`.
- **Result**: best Mask mAP50-95 = **0.2215 @ epoch 32**, but this is a **single-epoch spike**
  (ep31=0.1965, ep33=0.1946; Box mAP50-95 also spiked to 0.251 that epoch). Sustained level over
  ep50–57 is ~0.20–0.212. True level ≈ **0.21** → ~−0.02 vs the V6/V10 baseline (~0.234); did
  **not** beat the baseline.
- **Decisive evidence P2 failed**: **recall did not improve** — Mask R = 0.393 @ best epoch, below
  V10's 0.468 and V6's 0.405. The high-resolution head was supposed to recover tiny lesions; it
  detected fewer. Mask mAP50 also fell (0.394 vs 0.41+). Not a P/R trade-off — a real quality loss.
- **Not under-training**: P2 branch starts at high loss (ep1 seg_loss ≈4.68 vs ~2.6) due to
  random init, but caught up by ep32 and produced no new peak in the next 25 epochs.
  `val/seg_loss` bottoms ~ep26 (≈2.26) then drifts 2.30–2.45 (milder overfit than V11).
- **Judgment**: the plateau is a property of the **full-image input**, not the detection head.
  Tweaking the head is not enough.
- **Next experiment (V13)**: **crop / tile-based training** (change the input scale so tiny
  lesions occupy a larger fraction of the input) — the on-target fix the P2 head failed to deliver
  at the architecture level. Lower-effort alternative first: clean copy-paste ablation with
  `mosaic=1.0` kept on.
- **Note**: `results/results.csv` is the same V12 run (live file); not deleted — flag for cleanup.

### 2026-06-14 — V13 (crop/tile training) implemented in notebooks 01 + 02
- **Decision**: user committed to the full crop/tile pipeline. Structure chosen (after a
  Kaggle data-portability discussion): **tile + train in `src/01`** (no separate tiled-dataset
  Kaggle Dataset — re-tiles in `/kaggle/working` each run, self-contained), **tiled inference +
  submission in `src/02`**. Honors the 01/02 train-vs-inference separation (tiling is train-time
  data prep).
- **New file**: `tools/tile_yolo_seg.py` — canonical tiling lib. Forward = `build_tiled_dataset`
  (slice image, Sutherland-Hodgman clip polygons to tile + renormalize, subsample empty TRAIN
  tiles, val keeps all). Reverse = `untile_polygon` + `merge_detections` (class-wise bbox-IoU NMS).
  Geometry round-trip unit-tested (exact). Mirrored inline into both notebooks.
- **`src/01` rewired V13**: dropped the P2 head (back to stock `yolov8s-seg`), inline tile-build
  cell creates the tiled dataset from `train_path`/`val_path`, trains at `imgsz=TILE_SIZE`, clean
  V6 aug, `RUN_NAME` tags `v13_tile`. Single variable vs V6 = tiled input.
- **`src/02` rewritten**: per-tile predict on numpy crops → `masks.xyn` (tile-norm) →
  `untile_polygon` → full-image-norm → `merge_detections` → submission rows. Submission format
  unchanged (`id,patient_id,class_id,confidence,poly`, full-image normalized polys); downstream
  build/sanity/save cells untouched.
- **Defaults**: TILE_SIZE=640, OVERLAP=0.20, KEEP_EMPTY=0.15 (train), MIN_AREA_FRAC=0.35. `02`
  TILE_SIZE/OVERLAP MUST match `01`.
- **Caveat baked into 01**: the val mAP reported DURING training is on the TILED val split (easier
  task) and is **not** comparable to the ~0.234 full-image baseline. The comparable number needs
  tiled+merged inference on FULL val images — deferred to a future error-analysis notebook.
- **Status**: implemented and **training in progress** (user started the V13 run). README +
  EN/CN logs updated with a V13 section (config + implementation + the tiled-val-mAP caveat),
  history tables show V13 as "training in progress / pending". Awaiting
  `results/version13_results.csv` for the result write-up.

### 2026-06-17 — V13 trained & FAILED (−0.11), comparable eval built in src/03, docs updated
- **V13 result**: run interrupted by Kaggle at ep61/120 but already converged on tiled-val (best
  ~0.217 @ ep44, flat from ep34, `val/seg_loss` rising = overfit onset) → **not resumed**.
  `results/version13_results.csv` holds the 61-epoch curve. `version13_log.txt` deleted (training
  trace, no lasting value).
- **New notebook `src/03-alphadent-val-map-eval.ipynb`** (evaluation only): tiled+merged inference
  on FULL val images for V13, AND V6 `best.pt` re-scored on the same images with the SAME
  self-contained mask-mAP code (mask-IoU → 10 IoU thr → 101-pt AP) so the V13-vs-V6 delta is a
  true signal, not a metric artifact. Finds checkpoints by filename `V13_best.pt` / `V6_best.pt`.
  `evaluate_model` returns raw (tp,conf,pcls,tcls); AP is computed in a later cell so a metric
  crash doesn't waste the ~2.5h inference.
- **Result**: **V13 Mask mAP50-95 = 0.0993 vs V6 re-scored 0.2099 → −0.1106** (worst in project).
  V6 re-scored 0.2099 vs historical 0.234 = ~0.024 metric-impl gap, identical for both → delta valid.
  **Collapse is entirely in the LARGE classes** (Abrasion −0.41, Crown −0.43, Filling −0.10); tiny
  Caries moved only ±0.01–0.02. Cause: tiling drops large objects from training (MIN_AREA_FRAC=0.35,
  they straddle borders), fragments them at inference, and merge_detections never stitches
  non-overlapping fragments.
- **Key reframing**: **mAP weight ≠ object count.** "~78% objects <1% area" is the object-COUNT
  distribution; mAP is per-class-averaged and carried by the large/common classes. The small-object
  bottleneck framing overstated the headroom → re-explains the V6 plateau and V12/V13 failures.
- **Conclusion**: naive tiling is the wrong GLOBAL strategy; **V6 (≈0.234) remains best, use it for
  submissions.** A small-object effort must not sacrifice large classes (hybrid only). Updated
  README + EN/CN logs (history rows, full V13 sections, EN/CN §3.7 input-scale conclusion, post-V13
  "next direction" rewrite) and the project memory.
- **Gotcha**: Kaggle ships **NumPy 2.x (`np.trapz` removed → `np.trapezoid`)**; `getattr(np,
  "trapezoid", np.trapz)` is itself buggy (default arg eagerly evaluated). Use
  `np.trapezoid if hasattr(np, "trapezoid") else np.trapz`.

### 2026-06-17 — Docs synced to V11–V13 + two-stage research note created
- **Doc sync**: brought the lagging files up to date with the completed V11/V12/V13 experiments
  (README/EN/CN were already current). Updated: `docs/training_overview.md` (was stuck at V10 and
  still recommended crop/tile as the next step — now covers V11–V13 + the mAP-weight reframing and
  a post-V13 next-direction rewrite), `docs/future_loss_modification_notes.md` (motivation +
  recommended order), `CLAUDE.md` project overview, `tools/tile_yolo_seg.py` and
  `experiments/train_small_object_friendly.py` docstrings (added V13/V11 OUTCOME banners),
  `configs/yolov8s-seg-p2.yaml` header (V12 outcome), and the intro/closing markdown of `src/01`,
  `src/02` (V13-failed banners; code left intact). NB found during sync: `src/02`'s saved run
  loaded the **V6** checkpoint through the **tiled** path — a train/inference mismatch, flagged in
  the notebook banner, not silently changed.
- **New planning doc `docs/small_object_research_notes.md`** (research notes, NOT implemented):
  two-stage **detect-then-refine** — existing YOLO (V6) as Stage 1 for localization (tuned for
  recall) → map boxes to the original image → crop high-res ROI for **small** boxes only (large
  boxes trust Stage 1, the V13 guard) → trained Stage 2 (**ImageNet-pretrained lightweight CNN**
  first; pretraining > architecture) does **classification + fine segmentation** → map masks back,
  merge, eval on the `src/03` comparable metric.
- **Method decisions captured**: Phase 0 **oracle upper-bound** experiment first (GT boxes as a
  perfect Stage 1) to decouple "Stage 1 recall" from "Stage 2 usefulness"; ROI **padding/margin
  ablation** (tight / 1.5×/2.0× / absolute-px) framed as a context-vs-effective-resolution
  trade-off; pre-registered eval rules (comparable metric not ROI accuracy; only judge small
  classes with adequate support — Caries 4 n=4 / Caries 6 n=5 are noise; large classes must not
  regress). The "how much gain to continue" threshold is intentionally **left open** — run first,
  set the bar after Phase 0 numbers (and remember Phase 0 is an upper bound).
- **Phase 0 notebook built**: `src/04-stage2-oracle-roi.ipynb` (Kaggle self-contained, 19 cells).
  GT boxes → on-the-fly ROI crop with padding (relative/absolute ablation knob in the §2 config) →
  `smp.Unet(encoder="resnet18", encoder_weights="imagenet", aux_params=class head)` doing
  class + binary fine mask → masks placed back by crop offset (mask analog of `untile_polygon`) →
  full-resolution Mask mAP with the `src/03` evaluator, per-class vs documented V6 (`V6_REF_AP`),
  plus a hybrid (route large→V6, small→Stage2) upper-bound. ROI-level acc/IoU printed but flagged
  diagnostic-only (ROI-accuracy trap).

### 2026-06-17 — Stage-2 Phase 0 (oracle) RUN — direction validated; docs updated
- **Run**: `src/04` 30 epochs, oracle = GT boxes (perfect Stage 1), `PAD_FACTOR=1.5`, ROI 224,
  U-Net+ResNet18(imagenet)+cls head. Outputs archived by the user in **`stage2/`**
  (`stage2_history.csv`, `stage2_results.csv`; `stage2_best.pt` saved on Kaggle, gitignored).
- **Result (per-class Mask AP, oracle vs documented V6)**: Abrasion 0.862/0.647, Filling
  0.454/0.280, Crown **0.467/0.631 (−0.165)**, Caries1 0.234/0.120 (**+0.115**), Caries2
  0.259/0.085 (**+0.175**), Caries3 0.202/0.012 (**+0.191**), Caries5 0.329/0.110 (**+0.220**);
  Caries4/6 (n=4/5) ~0 = noise. **Oracle mAP=0.312, Hybrid=0.331, V6=0.2099.** Train converged
  ~ep12, best val ROI mask-IoU 0.813@ep21 (mild later overfit; best kept).
- **Verdict**: pre-registered signal MET — small Caries with support (1/2/3/5) clearly beat V6
  beyond noise → **direction validated at the oracle ceiling.** Crown regression confirms
  large→V6 routing.
- **Two caveats that define Phase 1**: (1) oracle = perfect recall, so part of the gain is perfect
  localization not refinement → real ceiling hinges on a real detector's small-box recall;
  (2) small classes are low-weight → aggregate competition mAP rises only modestly even at best.
- **Background class decided for Phase 1**: Phase-0 model has NO background output (classifier picks
  one of 9), fine for GT boxes but it CANNOT reject a recall-tuned Stage 1's false-positive boxes
  → Phase 1 must add a background class (`nc+1`, empty-mask target) trained on hard negatives
  (Stage-1 boxes matching no GT).
- **Save-bug fixed**: the run crashed at the very end (§9) with `NameError: _norm_class_key` (used
  in the save cell, defined only in the AP cell) — harmless, results/best.pt already written. Moved
  `_norm_class_key` to the config cell (cell-5) so the table + save cell both get it regardless of
  order; removed the duplicate def/`import re` from the AP cell. (Also why the user's
  `stage2_results.csv` had blank Caries V6_ref — that run used a pre-fix table cell.)
- **Docs updated**: README (structure + `stage2/` + a Stage-2 Phase-0 subsection),
  `docs/small_object_research_notes.md` (Phase 0 result table, background-class decision, Phase 1a/b/c
  plan), and the project memory.
- **Phase 1 plan (next)**: 1a measure V6-as-Stage1 per-class RECALL at low conf (cheap, gates
  everything); 1b optional transfer check (feed V6 boxes into current `stage2_best.pt` — expect weak,
  diagnostic only); 1c real Phase 1 = retrain Stage 2 on V6's TRAIN-split predicted boxes + a
  background class, warm-started from `stage2_best.pt`, eval full pipeline + hybrid on the
  comparable metric.

### 2026-06-17 — `src/05` built (Phase 1a/1b), running
- **New notebook `src/05-stage1-recall-and-transfer.ipynb`** (eval only, Kaggle self-contained,
  19 cells, built via one-off builder then validated). §6 Phase 1a = V6-as-Stage1 per-class
  **class-agnostic localization recall** (conf {0.05,0.10,0.25} × box-IoU {0.3,0.5}); §7 Phase 1b =
  feed V6 boxes into `stage2_best.pt` (batched per image, cached once) and score the full V6→Stage2
  pipeline with the src/04 local-IoU mask-mAP, two variants: `full@conf` (all boxes, FPs included)
  and `TPonly@0.05` (only V6 boxes matching a GT = perfect FP rejection). §8 saves
  `phase1a_recall.csv` / `phase1b_pipeline.csv` / `phase1_summary.json` to `/kaggle/working`.
- **Inputs**: the V6 detector + `stage2_best.pt` as Kaggle inputs; auto-detected by filename
  (`version6_best.pt` matches the "version6" keyword; stage2 excluded so they never collide). Added
  `MANUAL_V6_PATH` / `MANUAL_S2_PATH` overrides in cell-9 as a safety net. ROI framing constants
  (`ROI_INPUT=224, PAD_MODE=relative, PAD_FACTOR=1.5`) MUST match the Phase-0 training of
  `stage2_best.pt` or the transfer check is unfair.
- **Decision rule (in §9)**: proceed to Phase 1c iff (small-Caries localization recall is non-trivial)
  AND (`TPonly` small-Caries AP clearly beats V6). `full` < V6 is expected (no background class) and
  is not a reason to abandon.

### 2026-06-17 — Phase 1a/1b analysed; `src/06` (Phase 1c) built
- **Phase 1a/1b results** (user ran `src/05`, added `stage2/phase1a_recall.csv` +
  `stage2/phase1b_pipeline.csv`):
  - **1a gate PASSED.** V6 localization recall @conf0.05: Caries 1/2/3/5 = 0.89/0.73/0.58/0.80 @IoU0.3
    (0.74/0.62/0.36/0.72 @IoU0.5). Recall drops 40–60% at conf0.25 → **Stage 1 must stay at conf≈0.05**
    (user's decision; the supported small-Caries boxes exist, recall is not the bottleneck).
  - **1b transfer WEAK / rule not cleanly met.** Aggregate `full@0.05`=0.182 (<V6 0.210),
    `TPonly@0.05`=0.218 (perfect-FP upper bound, ≈V6, carried by Abrasion +0.13 not Caries). Oracle
    Caries gains evaporated (TPonly Caries1/2 ≈flat, 3 +0.03, 5 +0.05). Cause = **GT→V6 box-framing
    domain gap** + **missing background class** — both fixable, both addressed by Phase 1c.
- **Decision (with user)**: proceed to Phase 1c despite the soft 1b signal, because the weak transfer
  is explained by the two fixable causes, not by Stage 2 being useless.
- **New notebook `src/06-stage2-phase1c-real-boxes.ipynb`** (23 cells, Kaggle self-contained, built via
  one-off `tools/_build_src06.py` then deleted; all 11 code cells compile). Retrains Stage 2 on **V6's
  TRAIN-split predicted boxes at conf=0.05**: box-IoU **≥0.5**→foreground (that GT's class + GT mask
  rasterized in the V6-box ROI frame), **<0.3**→background (`nc+1`, empty mask), **[0.3,0.5)**→IGNORED;
  background subsampled **~3:1** vs FG. **Warm-started** from `stage2_best.pt` (encoder/decoder/seg head
  copied; classifier head grown nc→nc+1, overlapping rows copied, bg row at init). Seg loss FG-only, CE
  (incl. bg) on all. Checkpoint by a combined proxy (FG mask-IoU + acc) on a 10% train-ROI holdout. §9
  scores the full V6→Stage2 pipeline (`full@0.05` headline, FP rejection via bg; `TPonly@0.05` ceiling)
  + **hybrid (large→V6, small→Stage2)** with the src/03/04 comparable mask-mAP.
- **Go/no-go (set with user)**: **hybrid mAP > V6 0.2099 beyond the ~0.003 noise band** → integrate into
  a submission; otherwise Stage 2 stays a research result and V6 remains the production model.
- **Status**: implemented, **not yet trained**. README + `docs/small_object_research_notes.md` (Phase
  1a/1b result section, Phase 1c built in recommended-order item 4, status banner) + project memory
  updated. Awaiting the Kaggle run → `phase1c_pipeline.csv` / `phase1c_summary.json`.

### 2026-06-18 — Phase 1c TRAINED & FAILED (no-go); two-stage line closed; docs updated
- **Run**: user ran `src/06` and added `stage2/phase1c_pipeline.csv` + `stage2/stage2_p1c_history.csv`.
  30 epochs; `cls` loss 0.77→0.05 (converged, no crash) but the proxy metrics are **below the oracle**:
  `val_acc` ~0.70–0.73 (9 classes + bg), `val_fg_mask_iou` ~0.79 (oracle reached 0.813). Lower input
  quality (real V6 boxes vs tight GT boxes) caps Stage 2's refinement quality from the start.
- **Result (aggregate Mask mAP50-95 over the 9 classes, same metric, V6=0.2099):**
  `full@0.05 = 0.157`, `full@0.25 = 0.146`, `TPonly@0.05 = 0.178` (perfect-FP ceiling), and the
  **hybrid (large→V6, small→Stage2)** I derived from the per-class rows ≈ **0.203** (using TPonly
  Caries, the ceiling) / ≈ **0.196** (using full@0.05 Caries, realistic). **Every variant < V6 0.2099.**
- **Decisive failure: the oracle's Caries gains evaporated on real boxes — even at the TPonly ceiling.**
  TPonly Caries vs V6: Caries1 0.079 vs 0.120, Caries2 0.061 vs 0.085, Caries5 0.107 vs 0.110, Caries3
  0.018 vs 0.012 (Caries4/6 ~0, noise). Compare the oracle (0.234/0.259/0.329/0.202) — the +0.11..+0.22
  oracle headroom is **gone** once boxes are real. Crown also collapsed (TPonly 0.368 vs V6 0.631), but
  Crown is "large" so the hybrid routes it to V6 anyway — not the deciding factor.
- **Go/no-go → NO-GO.** hybrid (≤0.203) does not clear V6 0.2099 beyond the 0.003 noise band, even at the
  optimistic ceiling. **Stage 2 stays a research result; V6 (≈0.234) remains the production/submission model.**
- **Diagnosis (confirms the user's framing).** The whole oracle→real gap is **Stage-1 box quality**, not
  Stage-2 capability or FP rejection: TPonly removes FPs entirely and still ≈V6. The cause is the
  recall-vs-localization tension — Stage 1 must run at conf≈0.05 to recall small Caries (Phase 1a), but
  those boxes are loose (Phase 1a matched recall@IoU0.5 well below @IoU0.3), so the ROI is mis-framed
  (off-center / wrong scale / clipped) vs the tight GT boxes the oracle enjoyed. You can only reach the
  oracle with near-perfect boxes, which a real detector at this object size cannot give — and improving
  the detector IS the plateaued small-object problem (V11/V12/V13). So the two-stage **detect-then-refine
  line is closed**: oracle ceiling validated, but unreachable with a real Stage 1.
- **Next direction**: pivot off small objects to **all-class / capacity levers** (consistent with the
  mAP-weight reframing): cheap first — inference-time **TTA** and a **V6+V10 ensemble** (zero training);
  then a **larger backbone (yolov8m/l-seg @ imgsz=768)** as a single-variable run. Optional closure
  diagnostic before abandoning: bin V6 TP boxes by IoU-with-GT and plot Stage-2 Caries AP per bin —
  expect it to climb toward the oracle only in the IoU≳0.8 bin (which a real detector rarely produces).
- **Docs updated**: README (Phase 1c paragraph → trained/failed), `docs/small_object_research_notes.md`
  (status banner + Phase 1c result section + recommended-order item 4 outcome), EN/CN experiment logs
  (two-stage status line), and the project memory.
