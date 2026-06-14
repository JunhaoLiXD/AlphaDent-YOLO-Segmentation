# CLAUDE.md

Guidance for working in this repository.

## Project overview

YOLOv8 **instance segmentation** for the Kaggle competition *AlphaDent: Teeth Marking* —
detect and segment dental findings (Caries 1–6, Crown, Abrasion, etc.) on panoramic X-rays.
The development metric that matters is **Mask mAP50-95** (`metrics/mAP50-95(M)` in the CSVs).

Current best: **~0.234** Mask mAP50-95 (V6 imgsz=768, and V10 ≈ tied). The full-image
approach is plateaued at ~0.23–0.24; the main bottleneck is tiny lesions (~78% of objects
occupy <1% of the image) plus rare-class imbalance and overfitting. V11 (Plan D) regressed
to 0.2135 and did **not** beat the baseline.

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
- `tools/val_native_yolo_seg.py` — Exp 1A, the canonical mAP baseline every experiment is compared against.
- `tools/make_clahe_yolo_dataset.py` (1B), `tools/infer_sahi_yolo_seg.py` (1C, visual only — no mAP), `tools/sweep_yolo_conf.py` (1D, submission threshold).
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
