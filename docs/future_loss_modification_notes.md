# Future Loss Modification Notes

> **Status: RESEARCH NOTES — no code changes made**
> These are design notes for potential future experiments that require
> modifying the Ultralytics training loop.  None of this has been implemented.

---

## Motivation

Current best mask mAP50-95 = **0.2341** (V10). The plateau has since been confirmed
across V6–V13: augmentation changes (V11, −0.020), a P2 small-object head (V12, no gain),
and crop/tile training (V13, −0.11) all failed to beat it.

> **Important caveat from V13 (mAP weight ≠ object count).** Error analysis found ~78% of
> objects occupy <1% of the image, but that is the *object-count* distribution, not the
> *mAP-weight* distribution. mAP is averaged per class and is carried by the large/common
> classes (Abrasion, Crown), not the rare tiny Caries — whose AP is low and whose support is
> single-digit. So pushing the model toward rare Caries (the goal of the losses below) is
> unlikely to move the aggregate much on its own, and must not come at the cost of the large
> classes. Treat these as small-object recall experiments, not as plateau-breakers.

Standard cross-entropy + binary mask BCE (Ultralytics defaults) treat every
pixel and every class equally.  The following loss modifications aim to push
the model toward harder, less-frequent lesion types.

---

## Option A — Focal Loss for Classification Head

### What it does
Standard BCE for class confidence becomes:

```
FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)
```

The `(1 − p_t)^γ` factor down-weights easy (high-confidence) examples and
focuses gradient on hard (low-confidence) ones.  This is most useful when
the dataset has many easy negatives (background) dwarfing rare positives.

### Hyperparameters to try
| Parameter | Range | Notes |
|-----------|-------|-------|
| `gamma`   | 1.5–2.5 | Higher = more focus on hard examples |
| `alpha`   | 0.25–0.5 | Weight for positive class; tune with class freq |

### Implementation path in Ultralytics
The classification loss is computed in
`ultralytics/utils/loss.py` → `BboxLoss` / `v8SegmentationLoss`.
Override `__call__` in a subclass or monkey-patch after import.
No public API hook yet (as of Ultralytics 8.x).

### Risk level
Medium — focal loss can destabilize training if γ is too high.  Start with
γ=1.5 and verify val loss converges before pushing to 2.5.

---

## Option B — Class-Weighted Mask BCE

### What it does
Multiply mask BCE loss for rare-class pixels by a weight > 1.  This is
equivalent to oversampling at the loss level rather than at the data level.

```
L_mask = Σ_c  w_c · BCE(pred_mask_c, gt_mask_c)
```

Proposed weights for AlphaDent (rough relative frequencies from training data):

| Class | Frequency bucket | Suggested weight |
|-------|-----------------|-----------------|
| Crown, Implant, Filling | Common | 1.0 |
| Caries 1, Caries 2 | Moderate | 1.5 |
| Caries 3, Caries 4 | Rare | 3.0 |
| Caries 5, Caries 6 | Very rare | 5.0 |

> Weights are starting points.  Derive final values from class pixel counts
> in the training set to make them principled.

### Implementation path
Same loss file as Option A.  In `v8SegmentationLoss`, scale `lseg`
(the per-class mask loss tensor) by the weight vector before summing.

### Risk level
Low-to-medium.  Class-weighted BCE is well-understood.  Main risk: if the
weights are too aggressive, the model may over-predict rare classes and
hurt mAP on common classes (Crown, Implant).

---

## Option C — Tversky Loss for Mask Segmentation

### What it does
A generalization of Dice loss that allows separate control of false-positive
and false-negative penalties:

```
Tversky(α, β) = TP / (TP + α·FP + β·FN)
L_tversky = 1 − Tversky
```

Setting α < β (e.g., α=0.3, β=0.7) penalizes false negatives more than
false positives — useful when recall for rare lesions matters more than
precision.

### Implementation path
Replace or augment the mask BCE in `v8SegmentationLoss` with Tversky.
Can be added as an additional term: `L_total = L_bce + λ · L_tversky`.

### Risk level
Medium-high — Tversky loss is sensitive to initialisation and can conflict
with standard BCE.  Recommend adding as a soft regularisation term
(λ=0.1–0.3) rather than replacing BCE entirely.

---

## Recommended Experiment Order

1. **Clean copy-paste ablation first** (the V11 Plan-D template `experiments/train_small_object_friendly.py`
   was removed in the 2026-06-24 cleanup — re-derive it from `src/01`, this time with `mosaic=1.0` kept
   on) — no loss change, lower risk. NB: V11 already ran copy-paste
   *with mosaic disabled* and regressed (−0.020); the mosaic removal, not copy-paste, was the
   likely cause, so copy-paste still needs a clean test with mosaic on before any verdict.

2. **Class-weighted mask BCE** (Option B)
   — if copy-paste alone doesn't lift Caries 3-6 AP.

3. **Focal loss** (Option A)
   — if classification confidence for rare classes is consistently too low.

4. **Tversky loss** (Option C)
   — only if recall specifically (not precision) is the limiting factor after
   inspecting the per-class PR curves from `tools/val_native_yolo_seg.py`.

> Per the V13 reframing above, none of these is expected to break the ~0.23–0.24 plateau on its
> own — they target the low-weight tiny classes. The plateau-relevant lever is preserving the
> large classes (a hybrid full-image + auxiliary-tiling approach), not the loss function.

---

## Notes on Ultralytics Versioning

Loss internals changed between Ultralytics 8.0 and 8.2.  Always pin the
version in your Kaggle notebook (e.g., `pip install ultralytics==8.1.47`)
before making loss modifications so that the class names and file paths match.

Check: `ultralytics/__init__.py` → `__version__`.
