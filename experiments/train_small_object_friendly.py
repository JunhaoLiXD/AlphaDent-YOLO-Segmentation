"""
experiments/train_small_object_friendly.py
-------------------------------------------
Training template with small-object-friendly augmentation settings (Priority 2).

DO NOT run this unless you explicitly decide to retrain.
This is a configurable template, not an automatic pipeline.

Key augmentation changes vs V10 baseline:
  mosaic=0.0      Mosaic randomly downscales objects; disabling it keeps
                  small Caries lesions at their natural size.
  mixup=0.0       Mixup blurs boundaries between overlapping objects; not
                  helpful for tiny lesions with fine mask boundaries.
  copy_paste=0.2  Copy-paste pastes objects (with their masks) from other
                  images into the current image.  For tiny Caries lesions
                  this is a cheap form of synthetic augmentation that can
                  improve small-object recall without retraining the whole
                  pipeline.

Run a dry-run first to print config without training:
    python experiments/train_small_object_friendly.py --dry-run

Run training (example):
    python experiments/train_small_object_friendly.py \\
        --model yolov8s-seg.pt \\
        --data  /kaggle/input/.../yolo_seg_train.yaml \\
        --imgsz 768 \\
        --epochs 60 \\
        --batch  16 \\
        --device 0 \\
        --name   v11_copy_paste02_nomosaic
"""

import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
DEFAULT_MODEL   = "yolov8s-seg.pt"
DEFAULT_DATA    = None           # auto-detected from /kaggle/input if omitted
DEFAULT_PROJECT = "runs/train"
DEFAULT_NAME    = "v11_copy_paste_nomosaic"
# ---------------------------------------------------------------------------


def _find_data_yaml():
    import yaml
    kaggle_input = Path("/kaggle/input")
    candidates = sorted(kaggle_input.rglob("yolo_seg_train.yaml")) if kaggle_input.exists() else []
    if not candidates:
        raise FileNotFoundError(
            "yolo_seg_train.yaml not found under /kaggle/input. "
            "Pass --data explicitly."
        )
    return candidates[0]


def parse_args():
    p = argparse.ArgumentParser(
        description="Small-object-friendly YOLO training template (Priority 2)"
    )
    # Core
    p.add_argument("--model",    type=str, default=DEFAULT_MODEL,
                   help="Architecture name ('yolov8s-seg.pt') or path to pretrained weights.")
    p.add_argument("--data",     type=str, default=DEFAULT_DATA)
    p.add_argument("--imgsz",    type=int, default=768)
    p.add_argument("--epochs",   type=int, default=60,
                   help="Recommended: 60 for a quick ablation; 120 for a full run.")
    p.add_argument("--batch",    type=int, default=16)
    p.add_argument("--device",   type=str, default="0")
    p.add_argument("--workers",  type=int, default=2)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--project",  type=str, default=DEFAULT_PROJECT)
    p.add_argument("--name",     type=str, default=DEFAULT_NAME)
    p.add_argument("--exist-ok", action="store_true", default=False)
    # Augmentation (the main point of this template)
    p.add_argument("--mosaic",     type=float, default=0.0,
                   help="Mosaic probability. 0.0 = disabled (recommended for small objects).")
    p.add_argument("--close-mosaic", type=int, default=0,
                   help="Epochs before end to disable mosaic (irrelevant if mosaic=0).")
    p.add_argument("--mixup",      type=float, default=0.0,
                   help="Mixup probability. 0.0 = disabled.")
    p.add_argument("--copy-paste", type=float, default=0.2,
                   help="Copy-paste probability (0.1-0.3 recommended for small object boost).")
    p.add_argument("--fliplr",     type=float, default=0.0,
                   help="Horizontal flip probability. Keep 0.0 for dental panoramic images.")
    # LR
    p.add_argument("--lr0",    type=float, default=0.01)
    p.add_argument("--lrf",    type=float, default=0.01)
    # Utility
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Print configuration and exit without training.")
    return p.parse_args()


def run_training(args):
    data = Path(args.data) if args.data else _find_data_yaml()

    config = {
        # ── paths ─────────────────────────────────────────────────
        "data":          str(data),
        "project":       args.project,
        "name":          args.name,
        "exist_ok":      args.exist_ok,
        # ── core ──────────────────────────────────────────────────
        "task":          "segment",
        "imgsz":         args.imgsz,
        "epochs":        args.epochs,
        "batch":         args.batch,
        "device":        args.device,
        "workers":       args.workers,
        "patience":      args.patience,
        "seed":          args.seed,
        "pretrained":    True,
        "optimizer":     "auto",
        # ── small-object augmentation ──────────────────────────────
        "mosaic":        args.mosaic,
        "close_mosaic":  args.close_mosaic,
        "mixup":         args.mixup,
        "copy_paste":    args.copy_paste,
        "fliplr":        args.fliplr,
        # ── LR ────────────────────────────────────────────────────
        "lr0":           args.lr0,
        "lrf":           args.lrf,
        # ── output ────────────────────────────────────────────────
        "save":          True,
        "plots":         True,
        "verbose":       True,
        "cache":         False,
    }

    print("=" * 65)
    print("TRAINING CONFIGURATION")
    print("=" * 65)
    print(f"  {'model':<20s}: {args.model}")
    for k, v in config.items():
        print(f"  {k:<20s}: {v}")
    print("=" * 65)

    if args.dry_run:
        print("\n[dry-run] Config printed.  Exiting without training.")
        print("Remove --dry-run to start training.")
        return

    if not data.exists():
        raise FileNotFoundError(f"data YAML not found: {data}")

    from ultralytics import YOLO
    model = YOLO(args.model)

    train_kwargs = {k: v for k, v in config.items()}
    results = model.train(**train_kwargs)

    run_dir = Path(args.project) / args.name
    print(f"\nTraining complete. Outputs saved to: {run_dir}")
    print(f"Use {run_dir}/weights/best.pt for validation/submission.")

    return results


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
