"""
tools/infer_sahi_yolo_seg.py
-----------------------------
SAHI (Sliced Inference Helper) for small-object detection with YOLO (Exp 1C).

Why SAHI?
  Our error analysis showed ~78% of validation objects occupy <1% of image
  area.  SAHI splits each image into overlapping tiles, runs YOLO on each
  tile at full resolution, then merges predictions with NMS.  This gives
  small objects a much larger apparent size relative to the model input.

Important limitations
  - SAHI's primary integration is for detection boxes.  For segmentation
    models the boxes are returned reliably, but mask stitching across tile
    boundaries is not guaranteed.  Masks from one tile that overlap a
    neighbour tile may be incomplete.
  - This script does NOT compute mAP against ground-truth labels.
    It is for VISUAL INSPECTION only.  Use val_native_yolo_seg.py for mAP.
  - For objects that span multiple tiles SAHI keeps the detection from the
    tile with the highest score (post-NMS), so the associated mask covers
    only that tile's view of the object.

How to compare visually:
  1. Run val_native_yolo_seg.py  →  runs/val_compare/native_val/   (standard predictions)
  2. Run this script              →  runs/val_compare/sahi_val/     (sliced predictions)
  3. Open both result folders and compare predictions on the same image.
     Look for small Caries lesions that SAHI catches but the native run misses.

Requirements:
    pip install sahi ultralytics opencv-python

Usage:
    python tools/infer_sahi_yolo_seg.py
    python tools/infer_sahi_yolo_seg.py --conf 0.05 --slice-height 640 --slice-width 640
    python tools/infer_sahi_yolo_seg.py \\
        --weights /kaggle/input/.../best.pt \\
        --source  /kaggle/input/.../images/val \\
        --out-dir /kaggle/working/runs/val_compare/sahi_val
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = None   # auto-detected from /kaggle/input
DEFAULT_SOURCE  = None   # auto-detected from yolo_seg_train.yaml val split
DEFAULT_OUT_DIR = "runs/val_compare/sahi_val"
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ── Path helpers (shared with other tools) ───────────────────────────────────

def _find_weights():
    kaggle_input = Path("/kaggle/input")
    candidates = sorted(kaggle_input.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError("No .pt weights found under /kaggle/input.")

    def score(p):
        s = sum(10 for kw in ["yolov8s", "img768", "best"] if kw in str(p).lower())
        s += 20 if p.name == "best.pt" else 0
        s -= 5 if "last" in p.name else 0
        return s

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def _find_val_images():
    import yaml
    kaggle_input = Path("/kaggle/input")
    yamls = sorted(kaggle_input.rglob("yolo_seg_train.yaml"))
    if not yamls:
        raise FileNotFoundError("yolo_seg_train.yaml not found. Pass --source explicitly.")

    with open(yamls[0]) as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", yamls[0].parent))
    if not root.is_absolute():
        root = yamls[0].parent / root

    val_rel = cfg.get("val", cfg.get("valid", "images/val"))
    val_dir = (root / val_rel).resolve()
    if not val_dir.exists():
        raise FileNotFoundError(f"Val images dir not found: {val_dir}")
    return val_dir


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SAHI sliced inference (Exp 1C, visual only)")
    p.add_argument("--weights",              type=str,   default=DEFAULT_WEIGHTS)
    p.add_argument("--source",               type=str,   default=DEFAULT_SOURCE,
                   help="Folder of images to run inference on.")
    p.add_argument("--out-dir",              type=str,   default=DEFAULT_OUT_DIR)
    p.add_argument("--device",               type=str,   default="0")
    p.add_argument("--imgsz",                type=int,   default=768,
                   help="YOLO model input size applied within each tile.")
    p.add_argument("--conf",                 type=float, default=0.10,
                   help="Confidence threshold (lower = more detections on tiny objects).")
    p.add_argument("--iou",                  type=float, default=0.50,
                   help="IoU threshold for merging overlapping predictions from different tiles.")
    p.add_argument("--slice-height",         type=int,   default=512)
    p.add_argument("--slice-width",          type=int,   default=512)
    p.add_argument("--overlap-height-ratio", type=float, default=0.25)
    p.add_argument("--overlap-width-ratio",  type=float, default=0.25)
    p.add_argument("--max-images",           type=int,   default=None,
                   help="Limit to N images for a quick sanity check.")
    p.add_argument("--save-viz",             action="store_true", default=True,
                   help="Save annotated visualisation images.")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def run_sahi_inference(args):
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
    except ImportError:
        raise ImportError(
            "sahi is not installed.\n"
            "Run: pip install sahi\n"
            "On Kaggle offline mode: add sahi as a dataset wheel first."
        )

    # Resolve paths
    weights = Path(args.weights) if args.weights else _find_weights()
    source  = Path(args.source)  if args.source  else _find_val_images()
    out_dir = Path(args.out_dir)

    print(f"Weights : {weights}")
    print(f"Source  : {source}")
    print(f"Out dir : {out_dir}\n")

    viz_dir  = out_dir / "visualizations"
    json_dir = out_dir / "predictions_json"
    viz_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # Load model via SAHI's Ultralytics wrapper.
    # "ultralytics" model_type supports both YOLOv8 detection and segmentation.
    device_str = f"cuda:{args.device}" if args.device.isdigit() else args.device
    print(f"Loading model (device={device_str}) ...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights),
        confidence_threshold=args.conf,
        device=device_str,
    )

    img_paths = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.max_images:
        img_paths = img_paths[:args.max_images]

    print(f"\nImages to process : {len(img_paths)}")
    print(f"Slice size        : {args.slice_height}×{args.slice_width}")
    print(f"Overlap ratio     : {args.overlap_height_ratio}×{args.overlap_width_ratio}")
    print(f"Merge IoU         : {args.iou}\n")

    summary_rows = []
    t_total = time.time()

    for img_path in img_paths:
        print(f"  {img_path.name}", end=" ... ", flush=True)
        t0 = time.time()

        result = get_sliced_prediction(
            image=str(img_path),
            detection_model=detection_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_height_ratio,
            overlap_width_ratio=args.overlap_width_ratio,
            postprocess_type="NMS",
            postprocess_match_threshold=args.iou,
            verbose=0,
        )

        n = len(result.object_prediction_list)
        elapsed = time.time() - t0
        print(f"{n} preds  {elapsed:.1f}s")

        # Save per-image JSON
        preds = []
        for pred in result.object_prediction_list:
            preds.append({
                "category_id":   pred.category.id,
                "category_name": pred.category.name,
                "confidence":    round(pred.score.value, 4),
                "bbox_xyxy":     [round(v, 2) for v in pred.bbox.to_xyxy()],
                # mask is present when SAHI forwards it from the seg model,
                # but it only covers the tile's view of the object.
                "has_mask":      pred.mask is not None,
            })

        with open(json_dir / (img_path.stem + ".json"), "w") as f:
            json.dump(preds, f, indent=2)

        # Save visualization
        if args.save_viz:
            result.export_visuals(
                export_dir=str(viz_dir),
                file_name=img_path.stem,
                hide_labels=False,
                hide_conf=False,
            )

        summary_rows.append({
            "image":             img_path.name,
            "n_predictions":     n,
            "inference_time_s":  round(elapsed, 2),
        })

    total = time.time() - t_total

    df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "run_summary.csv"
    df.to_csv(summary_csv, index=False)

    print(f"\n{'='*60}")
    print("SAHI INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"  Images processed    : {len(df)}")
    print(f"  Total predictions   : {df['n_predictions'].sum()}")
    print(f"  Mean preds/image    : {df['n_predictions'].mean():.1f}")
    print(f"  Total time          : {total:.1f}s")
    print(f"  Mean time/image     : {total/max(len(df),1):.1f}s")
    print(f"{'='*60}")
    print(f"\nOutputs → {out_dir}")
    print(f"  Visualizations : {viz_dir}/")
    print(f"  JSON preds     : {json_dir}/")
    print(f"  Summary CSV    : {summary_csv}")
    print()
    print("NOTE: This is VISUAL INSPECTION only (no mAP computed).")
    print("      Compare these images against runs/val_compare/native_val/")
    print("      and look for small Caries lesions that SAHI catches but")
    print("      the native run misses.")


if __name__ == "__main__":
    args = parse_args()
    run_sahi_inference(args)
