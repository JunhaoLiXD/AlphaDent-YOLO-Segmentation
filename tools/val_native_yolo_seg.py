"""
tools/val_native_yolo_seg.py
----------------------------
Run native YOLO segmentation validation and save a full metrics report.

This is Experiment 1A: the clean baseline that every other experiment is
compared against.  It mirrors what Ultralytics does internally during
training, so the numbers are directly comparable to what you see in
results.csv.

Usage (Kaggle or local):
    python tools/val_native_yolo_seg.py
    python tools/val_native_yolo_seg.py --weights /kaggle/input/.../best.pt
    python tools/val_native_yolo_seg.py --weights best.pt --data data.yaml --name clahe_val
"""

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Path defaults — override with CLI args when running in a new environment.
# The script first tries to auto-detect paths using the same search strategy
# as the training notebook.  Set these only when auto-detection fails.
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = None   # auto-detect from /kaggle/input
DEFAULT_DATA    = None   # auto-detect yolo_seg_train.yaml from /kaggle/input
DEFAULT_IMGSZ   = 768
DEFAULT_DEVICE  = "0"
DEFAULT_PROJECT = "runs/val_compare"
DEFAULT_NAME    = "native_val"
# ---------------------------------------------------------------------------


# ── Auto-detection helpers (mirrors the training notebook) ──────────────────

def find_data_yaml(override: str | None = None) -> Path:
    """
    Return a Path to the YOLO dataset YAML.

    Priority:
      1. --data CLI argument (if given and the file exists)
      2. yolo_seg_train.yaml found under /kaggle/input
      3. FileNotFoundError
    """
    if override:
        p = Path(override)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified --data does not exist: {p}")

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates = sorted(kaggle_input.rglob("yolo_seg_train.yaml"))
        if candidates:
            print(f"Auto-detected data YAML: {candidates[0]}")
            return candidates[0]

    raise FileNotFoundError(
        "Could not find yolo_seg_train.yaml under /kaggle/input.\n"
        "Pass --data /path/to/your.yaml explicitly."
    )


def find_weights(override: str | None = None) -> Path:
    """
    Return a Path to the model weights.

    Priority:
      1. --weights CLI argument (if given and the file exists)
      2. best.pt files ranked by path score under /kaggle/input
      3. FileNotFoundError
    """
    if override:
        p = Path(override)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified --weights does not exist: {p}")

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates = sorted(kaggle_input.rglob("*.pt"))
        if not candidates:
            raise FileNotFoundError("No .pt files found under /kaggle/input.")

        def score(p: Path) -> int:
            s = 0
            text = str(p).lower()
            for kw in ["yolov8s", "img768", "best"]:
                if kw in text:
                    s += 10
            if p.name == "best.pt":
                s += 20
            if "last" in p.name:
                s -= 5
            return s

        candidates.sort(key=score, reverse=True)
        print(f"Auto-detected weights: {candidates[0]}")
        return candidates[0]

    raise FileNotFoundError(
        "Could not find any .pt weights under /kaggle/input.\n"
        "Pass --weights /path/to/best.pt explicitly."
    )


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Native YOLO segmentation validation (Exp 1A)")
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                   help="Path to model weights (.pt). Auto-detected if omitted.")
    p.add_argument("--data",    type=str, default=DEFAULT_DATA,
                   help="Path to YOLO data.yaml. Auto-detected if omitted.")
    p.add_argument("--imgsz",   type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--device",  type=str, default=DEFAULT_DEVICE,
                   help="Device: '0', '0,1', 'cpu'")
    p.add_argument("--half",    action="store_true", default=True,
                   help="FP16 inference on GPU")
    p.add_argument("--conf",    type=float, default=0.001,
                   help="Confidence threshold. Keep at 0.001 for standard mAP.")
    p.add_argument("--iou",     type=float, default=0.6,
                   help="NMS IoU threshold.")
    p.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    p.add_argument("--name",    type=str, default=DEFAULT_NAME)
    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────────────

def run_val(args):
    from ultralytics import YOLO

    weights = find_weights(args.weights)
    data    = find_data_yaml(args.data)

    print(f"\nWeights : {weights}")
    print(f"Data    : {data}")
    print(f"imgsz   : {args.imgsz}")
    print(f"device  : {args.device}")
    print(f"project : {args.project}/{args.name}\n")

    model = YOLO(str(weights))

    metrics = model.val(
        data=str(data),
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        half=args.half and args.device != "cpu",
        plots=True,
        save_json=True,
        project=args.project,
        name=args.name,
        verbose=True,
    )

    # ── Extract summary metrics ──────────────────────────────────────────────
    # Use results_dict first (stable across Ultralytics versions), fall back to
    # the .box / .seg attribute API.
    rd = getattr(metrics, "results_dict", {})

    def get_metric(dict_key, attr_path):
        if dict_key in rd:
            return round(float(rd[dict_key]), 4)
        try:
            obj = metrics
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            return round(float(obj), 4)
        except Exception:
            return None

    summary = {
        "weights":        str(weights),
        "data":           str(data),
        "imgsz":          args.imgsz,
        "conf":           args.conf,
        "iou":            args.iou,
        "box_map50":      get_metric("metrics/mAP50(B)",    "box.map50"),
        "box_map50_95":   get_metric("metrics/mAP50-95(B)", "box.map"),
        "mask_map50":     get_metric("metrics/mAP50(M)",    "seg.map50"),
        "mask_map50_95":  get_metric("metrics/mAP50-95(M)", "seg.map"),
        "box_precision":  get_metric("metrics/precision(B)", "box.mp"),
        "box_recall":     get_metric("metrics/recall(B)",    "box.mr"),
        "mask_precision": get_metric("metrics/precision(M)", "seg.mp"),
        "mask_recall":    get_metric("metrics/recall(M)",    "seg.mr"),
    }

    # ── Per-class mask metrics ───────────────────────────────────────────────
    try:
        class_names = metrics.names          # {0: 'Caries 1', 1: 'Crown', …}
        per_class = {}
        for i, (m50, m) in enumerate(zip(metrics.seg.ap50, metrics.seg.ap)):
            name = class_names.get(i, str(i))
            per_class[name] = {
                "mask_ap50":    round(float(m50), 4),
                "mask_ap50_95": round(float(m), 4),
            }
        summary["per_class_mask"] = per_class
    except Exception as e:
        print(f"Note: per-class extraction failed ({e}). Overall metrics are still valid.")

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Box  mAP50     : {summary['box_map50']}")
    print(f"  Box  mAP50-95  : {summary['box_map50_95']}")
    print(f"  Mask mAP50     : {summary['mask_map50']}")
    print(f"  Mask mAP50-95  : {summary['mask_map50_95']}")
    print(f"  Mask Precision : {summary['mask_precision']}")
    print(f"  Mask Recall    : {summary['mask_recall']}")
    print("=" * 60)

    if "per_class_mask" in summary:
        print("\nPer-class Mask AP50-95:")
        for cls, vals in sorted(summary["per_class_mask"].items()):
            print(f"  {cls:<22s}: {vals['mask_ap50_95']:.4f}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "metrics_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to: {out_file}")

    return summary


if __name__ == "__main__":
    args = parse_args()
    run_val(args)
