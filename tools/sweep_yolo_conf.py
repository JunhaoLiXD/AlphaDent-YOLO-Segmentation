"""
tools/sweep_yolo_conf.py
-------------------------
Sweep confidence thresholds and report mAP + precision/recall at each (Exp 1D).

Why do this?
  The submission notebook uses CONF=0.05.  That threshold was chosen
  conservatively.  Lowering it recovers more small-lesion predictions but
  may add false positives.  This sweep maps the full precision-recall
  operating curve so you can pick the threshold that best fits the
  competition metric.

Note on mAP vs conf:
  Ultralytics model.val(conf=X) sets the minimum confidence for a detection
  to be included in the NMS / PR-curve computation.  At conf=0.001 you see
  essentially the full PR curve and therefore the 'true' mAP50-95.
  At higher conf values some true-positive low-confidence detections are
  dropped, which reduces recall.  The mAP numbers in this sweep are therefore
  conf-threshold-dependent; they are NOT the same as the standard mAP (which
  always uses conf=0.001).  Use them to find the best submission threshold.

Output:
  runs/val_compare/conf_sweep/results.csv  — one row per conf value.

Usage:
    python tools/sweep_yolo_conf.py
    python tools/sweep_yolo_conf.py --conf-values 0.001 0.01 0.05 0.10 0.20 0.30
    python tools/sweep_yolo_conf.py --weights /kaggle/input/.../best.pt --imgsz 768
"""

import argparse
import csv
import json
from pathlib import Path


# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS     = None
DEFAULT_DATA        = None
DEFAULT_OUT_DIR     = "runs/val_compare/conf_sweep"
DEFAULT_CONF_VALUES = [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25]
# ---------------------------------------------------------------------------


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


def _find_data_yaml():
    import yaml as _yaml
    kaggle_input = Path("/kaggle/input")
    candidates = sorted(kaggle_input.rglob("yolo_seg_train.yaml")) if kaggle_input.exists() else []
    if not candidates:
        raise FileNotFoundError("yolo_seg_train.yaml not found. Pass --data explicitly.")
    return candidates[0]


def parse_args():
    p = argparse.ArgumentParser(description="YOLO confidence sweep (Exp 1D)")
    p.add_argument("--weights",     type=str,   default=DEFAULT_WEIGHTS)
    p.add_argument("--data",        type=str,   default=DEFAULT_DATA)
    p.add_argument("--imgsz",       type=int,   default=768)
    p.add_argument("--batch",       type=int,   default=16)
    p.add_argument("--device",      type=str,   default="0")
    p.add_argument("--half",        action="store_true", default=True)
    p.add_argument("--iou",         type=float, default=0.6,
                   help="NMS IoU threshold (held constant across the sweep).")
    p.add_argument("--out-dir",     type=str,   default=DEFAULT_OUT_DIR)
    p.add_argument("--conf-values", type=float, nargs="+",
                   default=DEFAULT_CONF_VALUES,
                   help="Confidence thresholds to sweep (space-separated).")
    return p.parse_args()


def run_sweep(args):
    from ultralytics import YOLO

    weights = Path(args.weights) if args.weights else _find_weights()
    data    = Path(args.data)    if args.data    else _find_data_yaml()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Weights : {weights}")
    print(f"Data    : {data}")
    print(f"Conf sweep: {args.conf_values}\n")

    model = YOLO(str(weights))
    rows  = []

    for conf in args.conf_values:
        tag = f"conf_{conf:.3f}".replace(".", "p")
        print(f"{'─'*50}")
        print(f"conf = {conf:.3f}")

        metrics = model.val(
            data=str(data),
            imgsz=args.imgsz,
            batch=args.batch,
            conf=conf,
            iou=args.iou,
            device=args.device,
            half=args.half and args.device != "cpu",
            plots=False,
            save_json=False,
            project=str(out_dir / "_runs"),
            name=tag,
            verbose=False,
        )

        rd = getattr(metrics, "results_dict", {})

        def g(dk, ap):
            if dk in rd:
                return round(float(rd[dk]), 4)
            try:
                obj = metrics
                for part in ap.split("."):
                    obj = getattr(obj, part)
                return round(float(obj), 4)
            except Exception:
                return None

        row = {
            "conf":          conf,
            "box_map50":     g("metrics/mAP50(B)",     "box.map50"),
            "box_map50_95":  g("metrics/mAP50-95(B)",  "box.map"),
            "box_precision": g("metrics/precision(B)",  "box.mp"),
            "box_recall":    g("metrics/recall(B)",     "box.mr"),
            "mask_map50":    g("metrics/mAP50(M)",      "seg.map50"),
            "mask_map50_95": g("metrics/mAP50-95(M)",   "seg.map"),
            "mask_precision":g("metrics/precision(M)",  "seg.mp"),
            "mask_recall":   g("metrics/recall(M)",     "seg.mr"),
        }

        # Compute F1 from precision/recall (useful for picking the best conf)
        if row["mask_precision"] and row["mask_recall"]:
            p, r = row["mask_precision"], row["mask_recall"]
            row["mask_f1"] = round(2 * p * r / (p + r) if (p + r) > 0 else 0.0, 4)
        else:
            row["mask_f1"] = None

        rows.append(row)

        print(f"  Mask mAP50-95 = {row['mask_map50_95']}  "
              f"P = {row['mask_precision']}  R = {row['mask_recall']}  "
              f"F1 = {row['mask_f1']}")

    # ── Save results CSV ─────────────────────────────────────────────────────
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ── Print summary table ──────────────────────────────────────────────────
    w = 8
    print(f"\n{'='*75}")
    print("CONFIDENCE SWEEP — SUMMARY TABLE")
    print(f"{'='*75}")
    header = f"{'conf':>{w}} | {'mask_mAP50':>10} | {'mask_mAP50-95':>13} | {'mask_P':>7} | {'mask_R':>7} | {'mask_F1':>8}"
    print(header)
    print("─" * len(header))
    for row in rows:
        print(
            f"{row['conf']:>{w}.3f} | "
            f"{str(row['mask_map50']):>10} | "
            f"{str(row['mask_map50_95']):>13} | "
            f"{str(row['mask_precision']):>7} | "
            f"{str(row['mask_recall']):>7} | "
            f"{str(row['mask_f1']):>8}"
        )
    print(f"{'='*75}")

    valid = [r for r in rows if r["mask_map50_95"] is not None]
    if valid:
        best_map  = max(valid, key=lambda r: r["mask_map50_95"])
        best_f1   = max(valid, key=lambda r: r["mask_f1"] or 0)
        print(f"\nBest Mask mAP50-95 at conf = {best_map['conf']:.3f}  "
              f"(value: {best_map['mask_map50_95']})")
        print(f"Best Mask F1       at conf = {best_f1['conf']:.3f}  "
              f"(value: {best_f1['mask_f1']})")
        print("\nTip: use the F1-optimal conf for submission if the competition")
        print("     metric rewards a precision/recall balance.")

    print(f"\nFull results saved to: {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args)
