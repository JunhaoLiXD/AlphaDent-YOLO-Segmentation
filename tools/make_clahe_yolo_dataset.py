"""
tools/make_clahe_yolo_dataset.py
---------------------------------
Generate a CLAHE-enhanced copy of the YOLO validation images (Experiment 1B).

Why CLAHE?
  Dental X-rays often have low local contrast in regions where Caries lesions
  form.  CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances
  local contrast without over-amplifying noise, which can make lesion edges
  more visible to the model at inference time — at zero retraining cost.

What this script does:
  - Reads validation images from the original dataset.
  - Applies CLAHE to each image:
      grayscale  → CLAHE on the single channel directly.
      color BGR  → convert to LAB, apply CLAHE to the L channel, convert back.
  - Copies label files unchanged (segmentation polygons are image-size-independent).
  - Writes a new data_clahe_val.yaml that you can pass to val_native_yolo_seg.py.

After running this script, run:
    python tools/val_native_yolo_seg.py --data datasets/alphadent_clahe_val/data_clahe_val.yaml --name clahe_val

Usage:
    python tools/make_clahe_yolo_dataset.py
    python tools/make_clahe_yolo_dataset.py --clip-limit 3.0 --tile-size 16
    python tools/make_clahe_yolo_dataset.py \\
        --src-images /kaggle/input/competitions/alpha-dent/AlphaDent/images/val \\
        --src-labels /kaggle/input/competitions/alpha-dent/AlphaDent/labels/val \\
        --out-dir    /kaggle/working/datasets/alphadent_clahe_val
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Path defaults.
# The script first tries to auto-detect the val split from yolo_seg_train.yaml.
# Override only when auto-detection is wrong or you're running locally.
# ---------------------------------------------------------------------------
DEFAULT_SRC_IMAGES = None   # auto-detected from yolo_seg_train.yaml
DEFAULT_SRC_LABELS = None   # inferred from images path (images/ → labels/)
DEFAULT_OUT_DIR    = "/kaggle/working/datasets/alphadent_clahe_val"
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ── Path helpers ─────────────────────────────────────────────────────────────

def find_val_paths():
    """
    Find val images and labels dirs by reading yolo_seg_train.yaml.
    Returns (images_dir, labels_dir) as Path objects.
    """
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        raise FileNotFoundError(
            "Cannot auto-detect paths outside Kaggle. "
            "Pass --src-images and --src-labels explicitly."
        )

    candidates = sorted(kaggle_input.rglob("yolo_seg_train.yaml"))
    if not candidates:
        raise FileNotFoundError(
            "yolo_seg_train.yaml not found under /kaggle/input. "
            "Pass --src-images and --src-labels explicitly."
        )

    yaml_path = candidates[0]
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    yaml_root = Path(cfg.get("path", yaml_path.parent))
    if not yaml_root.is_absolute():
        yaml_root = yaml_path.parent / yaml_root

    val_rel = cfg.get("val", cfg.get("valid", "images/val"))
    val_images = (yaml_root / val_rel).resolve()
    val_labels = Path(str(val_images).replace("/images/", "/labels/")).resolve()

    if not val_images.exists():
        raise FileNotFoundError(f"Val images not found at: {val_images}")

    return val_images, val_labels


def images_to_labels(images_dir: Path) -> Path:
    """Derive labels dir from images dir using the standard YOLO convention."""
    return Path(str(images_dir).replace("/images/", "/labels/"))


# ── CLAHE ────────────────────────────────────────────────────────────────────

def apply_clahe(img: np.ndarray, clip_limit: float, tile: int) -> np.ndarray:
    """
    Apply CLAHE to a BGR or grayscale image.

    For grayscale inputs CLAHE is applied directly to the luminance.
    For color inputs we work in the LAB color space so that only the
    lightness channel L is modified, which avoids color hue shifts.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))

    if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
        gray = img if img.ndim == 2 else img[:, :, 0]
        return clahe.apply(gray)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab_clahe = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate CLAHE validation dataset (Exp 1B)")
    p.add_argument("--src-images",  type=str, default=DEFAULT_SRC_IMAGES,
                   help="Val images directory. Auto-detected if omitted.")
    p.add_argument("--src-labels",  type=str, default=DEFAULT_SRC_LABELS,
                   help="Val labels directory. Inferred from images path if omitted.")
    p.add_argument("--out-dir",     type=str, default=DEFAULT_OUT_DIR,
                   help="Root output directory for the CLAHE dataset.")
    p.add_argument("--clip-limit",  type=float, default=2.0,
                   help="CLAHE clip limit (higher = stronger enhancement, more noise risk).")
    p.add_argument("--tile-size",   type=int, default=8,
                   help="CLAHE tile grid size N (uses NxN tiles).")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def make_clahe_dataset(args):
    # Resolve source paths
    if args.src_images:
        src_images = Path(args.src_images)
        src_labels = Path(args.src_labels) if args.src_labels else images_to_labels(src_images)
    else:
        src_images, src_labels = find_val_paths()
        print(f"Auto-detected val images : {src_images}")
        print(f"Inferred val labels      : {src_labels}")

    if not src_images.exists():
        raise FileNotFoundError(f"Source images not found: {src_images}")

    out_dir    = Path(args.out_dir)
    out_images = out_dir / "images" / "val"
    out_labels = out_dir / "labels" / "val"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    img_files = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not img_files:
        raise RuntimeError(f"No images found in {src_images}")

    print(f"\nImages found   : {len(img_files)}")
    print(f"CLAHE settings : clipLimit={args.clip_limit}, tileGrid=({args.tile_size}x{args.tile_size})")
    print(f"Output root    : {out_dir}\n")

    n_ok = n_fail = 0

    for img_path in tqdm(img_files, desc="Applying CLAHE"):
        dst = out_images / img_path.name
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("cv2.imread returned None")
            cv2.imwrite(str(dst), apply_clahe(img, args.clip_limit, args.tile_size))
            n_ok += 1
        except Exception as e:
            print(f"  WARNING: {img_path.name} — {e}. Copying original.")
            shutil.copy2(img_path, dst)
            n_fail += 1

        # Labels are not modified; copy them as-is.
        lbl_src = src_labels / (img_path.stem + ".txt")
        lbl_dst = out_labels / (img_path.stem + ".txt")
        if lbl_src.exists():
            shutil.copy2(lbl_src, lbl_dst)
        else:
            lbl_dst.touch()   # empty file for images without annotations

    print(f"\nDone: {n_ok} processed, {n_fail} fallback copies.\n")

    # ── Read class info from the original YAML to populate the new one ────────
    try:
        kaggle_input = Path("/kaggle/input")
        yaml_candidates = sorted(kaggle_input.rglob("yolo_seg_train.yaml")) if kaggle_input.exists() else []
        orig_cfg = {}
        if yaml_candidates:
            with open(yaml_candidates[0]) as f:
                orig_cfg = yaml.safe_load(f)
    except Exception:
        orig_cfg = {}

    nc    = orig_cfg.get("nc", 9)
    names = orig_cfg.get("names", list(range(nc)))

    # ── Write data_clahe_val.yaml ─────────────────────────────────────────────
    yaml_out = out_dir / "data_clahe_val.yaml"
    clahe_cfg = {
        "path":  str(out_dir.resolve()),
        # val points to the CLAHE-processed images
        "val":   "images/val",
        # train still points to the ORIGINAL images (we haven't processed them)
        "train": str(src_images.parent.parent / "train"),
        "nc":    nc,
        "names": names,
    }
    with open(yaml_out, "w") as f:
        yaml.dump(clahe_cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"CLAHE dataset YAML written to:\n  {yaml_out}\n")
    print("Next step — run native validation on the CLAHE val set:")
    print(f"  python tools/val_native_yolo_seg.py --data {yaml_out} --name clahe_val")

    return str(yaml_out)


if __name__ == "__main__":
    args = parse_args()
    make_clahe_dataset(args)
