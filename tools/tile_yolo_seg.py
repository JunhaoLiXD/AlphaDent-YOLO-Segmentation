"""
tools/tile_yolo_seg.py
----------------------
Canonical crop / tile-based training utilities for AlphaDent YOLO segmentation
(Experiment V13).

⚠️ OUTCOME: V13 FAILED. Naive tiling regressed the comparable full-image Mask mAP50-95 to
   0.0993 vs V6's re-scored 0.2099 (−0.11, the worst result in the project). Tiling clips large
   objects out of training (MIN_AREA_FRAC), fragments them at inference, and merge_detections never
   stitches the non-overlapping fragments back together, so the large classes that carry most of
   the per-class-averaged mAP collapse (Abrasion −0.41, Crown −0.43). V6 (≈0.234) remains the best
   model; use it for submissions. This library is kept for reference / a possible hybrid
   (full-image model + tiling as an auxiliary small-object branch only). The "Why tiling?" rationale
   below is the original (pre-result) hypothesis — read it knowing the experiment did not pan out.

Why tiling? (original hypothesis — see the OUTCOME note above)
  The V6 error analysis found ~78% of objects occupy <1% of the image area, and
  every full-image lever (image size, model size, oversampling, augmentation, and
  the V12 P2 head) plateaued at ~0.234 Mask mAP50-95. V12 proved the bottleneck is
  the *full-image input*, not the detection head: a tiny lesion is only a few pixels
  after the panoramic image is downscaled to imgsz=768.

  Tiling attacks that directly. We slice each panoramic image into overlapping
  tiles and train on the tiles, so a lesion that was ~5 px in the full-image input
  becomes ~20-40 px in a tile. The model finally gets enough pixels to learn a
  fine-grained mask. At inference the test image is sliced the same way, each tile
  is predicted, and the predictions are mapped back to full-image coordinates and
  merged (this is what notebook 02 does).

This module is the single source of truth for the tiling geometry and the
polygon <-> tile coordinate round-trip. The SAME core functions are mirrored
inline into src/01 (build tiled dataset + train) and src/02 (tiled inference +
submission) so those notebooks stay Kaggle-self-contained (they cannot import this
repo on Kaggle). Keep the inline copies in sync with this file.

Forward (build tiled dataset), used by notebook 01:
    python tools/tile_yolo_seg.py
    python tools/tile_yolo_seg.py --tile-size 640 --overlap 0.2 --keep-empty 0.15
    python tools/tile_yolo_seg.py \
        --src-yaml /kaggle/input/.../yolo_seg_train.yaml \
        --out-dir  /kaggle/working/datasets/alphadent_tiles

Reverse (untile_polygon / merge_detections) is imported by the inference notebook
and by any later error-analysis notebook that needs full-image merged predictions.
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Tiling defaults (see the experiment log / CLAUDE.md for the rationale).
DEFAULT_TILE_SIZE = 640      # tile side in pixels
DEFAULT_OVERLAP = 0.20       # fractional overlap between neighbouring tiles
DEFAULT_KEEP_EMPTY = 0.15    # fraction of object-free TRAIN tiles to keep
DEFAULT_MIN_AREA_FRAC = 0.35 # keep a clipped object only if >= this share of its
                             # original area survives the clip to the tile
DEFAULT_SEED = 42


# ───────────────────────────── geometry ──────────────────────────────────────

def _starts(length: int, tile: int, step: int):
    """1-D tile start offsets covering [0, length) with the last tile flush to the edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def compute_tiles(img_w: int, img_h: int, tile_size: int, overlap: float):
    """
    Return a list of pixel tile boxes (x0, y0, x1, y1) covering the whole image.

    Tiles are square (side = tile_size) and overlap by `overlap`. Images smaller
    than tile_size in a dimension produce a single full-extent tile in that
    dimension. Edge tiles are shifted inward so the last one is flush with the
    image border (so the overlap near edges may be slightly larger than `overlap`).
    """
    step = max(1, int(round(tile_size * (1.0 - overlap))))
    xs = _starts(img_w, tile_size, step)
    ys = _starts(img_h, tile_size, step)
    boxes = []
    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + tile_size, img_w)
            y1 = min(y0 + tile_size, img_h)
            boxes.append((x0, y0, x1, y1))
    return boxes


def _polygon_area(pts) -> float:
    """Shoelace area of a polygon given as a list/array of (x, y) points."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_polygon_to_rect(poly, x0, y0, x1, y1):
    """
    Sutherland-Hodgman clip of a polygon to the axis-aligned rectangle
    [x0, y0, x1, y1]. `poly` is a list of (x, y) pixel points. Returns the clipped
    polygon as a list of (x, y) points (possibly empty). The rectangle is convex,
    so a single output ring is correct.
    """
    def clip_edge(points, inside, intersect):
        out = []
        n = len(points)
        if n == 0:
            return out
        for i in range(n):
            cur = points[i]
            prev = points[i - 1]
            cur_in = inside(cur)
            prev_in = inside(prev)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur))
        return out

    def isect_vert(a, b, cx):
        # intersection of segment a-b with vertical line x = cx
        dx = b[0] - a[0]
        t = 0.0 if dx == 0 else (cx - a[0]) / dx
        return (cx, a[1] + t * (b[1] - a[1]))

    def isect_horz(a, b, cy):
        dy = b[1] - a[1]
        t = 0.0 if dy == 0 else (cy - a[1]) / dy
        return (a[0] + t * (b[0] - a[0]), cy)

    pts = [tuple(p) for p in poly]
    pts = clip_edge(pts, lambda p: p[0] >= x0, lambda a, b: isect_vert(a, b, x0))  # left
    pts = clip_edge(pts, lambda p: p[0] <= x1, lambda a, b: isect_vert(a, b, x1))  # right
    pts = clip_edge(pts, lambda p: p[1] >= y0, lambda a, b: isect_horz(a, b, y0))  # top
    pts = clip_edge(pts, lambda p: p[1] <= y1, lambda a, b: isect_horz(a, b, y1))  # bottom
    return pts


# ──────────────────────── label <-> tile round-trip ──────────────────────────

def parse_seg_label_line(line):
    """Parse one YOLO-seg label line -> (class_id:int, poly_norm:np.ndarray[N,2]) or None."""
    parts = line.strip().split()
    if len(parts) < 7:  # class + >=3 (x,y) pairs
        return None
    try:
        cls = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if len(coords) % 2 != 0:
        coords = coords[:-1]
    poly = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    if len(poly) < 3:
        return None
    return cls, poly


def tile_label_for_box(full_polys, box, img_w, img_h, min_area_frac):
    """
    Clip full-image-normalized polygons to one tile box and return YOLO-seg label
    lines normalized to the tile. `full_polys` is a list of (class_id, poly_norm).
    A clipped object is kept only if >= min_area_frac of its original area survives.
    """
    x0, y0, x1, y1 = box
    tw = x1 - x0
    th = y1 - y0
    lines = []
    for cls, poly_norm in full_polys:
        poly_px = poly_norm.copy()
        poly_px[:, 0] *= img_w
        poly_px[:, 1] *= img_h
        orig_area = _polygon_area(poly_px)
        if orig_area <= 0:
            continue
        clipped = clip_polygon_to_rect(poly_px, x0, y0, x1, y1)
        if len(clipped) < 3:
            continue
        clipped = np.asarray(clipped, dtype=np.float64)
        if _polygon_area(clipped) / orig_area < min_area_frac:
            continue
        # Normalize to the tile's own coordinate frame.
        clipped[:, 0] = (clipped[:, 0] - x0) / tw
        clipped[:, 1] = (clipped[:, 1] - y0) / th
        clipped = np.clip(clipped, 0.0, 1.0)
        coord_str = " ".join(f"{v:.6f}" for v in clipped.reshape(-1))
        lines.append(f"{cls} {coord_str}")
    return lines


def untile_polygon(poly_norm_tile, box, img_w, img_h):
    """
    Map a tile-normalized polygon (from a prediction on one tile) back to a
    full-image-normalized polygon. Inverse of tile_label_for_box's normalization.
    """
    x0, y0, x1, y1 = box
    tw = x1 - x0
    th = y1 - y0
    poly = np.asarray(poly_norm_tile, dtype=np.float64).reshape(-1, 2).copy()
    poly[:, 0] = (poly[:, 0] * tw + x0) / img_w
    poly[:, 1] = (poly[:, 1] * th + y0) / img_h
    return np.clip(poly, 0.0, 1.0)


# ───────────────────────── inference-side merge ──────────────────────────────

def _poly_bbox(poly_norm):
    p = np.asarray(poly_norm, dtype=np.float64).reshape(-1, 2)
    return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


def _bbox_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_detections(detections, iou_thres=0.5):
    """
    Class-wise greedy NMS over detections collected from all tiles of one image,
    de-duplicating the same object seen in overlapping tiles.

    `detections` is a list of dicts with keys:
        class_id:int, confidence:float, poly:np.ndarray[N,2] (full-image normalized)
    Returns the kept detections, highest-confidence first.
    """
    kept = []
    for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        box = _poly_bbox(det["poly"])
        duplicate = False
        for k in kept:
            if k["class_id"] != det["class_id"]:
                continue
            if _bbox_iou(box, _poly_bbox(k["poly"])) > iou_thres:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


# ─────────────────────── forward: build tiled dataset ────────────────────────

def _images_to_labels(images_dir: Path) -> Path:
    return Path(str(images_dir).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\"))


def _resolve_split(yaml_root: Path, yaml_dir: Path, value):
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    for base in (yaml_root, yaml_dir):
        cand = base / p
        if cand.exists():
            return cand
    return yaml_root / p


def tile_split(src_images: Path, out_images: Path, out_labels: Path,
               tile_size, overlap, min_area_frac, keep_empty, rng):
    """Tile every image in one split and write tile images + labels. Returns (n_tiles, n_kept_obj_tiles)."""
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    src_labels = _images_to_labels(src_images)

    img_files = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not img_files:
        raise RuntimeError(f"No images found in {src_images}")

    n_tiles = 0
    n_obj_tiles = 0
    for img_path in tqdm(img_files, desc=f"Tiling {src_images.name}"):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  WARNING: cannot read {img_path.name}, skipping.")
            continue
        h, w = img.shape[:2]

        lbl_path = src_labels / (img_path.stem + ".txt")
        full_polys = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parsed = parse_seg_label_line(line)
                if parsed is not None:
                    full_polys.append(parsed)

        for ti, box in enumerate(compute_tiles(w, h, tile_size, overlap)):
            x0, y0, x1, y1 = box
            lines = tile_label_for_box(full_polys, box, w, h, min_area_frac)
            has_obj = len(lines) > 0

            # Subsample object-free tiles (only when keep_empty < 1).
            if not has_obj and rng.random() > keep_empty:
                continue

            tile_name = f"{img_path.stem}__t{ti}_{x0}_{y0}"
            crop = img[y0:y1, x0:x1]
            cv2.imwrite(str(out_images / f"{tile_name}.jpg"), crop)
            (out_labels / f"{tile_name}.txt").write_text(
                ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
            )
            n_tiles += 1
            n_obj_tiles += int(has_obj)

    return n_tiles, n_obj_tiles


def build_tiled_dataset(src_yaml: Path, out_dir: Path,
                        tile_size=DEFAULT_TILE_SIZE, overlap=DEFAULT_OVERLAP,
                        min_area_frac=DEFAULT_MIN_AREA_FRAC, keep_empty=DEFAULT_KEEP_EMPTY,
                        seed=DEFAULT_SEED):
    """
    Build a tiled YOLO-seg dataset from a source yolo_seg_train.yaml and write a
    new data yaml (yolo_seg_tiles.yaml) pointing at it. Train tiles subsample
    object-free tiles (keep_empty); val keeps ALL tiles so early-stopping monitors
    a stable, representative split. Returns the path to the new yaml.
    """
    src_yaml = Path(src_yaml)
    with open(src_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    yaml_dir = src_yaml.parent
    yaml_root = Path(cfg.get("path", yaml_dir))
    if not yaml_root.is_absolute():
        yaml_root = yaml_dir / yaml_root

    train_dir = _resolve_split(yaml_root, yaml_dir, cfg.get("train"))
    val_dir = _resolve_split(yaml_root, yaml_dir, cfg.get("val", cfg.get("valid")))
    if train_dir is None or not train_dir.exists():
        raise FileNotFoundError(f"Train images dir not found: {train_dir}")
    if val_dir is None or not val_dir.exists():
        raise FileNotFoundError(f"Val images dir not found: {val_dir}")

    out_dir = Path(out_dir)
    rng = random.Random(seed)

    print(f"Tiling config: tile={tile_size} overlap={overlap} "
          f"min_area_frac={min_area_frac} keep_empty(train)={keep_empty} seed={seed}")
    print(f"Source train : {train_dir}")
    print(f"Source val   : {val_dir}")
    print(f"Output root  : {out_dir}\n")

    nt_tr, no_tr = tile_split(train_dir, out_dir / "images" / "train", out_dir / "labels" / "train",
                              tile_size, overlap, min_area_frac, keep_empty, rng)
    # Val: keep every tile (keep_empty=1.0) for a stable validation distribution.
    nt_va, no_va = tile_split(val_dir, out_dir / "images" / "val", out_dir / "labels" / "val",
                              tile_size, overlap, min_area_frac, 1.0, rng)

    print(f"\nTrain tiles : {nt_tr} ({no_tr} with objects, {nt_tr - no_tr} background)")
    print(f"Val   tiles : {nt_va} ({no_va} with objects, {nt_va - no_va} background)")

    nc = cfg.get("nc", 9)
    names = cfg.get("names", list(range(nc)))
    out_yaml = out_dir / "yolo_seg_tiles.yaml"
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "path": str(out_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": nc,
            "names": names,
        }, f, sort_keys=False, allow_unicode=True)

    print(f"\nTiled dataset YAML written to:\n  {out_yaml}")
    print("\nNOTE: mAP reported while training on this tiled val split is computed on")
    print("      TILES (an easier task) and is NOT directly comparable to the ~0.234")
    print("      full-image baseline. The comparable number requires tiled+merged")
    print("      inference on the FULL val images (done in a separate analysis notebook).")
    return out_yaml


# ──────────────────────────────── CLI ────────────────────────────────────────

def _find_src_yaml():
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        raise FileNotFoundError("Cannot auto-detect outside Kaggle. Pass --src-yaml explicitly.")
    cands = sorted(kaggle_input.rglob("yolo_seg_train.yaml"))
    if not cands:
        raise FileNotFoundError("yolo_seg_train.yaml not found under /kaggle/input. Pass --src-yaml.")
    return cands[0]


def parse_args():
    p = argparse.ArgumentParser(description="Build a tiled YOLO-seg dataset (Exp V13)")
    p.add_argument("--src-yaml", type=str, default=None, help="Source yolo_seg_train.yaml. Auto-detected if omitted.")
    p.add_argument("--out-dir", type=str, default="/kaggle/working/datasets/alphadent_tiles")
    p.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    p.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    p.add_argument("--min-area-frac", type=float, default=DEFAULT_MIN_AREA_FRAC)
    p.add_argument("--keep-empty", type=float, default=DEFAULT_KEEP_EMPTY)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src_yaml = Path(args.src_yaml) if args.src_yaml else _find_src_yaml()
    build_tiled_dataset(
        src_yaml=src_yaml,
        out_dir=Path(args.out_dir),
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_area_frac=args.min_area_frac,
        keep_empty=args.keep_empty,
        seed=args.seed,
    )
