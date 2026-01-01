# string_art_dither_disconnected.py
#
# Phase 1 (this file): DITHERING + DISCONNECTED-LINE STRING ART
# - Build a grayscale "darkness" target from an input image (circle-masked).
# - Create a 1-bit Floyd–Steinberg dithered target (black/white).
# - Fit the dithered target using DISCONNECTED line segments between nails:
#     * no ordering / no path constraint
#     * repeats allowed (counts)
#     * stops when improvement converges (or hits MAX_ITERS)
#
# Outputs (all start with PROJECT_NAME + "_"):
#   output/<PROJECT_NAME>_target_gs.jpg        (preprocessed grayscale target)
#   output/<PROJECT_NAME>_target_bw.jpg        (simple threshold BW target)
#   output/<PROJECT_NAME>_dither_fs.jpg        (FS dither BW target)
#   output/<PROJECT_NAME>_preview_lines.jpg    (render of fitted darkness C)
#   output/<PROJECT_NAME>_preview_lines.svg    (SVG of chosen segments; compact if huge)
#   output/<PROJECT_NAME>_lines.txt            (segment list: i j count)
#
# Requirements:
#   pip install pillow numpy svgwrite
#
# Notes:
# - "darkness" convention: 0=white, 1=black.
# - Each chosen segment adds ALPHA darkness along its rasterized pixels (clipped to 1).
#
import os
import math
import random
from collections import Counter

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import svgwrite

# =========================
# CONFIG
# =========================
CANVAS_W_MM, CANVAS_H_MM = 460, 515
MARGIN_MM = 20

# Nails
N_NAILS = 240

# Raster size (internal)
RASTER_W = 900
RASTER_H = int(RASTER_W * CANVAS_H_MM / CANVAS_W_MM)

# Input / output
INPUT_IMAGE = "input/maria.jpg"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)
PROJECT_NAME = os.path.splitext(os.path.basename(INPUT_IMAGE))[0]

# Preprocess tuning
BLUR_RADIUS = 0.8
CONTRAST = 1.35

# Convert grayscale -> darkness target
DARK_GAIN = 1.15          # multiply darkness (>=1 makes image darker)
DARK_GAMMA = 1.00         # gamma on darkness, 1.0 means unchanged

# Simple BW threshold target (optional export)
TARGET_BW_THRESHOLD = 140

# Dither settings
FS_SERPENTINE = True
RANDOM_SEED = 1

# Rasterized line thickness (simple cross kernel)
LINE_OFFSETS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

# =========================
# DISCONNECTED LINE FIT (QUALITY-FIRST)
# =========================
# Each chosen segment darkens its pixels by this amount (smaller => more segments but finer tone)
DIS_ALPHA = 0.03

# Candidate evaluation per iteration (higher => better quality, slower).
# With N=240, unique lines ~ 28680. For best quality, push toward 20000-28680.
DIS_CANDIDATES_PER_ITER = 12000

# Stop when best per-iteration gain falls below this (lower => more segments, better fit).
DIS_STOP_GAIN = 1e-3

# Hard cap (very large; convergence usually stops earlier)
DIS_MAX_ITERS = 200000

# Optional: exclude too-short chords by enforcing min circular distance between endpoints.
# Set to 0 to allow all. If you want to avoid very short lines, try 4..8.
DIS_MIN_SKIP = 0

# Report progress every N iterations
DIS_REPORT_EVERY = 1000

# SVG preview look
SVG_STROKE_MM = 0.30
SVG_OPACITY = 0.55

# If total passes exceeds this, export SVG in "compact" mode (one line per pair, opacity by count)
SVG_COMPACT_IF_PASSES_OVER = 20000


def out(name: str) -> str:
    """output/<PROJECT_NAME>_<name>"""
    return os.path.join(OUT_DIR, f"{PROJECT_NAME}_{name}")


# =========================
# GEOMETRY
# =========================
def mm_to_px(x_mm: float, y_mm: float) -> tuple[float, float]:
    return (
        x_mm / CANVAS_W_MM * (RASTER_W - 1),
        y_mm / CANVAS_H_MM * (RASTER_H - 1),
    )


def generate_circle_nails():
    cx = CANVAS_W_MM / 2
    cy = CANVAS_H_MM / 2
    r = min(CANVAS_W_MM, CANVAS_H_MM) / 2 - MARGIN_MM

    nails_mm, nails_px = [], []
    for k in range(N_NAILS):
        i = k + 1
        a = 2 * math.pi * k / N_NAILS
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        nails_mm.append((i, x, y))
        xp, yp = mm_to_px(x, y)
        nails_px.append((i, xp, yp))

    return nails_mm, nails_px, (cx, cy), r


def circular_distance(i: int, j: int, n: int) -> int:
    d = abs(i - j)
    return min(d, n - d)


# =========================
# IMAGE PREP
# =========================
def center_crop_to_aspect(img: Image.Image, target_aspect: float) -> Image.Image:
    w, h = img.size
    src_aspect = w / h
    if src_aspect > target_aspect:
        new_w = int(h * target_aspect)
        x0 = (w - new_w) // 2
        return img.crop((x0, 0, x0 + new_w, h))
    else:
        new_h = int(w / target_aspect)
        y0 = (h - new_h) // 2
        return img.crop((0, y0, w, y0 + new_h))


def preprocess_common(img_path: str) -> Image.Image:
    """
    Loads and prepares image:
    - crop to aspect
    - resize
    - light blur + contrast
    Returns PIL image in "L" with size (RASTER_W, RASTER_H).
    """
    img = Image.open(img_path).convert("L")
    img = center_crop_to_aspect(img, RASTER_W / RASTER_H)
    img = img.resize((RASTER_W, RASTER_H), Image.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    img = ImageEnhance.Contrast(img).enhance(CONTRAST)
    return img


def apply_circle_mask(dark: np.ndarray, center_mm, radius_mm) -> np.ndarray:
    """
    dark in [0,1] (1=needs dark/blackness). Outside circle -> 0.
    """
    cx_mm, cy_mm = center_mm
    cx_px, cy_px = mm_to_px(cx_mm, cy_mm)
    r_px = (radius_mm / CANVAS_W_MM) * (RASTER_W - 1)

    yy, xx = np.mgrid[0:RASTER_H, 0:RASTER_W]
    mask = ((xx - cx_px) ** 2 + (yy - cy_px) ** 2) <= (r_px ** 2)
    return dark * mask.astype(np.float32)


def make_targets(img_path: str, center_mm, radius_mm):
    """
    Returns:
      target_gs_dark: float32 in [0,1]
      target_bw_dark: float32 in {0,1}
    Also saves:
      output/<PROJECT_NAME>_target_gs.jpg
      output/<PROJECT_NAME>_target_bw.jpg
    """
    img = preprocess_common(img_path)
    arr = np.asarray(img, dtype=np.float32)

    # --- GS target (continuous darkness)
    gs_dark = 1.0 - (arr / 255.0)
    gs_dark = apply_circle_mask(gs_dark, center_mm, radius_mm)
    gs_dark = np.clip(gs_dark * float(DARK_GAIN), 0.0, 1.0)
    if DARK_GAMMA != 1.0:
        gs_dark = np.clip(gs_dark, 0.0, 1.0) ** float(DARK_GAMMA)

    target_gs_view = (255.0 * (1.0 - gs_dark)).astype(np.uint8)
    Image.fromarray(target_gs_view).save(out("target_gs.jpg"), quality=95)

    # --- Simple BW threshold target (binary)
    bw = np.where(arr >= TARGET_BW_THRESHOLD, 255.0, 0.0).astype(np.float32)
    bw_dark = 1.0 - (bw / 255.0)  # black->1, white->0
    bw_dark = apply_circle_mask(bw_dark, center_mm, radius_mm)
    bw_dark = np.clip(bw_dark, 0.0, 1.0)

    bw_view = (255.0 * (1.0 - bw_dark)).astype(np.uint8)
    Image.fromarray(bw_view).save(out("target_bw.jpg"), quality=95)

    return gs_dark.astype(np.float32), bw_dark.astype(np.float32)


# =========================
# DITHERING
# =========================
def dither_floyd_steinberg_dark(dark: np.ndarray, serpentine: bool = True) -> np.ndarray:
    """
    Floyd–Steinberg error diffusion on darkness in [0,1].
    Returns bw_dark in {0,1} float32 where 1=black.
    """
    h, w = dark.shape
    work = dark.astype(np.float32).copy()
    out_bw = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        if serpentine and (y % 2 == 1):
            xs = range(w - 1, -1, -1)
            dir_sign = -1
        else:
            xs = range(w)
            dir_sign = +1

        for x in xs:
            old = work[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            out_bw[y, x] = new
            err = old - new

            # Standard FS weights; mirrored for right-to-left scan
            if dir_sign == +1:
                if x + 1 < w:
                    work[y, x + 1] += err * (7.0 / 16.0)
                if y + 1 < h and x - 1 >= 0:
                    work[y + 1, x - 1] += err * (3.0 / 16.0)
                if y + 1 < h:
                    work[y + 1, x] += err * (5.0 / 16.0)
                if y + 1 < h and x + 1 < w:
                    work[y + 1, x + 1] += err * (1.0 / 16.0)
            else:
                if x - 1 >= 0:
                    work[y, x - 1] += err * (7.0 / 16.0)
                if y + 1 < h and x + 1 < w:
                    work[y + 1, x + 1] += err * (3.0 / 16.0)
                if y + 1 < h:
                    work[y + 1, x] += err * (5.0 / 16.0)
                if y + 1 < h and x - 1 >= 0:
                    work[y + 1, x - 1] += err * (1.0 / 16.0)

        # keep sane (optional)
        work[y, :] = np.clip(work[y, :], -1.0, 2.0)

    return out_bw


# =========================
# LINE PRECOMPUTE
# =========================
def line_indices(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    dist = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(dist))  # ~1 sample per pixel
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    xi = np.clip(np.rint(xs).astype(np.int32), 0, RASTER_W - 1)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, RASTER_H - 1)

    idx_list = []
    for ox, oy in LINE_OFFSETS:
        xj = np.clip(xi + ox, 0, RASTER_W - 1)
        yj = np.clip(yi + oy, 0, RASTER_H - 1)
        idx_list.append(yj * RASTER_W + xj)

    return np.unique(np.concatenate(idx_list))


def precompute_all_lines(nails_px):
    pos = {i: (x, y) for i, x, y in nails_px}
    lines = [[None] * (N_NAILS + 1) for _ in range(N_NAILS + 1)]
    for i in range(1, N_NAILS + 1):
        x0, y0 = pos[i]
        for j in range(1, N_NAILS + 1):
            if i != j:
                x1, y1 = pos[j]
                lines[i][j] = line_indices(x0, y0, x1, y1)
    return lines


def build_line_list(lines, min_skip: int = 0):
    """
    Flatten lines[i][j] into a list of undirected segments (i<j) with raster indices.
    Optionally require circular_distance(i,j) >= min_skip.
    """
    line_list = []
    for i in range(1, N_NAILS + 1):
        for j in range(i + 1, N_NAILS + 1):
            if min_skip and circular_distance(i, j, N_NAILS) < min_skip:
                continue
            idx = lines[i][j]
            if idx is not None and len(idx) > 0:
                line_list.append((i, j, idx))
    return line_list


# =========================
# DISCONNECTED SEGMENT FIT
# =========================
def greedy_disconnected_mse_gain(
    target_dark: np.ndarray,
    line_list,
    label: str,
    alpha: float,
    max_iters: int,
    candidates_per_iter: int,
    stop_gain: float,
    report_every: int,
    seed: int = 1,
):
    """
    Select disconnected segments to minimize squared error to target_dark.

    target_dark: float32 [0,1] (usually a BW-dithered map {0,1} but can be GS too).
    Each segment adds alpha darkness on its pixels:
        C[idx] = min(1, C[idx] + alpha)

    Stops when the best candidate gain < stop_gain.
    Returns:
      chosen_pairs: list[(i,j)] (repeats allowed)
      preview_img: uint8 image showing C (white=255, black=0)
      final_mse: float
    """
    rng = np.random.default_rng(seed)

    T = target_dark.reshape(-1).astype(np.float32)
    C = np.zeros_like(T, dtype=np.float32)

    n_lines = len(line_list)
    chosen = []

    mse = float(np.mean((T - C) ** 2))

    for it in range(1, max_iters + 1):
        k = min(candidates_per_iter, n_lines)
        cand_ids = rng.choice(n_lines, size=k, replace=False)

        best_gain = -1e30
        best_id = None

        for cid in cand_ids:
            i, j, idx = line_list[cid]
            c = C[idx]
            t = T[idx]
            c2 = np.minimum(1.0, c + alpha)
            gain = float(((t - c) ** 2 - (t - c2) ** 2).sum())
            if gain > best_gain:
                best_gain = gain
                best_id = cid

        if best_id is None or best_gain < stop_gain:
            print(f"[{label}] stop at iter={it}, best_gain={best_gain:.4g}, mse={mse:.6f}, passes={len(chosen)}")
            break

        i, j, idx = line_list[best_id]
        C[idx] = np.minimum(1.0, C[idx] + alpha)
        chosen.append((i, j))

        if (it % report_every) == 0:
            mse = float(np.mean((T - C) ** 2))
            print(f"[{label}] iter {it}  best_gain={best_gain:.2f}  mse={mse:.6f}  passes={len(chosen)}")

    mse = float(np.mean((T - C) ** 2))
    preview = (255.0 * (1.0 - C.reshape(RASTER_H, RASTER_W))).astype(np.uint8)
    return chosen, preview, mse


def compress_pairs(pairs):
    """
    Convert list[(i,j)] with repeats into Counter[(i,j)] = count (normalized i<j).
    """
    norm = [(a, b) if a < b else (b, a) for a, b in pairs]
    return Counter(norm)


# =========================
# EXPORTS
# =========================
def export_lines_txt_counts(counts: Counter, path: str):
    """
    Write lines as: i j count
    Sorted by count desc, then endpoints.
    """
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    with open(path, "w", encoding="utf-8") as f:
        for (i, j), c in items:
            f.write(f"{i} {j} {c}\n")


def export_lines_svg_counts_repeated(nails_mm, counts: Counter, filename: str):
    """
    Draw repeats literally (faithful but can be huge).
    """
    pos = {i: (x, y) for i, x, y in nails_mm}
    dwg = svgwrite.Drawing(
        filename,
        size=(f"{CANVAS_W_MM}mm", f"{CANVAS_H_MM}mm"),
        viewBox=f"0 0 {CANVAS_W_MM} {CANVAS_H_MM}",
    )
    dwg.add(dwg.rect((0, 0), (CANVAS_W_MM, CANVAS_H_MM), fill="white"))

    # draw most-used first (aesthetic)
    for (i, j), c in sorted(counts.items(), key=lambda kv: -kv[1]):
        for _ in range(int(c)):
            dwg.add(
                dwg.line(
                    pos[i],
                    pos[j],
                    stroke="black",
                    stroke_width=SVG_STROKE_MM,
                    stroke_opacity=SVG_OPACITY,
                )
            )
    dwg.save()


def export_lines_svg_counts_compact(nails_mm, counts: Counter, filename: str):
    """
    Compact SVG: draw each unique segment once, with opacity scaled by count.
    This keeps the file small and still conveys density well.
    """
    pos = {i: (x, y) for i, x, y in nails_mm}
    dwg = svgwrite.Drawing(
        filename,
        size=(f"{CANVAS_W_MM}mm", f"{CANVAS_H_MM}mm"),
        viewBox=f"0 0 {CANVAS_W_MM} {CANVAS_H_MM}",
    )
    dwg.add(dwg.rect((0, 0), (CANVAS_W_MM, CANVAS_H_MM), fill="white"))

    max_c = max(counts.values()) if counts else 1
    denom = math.log1p(max_c)

    for (i, j), c in sorted(counts.items(), key=lambda kv: -kv[1]):
        # scale opacity smoothly; keep within [0.05, SVG_OPACITY]
        s = math.log1p(c) / denom if denom > 0 else 1.0
        op = max(0.05, min(float(SVG_OPACITY), float(SVG_OPACITY) * s))
        dwg.add(
            dwg.line(
                pos[i],
                pos[j],
                stroke="black",
                stroke_width=SVG_STROKE_MM,
                stroke_opacity=op,
            )
        )
    dwg.save()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    nails_mm, nails_px, center_mm, radius_mm = generate_circle_nails()

    print("Preparing targets (GS + simple BW) ...")
    target_gs, target_bw = make_targets(INPUT_IMAGE, center_mm, radius_mm)

    print("Creating Floyd–Steinberg dither target from GS ...")
    dither_bw = dither_floyd_steinberg_dark(target_gs, serpentine=FS_SERPENTINE).astype(np.float32)
    Image.fromarray((255.0 * (1.0 - dither_bw)).astype(np.uint8)).save(out("dither_fs.jpg"), quality=95)

    print("Precomputing all line rasters (once) ...")
    lines = precompute_all_lines(nails_px)

    print("Building undirected line list ...")
    line_list = build_line_list(lines, min_skip=DIS_MIN_SKIP)
    print(f"Total unique segments: {len(line_list)} (min_skip={DIS_MIN_SKIP})")

    print("Fitting DISCONNECTED segments to dither target (quality-first) ...")
    chosen_pairs, preview_lines, mse = greedy_disconnected_mse_gain(
        target_dark=dither_bw,
        line_list=line_list,
        label="DIS",
        alpha=float(DIS_ALPHA),
        max_iters=int(DIS_MAX_ITERS),
        candidates_per_iter=int(DIS_CANDIDATES_PER_ITER),
        stop_gain=float(DIS_STOP_GAIN),
        report_every=int(DIS_REPORT_EVERY),
        seed=RANDOM_SEED,
    )

    counts = compress_pairs(chosen_pairs)
    total_passes = int(sum(counts.values()))
    unique_segments = int(len(counts))

    Image.fromarray(preview_lines).save(out("preview_lines.jpg"), quality=95)
    export_lines_txt_counts(counts, out("lines.txt"))

    # SVG export: repeated if small enough, compact otherwise
    svg_path = out("preview_lines.svg")
    if total_passes <= SVG_COMPACT_IF_PASSES_OVER:
        export_lines_svg_counts_repeated(nails_mm, counts, svg_path)
    else:
        export_lines_svg_counts_compact(nails_mm, counts, svg_path)

    print("DONE ✔")
    print("Created:")
    print(f"  {out('target_gs.jpg')}")
    print(f"  {out('target_bw.jpg')}")
    print(f"  {out('dither_fs.jpg')}")
    print(f"  {out('preview_lines.jpg')} + {out('preview_lines.svg')}")
    print(f"  {out('lines.txt')}")
    print(f"Stats: unique_segments={unique_segments}, total_passes={total_passes}, final_mse={mse:.6f}")
