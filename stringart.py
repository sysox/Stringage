# string_art_named_best.py
# Improved strategy vs simple "residual sum":
#   Greedy chooses the next line that maximizes the DECREASE in squared error
#   (MSE gain) between target darkness T and current darkness C.
#
# Outputs (all start with PROJECT_NAME + "_"):
#   output/<PROJECT_NAME>_target_gs.jpg
#   output/<PROJECT_NAME>_target_bw.jpg
#   output/<PROJECT_NAME>_preview_gs.jpg
#   output/<PROJECT_NAME>_preview_gs.svg
#   output/<PROJECT_NAME>_preview_bw.jpg
#   output/<PROJECT_NAME>_preview_bw.svg
#   output/<PROJECT_NAME>_order_gs.txt
#   output/<PROJECT_NAME>_order_bw.txt
#
# Requirements: pip install pillow numpy svgwrite

import os, math, random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import svgwrite

# =========================
# CONFIG
# =========================
CANVAS_W_MM, CANVAS_H_MM = 460, 515
MARGIN_MM = 20

N_NAILS = 240
MIN_SKIP = 6

# Number of lines (strings)
STEPS_GS = 3000
STEPS_BW = 3000

CANDIDATES_PER_STEP = 100

# How much each selected line darkens pixels it passes through
ALPHA_DARKEN = 0.06

RASTER_W = 900
RASTER_H = int(RASTER_W * CANVAS_H_MM / CANVAS_W_MM)

INPUT_IMAGE = "input/maria.jpg"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# Choose ONE base name here (no extension)
PROJECT_NAME = INPUT_IMAGE.split('/')[1].split('.')[0]

# SVG preview look
SVG_STROKE_MM = 0.30
SVG_OPACITY = 0.55

# BW threshold for making target_bw (lower => more black)
TARGET_BW_THRESHOLD = 140

AVOID_BACKTRACK = True

# Optional: avoid always preferring long lines (set True to normalize by line length)
NORMALIZE_GAIN_BY_LENGTH = False


def out(name: str) -> str:
    """output/<PROJECT_NAME>_<name>"""
    return os.path.join(OUT_DIR, f"{PROJECT_NAME}_{name}")


# =========================
# GEOMETRY
# =========================
def mm_to_px(x_mm, y_mm):
    return (
        x_mm / CANVAS_W_MM * (RASTER_W - 1),
        y_mm / CANVAS_H_MM * (RASTER_H - 1),
    )

def generate_circle_nails():
    cx = CANVAS_W_MM / 2
    cy = CANVAS_H_MM / 2
    r  = min(CANVAS_W_MM, CANVAS_H_MM) / 2 - MARGIN_MM

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

def circular_distance(i, j, n):
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
    Loads and prepares image for both GS and BW targets:
    - crop to aspect
    - resize
    - light blur + contrast
    Returns PIL image in "L" with size (RASTER_W, RASTER_H).
    """
    img = Image.open(img_path).convert("L")
    img = center_crop_to_aspect(img, RASTER_W / RASTER_H)
    img = img.resize((RASTER_W, RASTER_H), Image.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    img = ImageEnhance.Contrast(img).enhance(1.35)
    return img

def apply_circle_mask(dark: np.ndarray, center_mm, radius_mm) -> np.ndarray:
    """
    dark in [0,1] (1=needs dark). Outside circle -> 0.
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

    # --- GS target (continuous)
    gs_dark = 1.0 - (arr / 255.0)
    gs_dark = apply_circle_mask(gs_dark, center_mm, radius_mm)
    gs_dark = np.clip(gs_dark * 1.15, 0.0, 1.0)

    target_gs_view = (255.0 * (1.0 - gs_dark)).astype(np.uint8)
    Image.fromarray(target_gs_view).save(out("target_gs.jpg"), quality=95)

    # --- BW target (binary)
    bw = np.where(arr >= TARGET_BW_THRESHOLD, 255.0, 0.0).astype(np.float32)
    bw_dark = 1.0 - (bw / 255.0)  # black->1, white->0
    bw_dark = apply_circle_mask(bw_dark, center_mm, radius_mm)
    bw_dark = np.clip(bw_dark, 0.0, 1.0)

    bw_view = (255.0 * (1.0 - bw_dark)).astype(np.uint8)
    Image.fromarray(bw_view).save(out("target_bw.jpg"), quality=95)

    return gs_dark.astype(np.float32), bw_dark.astype(np.float32)


# =========================
# LINE PRECOMPUTE
# =========================
def line_indices(x0, y0, x1, y1):
    dist = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(dist))  # ~1 sample per pixel
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    xi = np.clip(np.rint(xs).astype(np.int32), 0, RASTER_W - 1)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, RASTER_H - 1)

    # thickness: small cross
    offsets = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
    idx_list = []
    for ox, oy in offsets:
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


# =========================
# GREEDY ORDER: best strategy (MSE gain)
# =========================
def greedy_order_mse_gain(target_dark, lines, label: str, steps: int):
    """
    Choose next line that maximizes decrease in squared error:
        gain = (T-C)^2 - (T-C_new)^2  summed over pixels on the candidate line.

    Works for GS (T in [0,1]) and BW (T in {0,1}).
    """
    T = target_dark.reshape(-1).astype(np.float32)
    C = np.zeros_like(T, dtype=np.float32)

    seq = [1]
    last = 1
    prev = None
    all_nails = list(range(1, N_NAILS + 1))

    alpha = float(ALPHA_DARKEN)

    for step in range(1, steps + 1):
        cand = random.sample(all_nails, min(CANDIDATES_PER_STEP, N_NAILS))

        best_j = None
        best_gain = -1e30

        for j in cand:
            if j == last:
                continue
            if AVOID_BACKTRACK and prev is not None and j == prev:
                continue
            if circular_distance(last, j, N_NAILS) < MIN_SKIP:
                continue

            idx = lines[last][j]
            if idx is None:
                continue

            c = C[idx]
            t = T[idx]
            c2 = np.minimum(1.0, c + alpha)

            gain = float(((t - c) ** 2 - (t - c2) ** 2).sum())
            if NORMALIZE_GAIN_BY_LENGTH:
                gain /= (len(idx) + 1e-9)

            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j is None:
            best_j = random.choice([j for j in all_nails if j != last])

        idx = lines[last][best_j]
        C[idx] = np.minimum(1.0, C[idx] + alpha)

        prev, last = last, best_j
        seq.append(best_j)

        if step % 250 == 0:
            # Optional progress diagnostic (MSE on full image is expensive; we just print gain)
            print(f"[{label}] step {step}/{steps} best_gain={best_gain:.2f}")

    preview = (255.0 * (1.0 - C.reshape(RASTER_H, RASTER_W))).astype(np.uint8)
    return seq, preview


# =========================
# EXPORTS
# =========================
def export_order_txt(seq, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(map(str, seq)) + "\n")

def export_rope_svg(nails_mm, seq, filename):
    pos = {i: (x, y) for i, x, y in nails_mm}
    dwg = svgwrite.Drawing(
        filename,
        size=(f"{CANVAS_W_MM}mm", f"{CANVAS_H_MM}mm"),
        viewBox=f"0 0 {CANVAS_W_MM} {CANVAS_H_MM}",
    )
    dwg.add(dwg.rect((0, 0), (CANVAS_W_MM, CANVAS_H_MM), fill="white"))
    for a, b in zip(seq[:-1], seq[1:]):
        dwg.add(dwg.line(
            pos[a], pos[b],
            stroke="black",
            stroke_width=SVG_STROKE_MM,
            stroke_opacity=SVG_OPACITY
        ))
    dwg.save()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    nails_mm, nails_px, center_mm, radius_mm = generate_circle_nails()

    print("Preparing targets (GS + BW) ...")
    target_gs, target_bw = make_targets(INPUT_IMAGE, center_mm, radius_mm)

    print("Precomputing all line rasters (once) ...")
    lines = precompute_all_lines(nails_px)

    # ----- GS run -----
    print(f"Optimizing order for GS target ({STEPS_GS} lines) ...")
    seq_gs, preview_gs = greedy_order_mse_gain(target_gs, lines, "GS", STEPS_GS)
    export_order_txt(seq_gs, out("order_gs.txt"))
    export_rope_svg(nails_mm, seq_gs, out("preview_gs.svg"))
    Image.fromarray(preview_gs).save(out("preview_gs.jpg"), quality=95)

    # ----- BW run -----
    print(f"Optimizing order for BW target ({STEPS_BW} lines) ...")
    seq_bw, preview_bw = greedy_order_mse_gain(target_bw, lines, "BW", STEPS_BW)
    export_order_txt(seq_bw, out("order_bw.txt"))
    export_rope_svg(nails_mm, seq_bw, out("preview_bw.svg"))
    Image.fromarray(preview_bw).save(out("preview_bw.jpg"), quality=95)

    print("DONE ✔")
    print("Created:")
    print(f"  {out('target_gs.jpg')}, {out('target_bw.jpg')}")
    print(f"  {out('preview_gs.jpg')} + {out('preview_gs.svg')}")
    print(f"  {out('preview_bw.jpg')} + {out('preview_bw.svg')}")
    print(f"  {out('order_gs.txt')}, {out('order_bw.txt')}")
