# string_art_gs_perceptual_fast.py
#
# Best-next-step implementation:
# - Fit the *grayscale* darkness target (not BW) for better quality.
# - Approximate "perceptual blur" by:
#     (a) optimizing primarily on a downsampled raster (low-res = blurred view)
#     (b) using a thicker line influence footprint (soft/tube-like basis)
#     (c) optionally weighting errors to protect highlights (white areas)
# - Disconnected segments: output is a multiset of nail-pairs with integer counts.
# - Faster than the previous version by:
#     * optimizing on OPT resolution (default 450px width)
#     * optional COARSE stage at even lower resolution (default 225px width)
#     * storing only undirected segments (i<j): ~N*(N-1)/2 masks instead of NxN
#
# Outputs (all start with PROJECT_NAME + "_"):
#   output/<PROJECT_NAME>_target_gs.jpg          (full-res target, visualization)
#   output/<PROJECT_NAME>_target_gs_opt.jpg      (opt-res target, visualization)
#   output/<PROJECT_NAME>_dither_fs.jpg          (FS dither of GS target, for reference)
#   output/<PROJECT_NAME>_preview_lines.jpg      (full-res fitted preview)
#   output/<PROJECT_NAME>_preview_lines.svg      (SVG of chosen segments)
#   output/<PROJECT_NAME>_lines.txt              (segment list: i j count)
#
# Requirements:
#   pip install pillow numpy svgwrite
#
# Run:
#   python3 string_art_gs_perceptual_fast.py
#
# Notes:
# - "darkness" convention: 0=white, 1=black.
# - Each chosen segment adds ALPHA darkness on its pixels (clipped to 1).
#
import os
import math
from collections import Counter

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import svgwrite

# =========================
# CONFIG (edit these)
# =========================
CANVAS_W_MM, CANVAS_H_MM = 460, 515
MARGIN_MM = 20

INPUT_IMAGE = "input/lena.jpg"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)
PROJECT_NAME = os.path.splitext(os.path.basename(INPUT_IMAGE))[0]

# Nails
N_NAILS = 240

# Preprocess tuning
BLUR_RADIUS = 0.8
CONTRAST = 1.35
DARK_GAIN = 1.15          # multiply darkness (>=1 makes image darker)
DARK_GAMMA = 1.00         # gamma on darkness

# Dither export (reference only; optimization uses GS)
FS_SERPENTINE = True

# Optimization resolutions (low-res acts like perceptual blur)
OPT_W = 450
COARSE_W = 225            # set to 0 to disable coarse stage
USE_COARSE_STAGE = True

# Line influence footprint:
# - thickness/influence radius in pixels at each stage.
#   Larger radius approximates perceptual averaging.
OPT_INFLUENCE_RADIUS = 2
COARSE_INFLUENCE_RADIUS = 2

# Fit model: each chosen segment adds this much darkness
ALPHA = 0.03              # smaller => finer tones, more passes

# Candidate evaluation per iteration (quality vs speed)
# Unique segments ~ N*(N-1)/2; for 240 nails it's ~28680.
# Using 4000-12000 is often a good speed/quality tradeoff.
CANDIDATES_PER_ITER_OPT = 8000
CANDIDATES_PER_ITER_COARSE = 6000

# Stop when best improvement (gain) becomes tiny
STOP_GAIN_OPT = 5e-4
STOP_GAIN_COARSE = 2e-4

# Hard caps (usually stops earlier via STOP_GAIN)
MAX_ITERS_OPT = 80000
MAX_ITERS_COARSE = 25000

# Protect highlights (white areas) by weighting error more there:
# weight = 1 + WHITE_PENALTY * (1 - target_dark)
# where target_dark near 0 => white => larger weight.
WHITE_PENALTY = 1.5       # set 0 to disable

# Skip very short chords (0 disables). If you dislike tiny local segments, try 4..8.
MIN_SKIP = 0

# Progress prints
REPORT_EVERY = 1000

# SVG preview look
SVG_STROKE_MM = 0.30
SVG_OPACITY = 0.55
SVG_COMPACT_IF_PASSES_OVER = 20000


def out(name: str) -> str:
    """output/<PROJECT_NAME>_<name>"""
    return os.path.join(OUT_DIR, f"{PROJECT_NAME}_{name}")


# =========================
# GEOMETRY
# =========================
def circular_distance(i: int, j: int, n: int) -> int:
    d = abs(i - j)
    return min(d, n - d)


def generate_circle_nails_mm():
    cx = CANVAS_W_MM / 2
    cy = CANVAS_H_MM / 2
    r = min(CANVAS_W_MM, CANVAS_H_MM) / 2 - MARGIN_MM

    nails_mm = []
    for k in range(N_NAILS):
        i = k + 1
        a = 2 * math.pi * k / N_NAILS
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        nails_mm.append((i, x, y))

    return nails_mm, (cx, cy), r


def mm_to_px(x_mm: float, y_mm: float, raster_w: int, raster_h: int) -> tuple[float, float]:
    return (
        x_mm / CANVAS_W_MM * (raster_w - 1),
        y_mm / CANVAS_H_MM * (raster_h - 1),
    )


def nails_mm_to_px(nails_mm, raster_w: int, raster_h: int):
    nails_px = []
    for i, x, y in nails_mm:
        xp, yp = mm_to_px(x, y, raster_w, raster_h)
        nails_px.append((i, xp, yp))
    return nails_px


def circle_mask_dark(dark: np.ndarray, center_mm, radius_mm, raster_w: int, raster_h: int) -> np.ndarray:
    """dark in [0,1]. Outside circle -> 0."""
    cx_mm, cy_mm = center_mm
    cx_px, cy_px = mm_to_px(cx_mm, cy_mm, raster_w, raster_h)
    r_px = (radius_mm / CANVAS_W_MM) * (raster_w - 1)

    yy, xx = np.mgrid[0:raster_h, 0:raster_w]
    mask = ((xx - cx_px) ** 2 + (yy - cy_px) ** 2) <= (r_px ** 2)
    return dark * mask.astype(np.float32)


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


def preprocess_common(img_path: str, raster_w: int, raster_h: int) -> Image.Image:
    """
    Returns PIL L image sized (raster_w, raster_h).
    """
    img = Image.open(img_path).convert("L")
    img = center_crop_to_aspect(img, raster_w / raster_h)
    img = img.resize((raster_w, raster_h), Image.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    img = ImageEnhance.Contrast(img).enhance(CONTRAST)
    return img


def make_target_gs_dark(img_path: str, center_mm, radius_mm, raster_w: int, raster_h: int) -> np.ndarray:
    """
    Returns target darkness T in [0,1], float32.
    """
    img = preprocess_common(img_path, raster_w, raster_h)
    arr = np.asarray(img, dtype=np.float32)
    dark = 1.0 - (arr / 255.0)
    dark = circle_mask_dark(dark, center_mm, radius_mm, raster_w, raster_h)
    dark = np.clip(dark * float(DARK_GAIN), 0.0, 1.0)
    if DARK_GAMMA != 1.0:
        dark = np.clip(dark, 0.0, 1.0) ** float(DARK_GAMMA)
    return dark.astype(np.float32)


# =========================
# DITHERING (reference export only)
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

        work[y, :] = np.clip(work[y, :], -1.0, 2.0)

    return out_bw


# =========================
# LINE RASTERIZATION (undirected list)
# =========================
def disk_offsets(radius: int):
    """Integer offsets within a filled disk of given radius (inclusive)."""
    if radius <= 0:
        return [(0, 0)]
    out = []
    r2 = radius * radius
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            if ox * ox + oy * oy <= r2:
                out.append((ox, oy))
    return out


def line_indices(x0: float, y0: float, x1: float, y1: float, raster_w: int, raster_h: int, offsets):
    dist = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(dist))  # ~1 sample per pixel
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    xi = np.clip(np.rint(xs).astype(np.int32), 0, raster_w - 1)
    yi = np.clip(np.rint(ys).astype(np.int32), 0, raster_h - 1)

    idx_list = []
    for ox, oy in offsets:
        xj = np.clip(xi + ox, 0, raster_w - 1)
        yj = np.clip(yi + oy, 0, raster_h - 1)
        idx_list.append(yj * raster_w + xj)

    return np.unique(np.concatenate(idx_list))


def build_line_list(nails_px, raster_w: int, raster_h: int, influence_radius: int, min_skip: int = 0):
    """
    Returns list of (i, j, idx) for i<j.
    """
    offsets = disk_offsets(influence_radius)
    pos = {i: (x, y) for i, x, y in nails_px}

    line_list = []
    for i in range(1, N_NAILS + 1):
        x0, y0 = pos[i]
        for j in range(i + 1, N_NAILS + 1):
            if min_skip and circular_distance(i, j, N_NAILS) < min_skip:
                continue
            x1, y1 = pos[j]
            idx = line_indices(x0, y0, x1, y1, raster_w, raster_h, offsets)
            if idx is not None and len(idx) > 0:
                line_list.append((i, j, idx))

    return line_list


# =========================
# GREEDY DISCONNECTED FIT (weighted MSE gain)
# =========================
def greedy_disconnected_weighted_mse_gain(
    target_dark: np.ndarray,
    line_list,
    alpha: float,
    candidates_per_iter: int,
    stop_gain: float,
    max_iters: int,
    report_every: int,
    white_penalty: float,
    seed: int = 1,
    label: str = "STAGE",
):
    """
    Optimize C to match target_dark using disconnected segments (repeats allowed).
    Objective: sum_p W_p * (T_p - C_p)^2, where W_p depends on T_p (optional).
    """
    rng = np.random.default_rng(seed)

    T = target_dark.reshape(-1).astype(np.float32)
    C = np.zeros_like(T, dtype=np.float32)

    if white_penalty > 0:
        W = (1.0 + float(white_penalty) * (1.0 - T)).astype(np.float32)
    else:
        W = None

    n_lines = len(line_list)
    chosen = []

    def mse_now():
        if W is None:
            return float(np.mean((T - C) ** 2))
        return float(np.mean(W * (T - C) ** 2))

    mse = mse_now()

    for it in range(1, max_iters + 1):
        k = min(candidates_per_iter, n_lines)
        cand_ids = rng.choice(n_lines, size=k, replace=False)

        best_gain = -1e30
        best_id = None

        # Evaluate exact weighted MSE gain on pixels of each candidate
        for cid in cand_ids:
            i, j, idx = line_list[cid]
            c = C[idx]
            t = T[idx]
            c2 = np.minimum(1.0, c + alpha)

            if W is None:
                gain = float(((t - c) ** 2 - (t - c2) ** 2).sum())
            else:
                w = W[idx]
                gain = float((w * ((t - c) ** 2 - (t - c2) ** 2)).sum())

            if gain > best_gain:
                best_gain = gain
                best_id = cid

        if best_id is None or best_gain < stop_gain:
            print(f"[{label}] stop at iter={it}, best_gain={best_gain:.4g}, mse={mse:.6f}, passes={len(chosen)}")
            break

        i, j, idx = line_list[best_id]
        C[idx] = np.minimum(1.0, C[idx] + alpha)
        chosen.append((i, j))

        if it % report_every == 0:
            mse = mse_now()
            print(f"[{label}] iter {it}  best_gain={best_gain:.3f}  mse={mse:.6f}  passes={len(chosen)}")

    mse = mse_now()
    return chosen, C.reshape(target_dark.shape).astype(np.float32), mse


def compress_pairs(pairs):
    norm = [(a, b) if a < b else (b, a) for a, b in pairs]
    return Counter(norm)


# =========================
# RENDER + EXPORT
# =========================
def render_counts_to_dark(counts: Counter, nails_mm, raster_w: int, raster_h: int, alpha: float, influence_radius: int):
    """
    Render final darkness image at (raster_w, raster_h) by applying each segment count times.
    This is used for the full-res preview.
    """
    nails_px = nails_mm_to_px(nails_mm, raster_w, raster_h)
    pos = {i: (x, y) for i, x, y in nails_px}
    offsets = disk_offsets(influence_radius)

    C = np.zeros((raster_h * raster_w,), dtype=np.float32)

    # Cache indices for segments that appear
    idx_cache = {}
    for (i, j), c in counts.items():
        if i > j:
            i, j = j, i
        key = (i, j)
        if key not in idx_cache:
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            idx_cache[key] = line_indices(x0, y0, x1, y1, raster_w, raster_h, offsets)

        idx = idx_cache[key]
        # Apply c times; combine in one add
        C[idx] = np.minimum(1.0, C[idx] + float(alpha) * float(c))

    return C.reshape((raster_h, raster_w)).astype(np.float32)


def export_lines_txt_counts(counts: Counter, path: str):
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    with open(path, "w", encoding="utf-8") as f:
        for (i, j), c in items:
            f.write(f"{i} {j} {c}\n")


def export_lines_svg(nails_mm, counts: Counter, filename: str, compact: bool):
    pos = {i: (x, y) for i, x, y in nails_mm}
    dwg = svgwrite.Drawing(
        filename,
        size=(f"{CANVAS_W_MM}mm", f"{CANVAS_H_MM}mm"),
        viewBox=f"0 0 {CANVAS_W_MM} {CANVAS_H_MM}",
    )
    dwg.add(dwg.rect((0, 0), (CANVAS_W_MM, CANVAS_H_MM), fill="white"))

    if not counts:
        dwg.save()
        return

    if compact:
        max_c = max(counts.values())
        denom = math.log1p(max_c)
        for (i, j), c in sorted(counts.items(), key=lambda kv: -kv[1]):
            s = (math.log1p(c) / denom) if denom > 0 else 1.0
            op = max(0.05, min(float(SVG_OPACITY), float(SVG_OPACITY) * s))
            dwg.add(dwg.line(
                pos[i], pos[j],
                stroke="black",
                stroke_width=SVG_STROKE_MM,
                stroke_opacity=op
            ))
    else:
        for (i, j), c in sorted(counts.items(), key=lambda kv: -kv[1]):
            for _ in range(int(c)):
                dwg.add(dwg.line(
                    pos[i], pos[j],
                    stroke="black",
                    stroke_width=SVG_STROKE_MM,
                    stroke_opacity=SVG_OPACITY
                ))

    dwg.save()


def save_dark_as_image(dark: np.ndarray, path: str):
    img = (255.0 * (1.0 - np.clip(dark, 0.0, 1.0))).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    np.random.seed(1)

    nails_mm, center_mm, radius_mm = generate_circle_nails_mm()

    # Full-res target for reference + final preview rendering
    FULL_W = 900
    FULL_H = int(FULL_W * CANVAS_H_MM / CANVAS_W_MM)

    # OPT dimensions
    OPT_H = int(OPT_W * CANVAS_H_MM / CANVAS_W_MM)

    print("Preparing targets (GS) ...")
    target_full = make_target_gs_dark(INPUT_IMAGE, center_mm, radius_mm, FULL_W, FULL_H)
    target_opt = make_target_gs_dark(INPUT_IMAGE, center_mm, radius_mm, OPT_W, OPT_H)

    save_dark_as_image(target_full, out("target_gs.jpg"))
    save_dark_as_image(target_opt, out("target_gs_opt.jpg"))

    print("Creating FS dither reference from full GS ...")
    dither_bw = dither_floyd_steinberg_dark(target_full, serpentine=FS_SERPENTINE)
    save_dark_as_image(dither_bw, out("dither_fs.jpg"))

    # Optional COARSE stage (lower-res = more blur/perceptual)
    counts = Counter()

    if USE_COARSE_STAGE and COARSE_W > 0:
        COARSE_H = int(COARSE_W * CANVAS_H_MM / CANVAS_W_MM)
        print(f"\n[COARSE] Building line list at {COARSE_W}x{COARSE_H} (radius={COARSE_INFLUENCE_RADIUS}) ...")
        nails_px_coarse = nails_mm_to_px(nails_mm, COARSE_W, COARSE_H)
        line_list_coarse = build_line_list(
            nails_px_coarse, COARSE_W, COARSE_H,
            influence_radius=COARSE_INFLUENCE_RADIUS,
            min_skip=MIN_SKIP
        )
        print(f"[COARSE] Unique segments: {len(line_list_coarse)}")

        target_coarse = make_target_gs_dark(INPUT_IMAGE, center_mm, radius_mm, COARSE_W, COARSE_H)

        print("[COARSE] Optimizing (disconnected, weighted MSE) ...")
        chosen_pairs, C_coarse, mse_coarse = greedy_disconnected_weighted_mse_gain(
            target_dark=target_coarse,
            line_list=line_list_coarse,
            alpha=float(ALPHA),
            candidates_per_iter=int(CANDIDATES_PER_ITER_COARSE),
            stop_gain=float(STOP_GAIN_COARSE),
            max_iters=int(MAX_ITERS_COARSE),
            report_every=int(REPORT_EVERY),
            white_penalty=float(WHITE_PENALTY),
            seed=1,
            label="COARSE",
        )
        counts = compress_pairs(chosen_pairs)
        print(f"[COARSE] Done: unique={len(counts)}, passes={sum(counts.values())}, mse={mse_coarse:.6f}")

    # OPT stage: continue from coarse solution (same segment space: nail-pairs)
    print(f"\n[OPT] Building line list at {OPT_W}x{OPT_H} (radius={OPT_INFLUENCE_RADIUS}) ...")
    nails_px_opt = nails_mm_to_px(nails_mm, OPT_W, OPT_H)
    line_list_opt = build_line_list(
        nails_px_opt, OPT_W, OPT_H,
        influence_radius=OPT_INFLUENCE_RADIUS,
        min_skip=MIN_SKIP
    )
    print(f"[OPT] Unique segments: {len(line_list_opt)}")

    # Build initial canvas C0 at OPT resolution from existing counts (if coarse ran)
    if counts:
        print("[OPT] Rendering coarse solution as initialization ...")
        C0_opt = render_counts_to_dark(
            counts, nails_mm, OPT_W, OPT_H,
            alpha=float(ALPHA),
            influence_radius=OPT_INFLUENCE_RADIUS
        )
    else:
        C0_opt = np.zeros((OPT_H, OPT_W), dtype=np.float32)

    # We’ll optimize by continuing from C0_opt:
    # Easiest way (fast + clean): re-run greedy but starting from current C.
    # For that, we create a modified target that "expects" remaining darkness.
    # Equivalent: set target := clip(target - C0, 0, 1) and start from zero.
    # This keeps the same gain formula and avoids modifying inner loop.
    print("[OPT] Optimizing refinement ...")
    target_rem = np.clip(target_opt - C0_opt, 0.0, 1.0).astype(np.float32)

    chosen_pairs_opt, C_opt_rem, mse_opt = greedy_disconnected_weighted_mse_gain(
        target_dark=target_rem,
        line_list=line_list_opt,
        alpha=float(ALPHA),
        candidates_per_iter=int(CANDIDATES_PER_ITER_OPT),
        stop_gain=float(STOP_GAIN_OPT),
        max_iters=int(MAX_ITERS_OPT),
        report_every=int(REPORT_EVERY),
        white_penalty=float(WHITE_PENALTY),
        seed=2,
        label="OPT",
    )

    counts_opt_add = compress_pairs(chosen_pairs_opt)
    counts.update(counts_opt_add)

    total_passes = int(sum(counts.values()))
    unique_segments = int(len(counts))
    print(f"[OPT] Done: unique={unique_segments}, passes={total_passes}, mse(rem)={mse_opt:.6f}")

    # Final full-res render + exports
    print("\n[FINAL] Rendering full-res preview ...")
    preview_full = render_counts_to_dark(
        counts, nails_mm, FULL_W, FULL_H,
        alpha=float(ALPHA),
        influence_radius=OPT_INFLUENCE_RADIUS  # keep same influence at full-res for similar look
    )
    save_dark_as_image(preview_full, out("preview_lines.jpg"))

    print("[FINAL] Exporting lines.txt and SVG ...")
    export_lines_txt_counts(counts, out("lines.txt"))

    compact_svg = total_passes > SVG_COMPACT_IF_PASSES_OVER
    export_lines_svg(nails_mm, counts, out("preview_lines.svg"), compact=compact_svg)

    print("\nDONE ✔")
    print("Created:")
    print(f"  {out('target_gs.jpg')}")
    print(f"  {out('target_gs_opt.jpg')}")
    print(f"  {out('dither_fs.jpg')}")
    print(f"  {out('preview_lines.jpg')} + {out('preview_lines.svg')}")
    print(f"  {out('lines.txt')}")
    print(f"Stats: unique_segments={unique_segments}, total_passes={total_passes}")
