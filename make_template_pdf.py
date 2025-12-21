import math
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors


def bottom_index_from_range(n_nails: int, start: int = 100) -> int:
    """
    bottom nail index = midpoint of [start..n_nails], rounded.
      n=180 => 140
      n=190 => 145
    """
    mid = (start + n_nails) / 2.0
    b = int(round(mid))
    return max(1, min(n_nails, b))


def make_nail_template_pdf(
    out_path: str,
    canvas_w_mm: float = 460,
    canvas_h_mm: float = 515,
    n_nails: int = 180,
    margin_mm: float = 20,
    nail_radius_mm: float = 1.6,

    # rotation rule: bottom = midpoint of [bottom_range_start..N]
    bottom_range_start: int = 100,
    clockwise: bool = False,

    # --- digits on radial line (from center) ---
    # offsets measured from the NAIL position outward (along radius)
    units_offset_mm: float = 10.0,
    tens_offset_mm: float = 18.0,
    hundreds_offset_mm: float = 26.0,

    units_font_mm: float = 6.0,
    tens_font_mm: float = 6.0,
    hundreds_font_mm: float = 6.0,

    # show hundreds digit: "always" | "auto" | "never"
    show_hundreds_mode: str = "auto",

    # readability halo
    halo_r_mm: float = 3.6,
):
    page_size = (canvas_w_mm * mm, canvas_h_mm * mm)
    c = canvas.Canvas(out_path, pagesize=page_size)

    # Background
    c.setFillColor(colors.white)
    c.rect(0, 0, canvas_w_mm * mm, canvas_h_mm * mm, stroke=0, fill=1)

    # Calibration line: 100 mm
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.6 * mm)
    c.line(20 * mm, 20 * mm, 120 * mm, 20 * mm)
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.black)
    c.drawString(22 * mm, 24 * mm, "100 mm")

    # Circle parameters
    cx = canvas_w_mm / 2.0
    cy = canvas_h_mm / 2.0
    r = min(canvas_w_mm, canvas_h_mm) / 2.0 - margin_mm

    # Choose which index goes to bottom (6 o'clock)
    bottom_index = bottom_index_from_range(n_nails, bottom_range_start)

    # angle(idx) = start_angle + dir * 2*pi*(idx-1)/N
    # enforce angle(bottom_index) = -pi/2
    dir_sign = -1.0 if clockwise else 1.0
    start_angle = (-math.pi / 2.0) - dir_sign * (2.0 * math.pi * (bottom_index - 1) / n_nails)

    def draw_digit(ch: str, x_mm: float, y_mm: float, font_mm: float):
        font_pt = font_mm * mm
        c.setFont("Helvetica", font_pt)

        # white halo behind digit
        c.setFillColor(colors.white)
        c.circle(x_mm * mm, y_mm * mm, halo_r_mm * mm, stroke=0, fill=1)
        c.setFillColor(colors.black)

        tw = c.stringWidth(ch, "Helvetica", font_pt)
        c.drawString(x_mm * mm - tw / 2.0, y_mm * mm - (font_pt / 3.0), ch)

    # Draw nails + digits
    c.setStrokeColor(colors.black)
    c.setFillColor(colors.black)

    for k in range(n_nails):
        idx = k + 1
        a = start_angle + dir_sign * (2.0 * math.pi * k / n_nails)

        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)

        # nail dot
        c.circle(x * mm, y * mm, nail_radius_mm * mm, stroke=1, fill=1)

        # radial unit vector outward (from center to nail)
        dx, dy = x - cx, y - cy
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L

        # digits
        units = idx % 10
        tens = (idx // 10) % 10
        hundreds = (idx // 100) % 10  # 0/1 for 1..180, 0/1 for 1..199 etc.

        # positions ALONG THE RADIAL LINE (center -> nail -> outward)
        xu, yu = x + ux * units_offset_mm, y + uy * units_offset_mm
        xt, yt = x + ux * tens_offset_mm,  y + uy * tens_offset_mm
        xh, yh = x + ux * hundreds_offset_mm, y + uy * hundreds_offset_mm

        # draw units + tens always
        draw_digit(str(units), xu, yu, units_font_mm)
        draw_digit(str(tens),  xt, yt, tens_font_mm)

        # hundreds depending on mode
        show_h = (
            show_hundreds_mode == "always"
            or (show_hundreds_mode == "auto" and idx >= 100)
        )
        if show_hundreds_mode != "never" and show_h:
            draw_digit(str(hundreds), xh, yh, hundreds_font_mm)
        elif show_hundreds_mode == "always":
            # for perfect symmetry even below 100, uncomment this instead:
            # draw_digit(str(hundreds), xh, yh, hundreds_font_mm)
            pass

    # Optional debug text
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.black)
    c.drawString(
        20 * mm, (canvas_h_mm - 12) * mm,
        f"N={n_nails}, bottom=mid({bottom_range_start}..{n_nails})={bottom_index}, clockwise={clockwise}"
    )

    c.showPage()
    c.save()


if __name__ == "__main__":
    # Example: N=180 => bottom 140
    make_nail_template_pdf(
        out_path="template_460x515_N180_radial_digits.pdf",
        n_nails=180,
        bottom_range_start=100,
        clockwise=False,
        show_hundreds_mode="auto",  # only from 100+
    )
    print("Created: template_460x515_N180_radial_digits.pdf")

    # Example: N=190 => bottom 145
    make_nail_template_pdf(
        out_path="template_460x515_N190_radial_digits.pdf",
        n_nails=190,
        bottom_range_start=100,
        clockwise=False,
        show_hundreds_mode="auto",
    )
    print("Created: template_460x515_N190_radial_digits.pdf")
