# split_to_2xA3.py
# pip install pypdf

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject

PT_PER_MM = 72.0 / 25.4

def mm_to_pt(mm: float) -> float:
    return mm * PT_PER_MM

def split_template_to_2xA3_landscape(
    in_pdf: str,
    out_pdf: str,
    overlap_mm: float = 10.0,   # 0..15mm typical (for taping)
    align: str = "bottom",      # "bottom" or "center"
):
    """
    Input: 1-page PDF (your 460x515mm template).
    Output: 2 pages A3 landscape stacked vertically.
    Note: It scales down to fit A3 width (420mm). For 460mm width => ~91.3% scale.
    """
    # A3 landscape
    A3_W_MM, A3_H_MM = 420.0, 297.0
    page_w = mm_to_pt(A3_W_MM)
    page_h = mm_to_pt(A3_H_MM)
    overlap = mm_to_pt(overlap_mm)

    reader = PdfReader(in_pdf)
    if len(reader.pages) != 1:
        raise ValueError("Expected a single-page PDF as input.")

    src = reader.pages[0]
    src_w = float(src.mediabox.width)
    src_h = float(src.mediabox.height)

    # scale to fit A3 width
    scale = page_w / src_w
    scaled_w = src_w * scale
    scaled_h = src_h * scale

    # center horizontally on A3
    x_off = (page_w - scaled_w) / 2.0

    # treat two A3 pages as one tall canvas
    total_h = (2 * page_h) - overlap

    if align == "center":
        y_base_total = (total_h - scaled_h) / 2.0
    else:
        y_base_total = 0.0

    top = PageObject.create_blank_page(width=page_w, height=page_h)
    bottom = PageObject.create_blank_page(width=page_w, height=page_h)

    def merge_on(dst: PageObject, y0_total: float):
        # place scaled content so that we see the right slice of the tall canvas
        y_off = y_base_total - y0_total
        t = Transformation().scale(scale, scale).translate(x_off, y_off)
        dst.merge_transformed_page(src, t)

    # bottom page shows total y in [0, page_h]
    merge_on(bottom, y0_total=0.0)

    # top page shows total y in [page_h - overlap, page_h - overlap + page_h]
    merge_on(top, y0_total=(page_h - overlap))

    writer = PdfWriter()
    writer.add_page(top)      # page 1 = top half
    writer.add_page(bottom)   # page 2 = bottom half

    with open(out_pdf, "wb") as f:
        writer.write(f)

    print(f"OK: {out_pdf}")
    print(f"Scale used: {scale*100:.2f}%")
    print(f"Overlap: {overlap_mm} mm")

if __name__ == "__main__":
    split_template_to_2xA3_landscape(
        in_pdf="template_460x515_N180_radial_digits.pdf",
        out_pdf="template_2xA3.pdf",
        overlap_mm=10.0,
        align="bottom",
    )
