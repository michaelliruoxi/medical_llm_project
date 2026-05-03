"""Polish the n=50 final presentation per docs/superpowers/specs/2026-04-27-n50-deck-polish-design.md.

Reads:  MedQuAD_Robustness_Final_Presentation_n50_updated.pptx
Writes: MedQuAD_Robustness_Final_Presentation_n50_polished.pptx
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "MedQuAD_Robustness_Final_Presentation_n50_updated.pptx"
DST = REPO / "MedQuAD_Robustness_Final_Presentation_n50_polished.pptx"

# ---------------------------------------------------------------------------
# Palette (committed)
# ---------------------------------------------------------------------------
TEAL = RGBColor(0x12, 0x7C, 0x7A)
SLATE = RGBColor(0x25, 0x32, 0x3D)
GOLD = RGBColor(0xE0, 0xA1, 0x3A)
CHERRY = RGBColor(0xC9, 0x4C, 0x4C)
GREEN = RGBColor(0x2D, 0x9D, 0x78)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SOFT_TEAL_BG = RGBColor(0xF2, 0xF6, 0xF7)
GRID = RGBColor(0xDC, 0xE9, 0xED)
MUTED_SLATE = RGBColor(0x52, 0x61, 0x6B)

# Colors to drop and the palette colors that replace them.
COLOR_REPLACEMENTS: dict[str, RGBColor] = {
    # random fills/text from the original deck → mapped to committed palette
    "3867D6": TEAL,
    "7A4B8F": GOLD,
    "0E5F60": TEAL,
    "EAF0FF": SOFT_TEAL_BG,
    "FCEBEB": SOFT_TEAL_BG,
    "FFF4DA": SOFT_TEAL_BG,
    "EAF7F1": SOFT_TEAL_BG,
    "F3F7F9": SOFT_TEAL_BG,
    "F7FAFC": SOFT_TEAL_BG,
    "E7F4F2": SOFT_TEAL_BG,
    "B9CED7": GRID,
    "31424E": SLATE,
    "17202A": SLATE,
    "2D9D78": GREEN,  # already palette but normalize representation
}

# Font floors
BODY_FLOOR_PT = 14.0
TABLE_FLOOR_PT = 12.0
FOOTER_PT = 10.0

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _norm_hex(rgb: RGBColor) -> str:
    return f"{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def _set_run_color(run, color: RGBColor) -> None:
    run.font.color.rgb = color


def _set_run_size(run, size_pt: float) -> None:
    run.font.size = Pt(size_pt)


def _iter_runs(text_frame):
    for paragraph in text_frame.paragraphs:
        for run in paragraph.runs:
            yield run


def _set_solid_fill(shape, color: RGBColor) -> None:
    shape.fill.solid()
    shape.fill.fore_color.rgb = color


def _set_no_line(shape) -> None:
    line = shape.line
    line.fill.background()


def _set_line(shape, color: RGBColor, width_pt: float = 0.75) -> None:
    line = shape.line
    line.color.rgb = color
    line.width = Pt(width_pt)


# ---------------------------------------------------------------------------
# Global palette + font enforcement
# ---------------------------------------------------------------------------
def _replace_colors_in_run(run) -> None:
    try:
        rgb = run.font.color.rgb
    except (AttributeError, KeyError):
        return
    if rgb is None:
        return
    key = _norm_hex(rgb)
    if key in COLOR_REPLACEMENTS:
        _set_run_color(run, COLOR_REPLACEMENTS[key])


def _replace_colors_in_shape_fill(shape) -> None:
    if not hasattr(shape, "fill"):
        return
    try:
        fill = shape.fill
        if fill.type != 1:  # solid
            return
        rgb = fill.fore_color.rgb
    except (AttributeError, KeyError):
        return
    if rgb is None:
        return
    key = _norm_hex(rgb)
    if key in COLOR_REPLACEMENTS:
        fill.fore_color.rgb = COLOR_REPLACEMENTS[key]


def _enforce_font_floor(run, floor_pt: float) -> None:
    if run.font.size is None:
        return
    if run.font.size.pt < floor_pt:
        _set_run_size(run, floor_pt)


EVIDENCE_SLIDE_INDICES = {11, 12, 13, 14}  # 0-indexed slides 12..15


def apply_global_palette_and_floors(prs: Presentation) -> None:
    """Pass 1: replace random palette colors and enforce font floors.

    Uses a lower body floor on the evidence-example slides (12-15) because those
    columns hold long answer text — bumping them to 14pt overflows the column box.
    """
    for slide_idx, slide in enumerate(prs.slides):
        # Evidence slides need a lower floor so column body text doesn't overflow
        body_floor = 11.0 if slide_idx in EVIDENCE_SLIDE_INDICES else BODY_FLOOR_PT
        for shape in slide.shapes:
            _replace_colors_in_shape_fill(shape)
            if shape.has_text_frame:
                # detect footer rows (small text near bottom): keep at FOOTER_PT
                is_footer = (
                    shape.top is not None
                    and Emu(shape.top).inches > 6.9
                    and Emu(shape.height).inches < 0.4
                )
                for run in _iter_runs(shape.text_frame):
                    _replace_colors_in_run(run)
                    if is_footer:
                        if run.font.size is None or run.font.size.pt < FOOTER_PT:
                            _set_run_size(run, FOOTER_PT)
                    else:
                        _enforce_font_floor(run, body_floor)


# ---------------------------------------------------------------------------
# Section motif: numbered teal circle next to section title
# ---------------------------------------------------------------------------
SECTION_TITLE_PATTERN = re.compile(r"^(\d{2})\s*\|\s*(.+)$") if False else None  # placeholder
# We can't import re lazily up top because we want a stable header. Inline below.
import re  # noqa: E402

SECTION_RE = re.compile(r"^\s*(\d{2})\s*\|\s*(.+?)\s*$")


def add_section_motif(slide, *, default_top_inches: float = 0.30) -> None:
    """Find the '01 | Problem' style title text and replace with a teal circle + clean title.

    Idempotent: if a circle already exists at that position, skip.
    """
    title_shape = None
    section_num = ""
    section_title = ""
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        txt = shape.text_frame.text.strip()
        m = SECTION_RE.match(txt)
        if m and shape.top is not None and Emu(shape.top).inches < 0.6:
            title_shape = shape
            section_num = m.group(1)
            section_title = m.group(2)
            break
    if title_shape is None:
        return

    # Replace the title shape's text with just the section title
    tf = title_shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = section_title
    run.font.name = "Aptos"
    run.font.bold = True
    run.font.size = Pt(20)
    run.font.color.rgb = SLATE

    # Shift title shape right to make room for the circle
    title_left_emu = title_shape.left
    title_top_emu = title_shape.top
    title_left_inches = Emu(title_left_emu).inches
    new_title_left = Inches(title_left_inches + 0.55)
    title_shape.left = new_title_left
    # widen if needed so existing text doesn't wrap
    try:
        title_shape.width = Inches(max(Emu(title_shape.width).inches, 5.5))
    except Exception:
        pass

    # Add the teal circle motif
    circle_size = Inches(0.42)
    circle = slide.shapes.add_shape(
        MSO_SHAPE.OVAL,
        Inches(title_left_inches),
        Inches(Emu(title_top_emu).inches - 0.05),
        circle_size,
        circle_size,
    )
    _set_solid_fill(circle, TEAL)
    _set_no_line(circle)
    circle.text_frame.text = section_num
    circle.text_frame.margin_left = Inches(0)
    circle.text_frame.margin_right = Inches(0)
    circle.text_frame.margin_top = Inches(0)
    circle.text_frame.margin_bottom = Inches(0)
    p_c = circle.text_frame.paragraphs[0]
    p_c.alignment = PP_ALIGN.CENTER
    run_c = p_c.runs[0]
    run_c.font.name = "Aptos"
    run_c.font.bold = True
    run_c.font.size = Pt(13)
    run_c.font.color.rgb = WHITE


# ---------------------------------------------------------------------------
# Slide 1: dark title, gold accent line, restyle pipeline circles
# ---------------------------------------------------------------------------
def _set_slide_background(slide, color: RGBColor) -> None:
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = color


def style_title_slide(slide) -> None:
    _set_slide_background(slide, SLATE)
    for shape in slide.shapes:
        if shape.has_text_frame:
            for run in _iter_runs(shape.text_frame):
                # white text by default on dark slide
                run.font.color.rgb = WHITE
        # Recolor the diagram container + pills regardless of empty text_frame.
        # AUTO_SHAPEs always carry a text_frame even when textually empty, so
        # we filter by geometry + text content rather than by text-frame presence.
        if not hasattr(shape, "fill"):
            continue
        if shape.left is None:
            continue
        text_content = (
            shape.text_frame.text.strip() if shape.has_text_frame else ""
        )
        try:
            left_in = Emu(shape.left).inches
            top_in = Emu(shape.top).inches
            width_in = Emu(shape.width).inches
            height_in = Emu(shape.height).inches
        except Exception:
            continue
        # Big outer container behind the pipeline diagram
        if (
            text_content == ""
            and left_in > 7.0
            and width_in > 4.5
            and height_in > 4.5
        ):
            _set_solid_fill(shape, RGBColor(0x2F, 0x3F, 0x4D))
            _set_no_line(shape)
            continue
        # The four pipeline pills (wide-and-short, empty rectangles)
        if (
            text_content == ""
            and 7.4 < left_in < 7.7
            and 0.5 < height_in < 0.7
            and width_in > 3.5
        ):
            _set_solid_fill(shape, TEAL)
            _set_no_line(shape)
            continue

    # Recolor pipeline label text (sits inside teal pills) — make it crisp white + bold
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        if shape.left is None:
            continue
        if 7.7 < Emu(shape.left).inches < 7.9 and 1.3 < Emu(shape.top).inches < 4.2:
            for run in _iter_runs(shape.text_frame):
                run.font.color.rgb = WHITE
                run.font.bold = True
                if run.font.size is None or run.font.size.pt < 14:
                    run.font.size = Pt(14)

    # Add a gold accent line under the headline
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.52),
        Inches(2.50),
        Inches(1.20),
        Inches(0.06),
    )
    _set_solid_fill(accent, GOLD)
    _set_no_line(accent)


# ---------------------------------------------------------------------------
# Slide 2: bump 11/50/5 stat numbers to 48pt gold
# ---------------------------------------------------------------------------
STAT_NUMBER_TEXTS = {"11", "50", "5", "550"}


def style_stat_numbers(slide) -> None:
    """Bump stat numbers to 36pt gold. Widen the frame so '550' doesn't wrap and
    move the box up so it doesn't crash into the labels positioned right below.
    """
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        txt = shape.text_frame.text.strip()
        if txt in STAT_NUMBER_TEXTS and Emu(shape.height).inches < 0.7:
            for run in _iter_runs(shape.text_frame):
                run.font.size = Pt(36)
                run.font.bold = True
                run.font.color.rgb = GOLD
            # widen so 3-digit "550" doesn't wrap; lift up so descender doesn't hit label
            try:
                cur_height = Emu(shape.height).inches
                shape.height = Inches(max(cur_height, 0.55))
                cur_width = Emu(shape.width).inches
                shape.width = Inches(max(cur_width, 1.5))
                shape.top = Inches(Emu(shape.top).inches - 0.10)
            except Exception:
                pass
            shape.text_frame.margin_top = Inches(0)
            shape.text_frame.margin_bottom = Inches(0)


# ---------------------------------------------------------------------------
# Slide 8: hero result — bigger chart, gold callout for averages, bigger headline
# ---------------------------------------------------------------------------
def style_headline_slide(slide) -> None:
    # First pass: identify and DELETE the duplicate "Mode averages / fixed_repair / 3.73 ..."
    # text boxes AND the empty container box that holds them, before resizing the chart.
    shapes_to_delete = []
    for shape in slide.shapes:
        if shape.left is None:
            continue
        left_in = Emu(shape.left).inches
        top_in = Emu(shape.top).inches
        width_in = Emu(shape.width).inches
        height_in = Emu(shape.height).inches
        # Empty rounded-rectangle container that originally held the mode-averages text
        if (
            not shape.has_text_frame
            and 8.5 < left_in < 9.6
            and 1.5 < top_in < 2.2
            and width_in > 3.0
            and height_in > 3.5
        ):
            shapes_to_delete.append(shape)
            continue
        if shape.has_text_frame:
            txt = shape.text_frame.text.strip()
            if (
                "3.73 clean" in txt
                or txt == "Mode averages"
                or txt in {"fixed_repair", "self_repair"}
            ):
                if 8.5 < left_in < 9.6 and 1.5 < top_in < 5.5:
                    shapes_to_delete.append(shape)
                    continue
            # Also delete the empty container (rectangle holding the original text block)
            if (
                txt == ""
                and 8.5 < left_in < 9.6
                and 1.5 < top_in < 2.2
                and width_in > 3.0
                and height_in > 3.5
            ):
                shapes_to_delete.append(shape)
    for shape in shapes_to_delete:
        sp = shape._element
        sp.getparent().remove(sp)

    # Move the bottom "Interpretation: ..." footer text up to a position that won't be covered
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        txt = shape.text_frame.text.strip()
        if txt.startswith("Interpretation: repair is not"):
            shape.left = Inches(0.6)
            shape.top = Inches(6.55)
            shape.width = Inches(11.5)
            shape.height = Inches(0.45)
            for run in _iter_runs(shape.text_frame):
                run.font.size = Pt(12)
                run.font.italic = True
                run.font.color.rgb = MUTED_SLATE

    # Resize/reposition the chart - keep it from overlapping the right callout
    for shape in slide.shapes:
        if shape.has_chart:
            shape.left = Inches(0.5)
            shape.top = Inches(2.05)
            shape.width = Inches(8.6)
            shape.height = Inches(4.3)
            _retheme_chart(shape.chart, ["clean", "noisy", "repaired"])
            break

    # Reduce the headline to 26pt and widen its frame so it stays single-line
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        if shape.top is None:
            continue
        top_in = Emu(shape.top).inches
        txt = shape.text_frame.text.strip()
        if 0.6 < top_in < 1.0 and txt.startswith("The n=50 pilot"):
            try:
                shape.width = Inches(12.3)
            except Exception:
                pass
            for run in _iter_runs(shape.text_frame):
                run.font.size = Pt(26)
                run.font.bold = True
        # Move the subhead down a bit if it sits right under the headline
        if 1.35 < top_in < 1.60:
            try:
                shape.top = Inches(1.65)
            except Exception:
                pass

    # Add the unified gold callout with mode averages
    callout = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(9.35),
        Inches(2.10),
        Inches(3.55),
        Inches(2.40),
    )
    _set_solid_fill(callout, RGBColor(0xFD, 0xF6, 0xE7))
    _set_line(callout, GOLD, 1.5)
    tf = callout.text_frame
    tf.clear()
    tf.margin_left = Inches(0.18)
    tf.margin_right = Inches(0.18)
    tf.margin_top = Inches(0.15)
    tf.margin_bottom = Inches(0.15)
    tf.word_wrap = True

    def _line(text: str, *, size: float = 12, bold: bool = False, color: RGBColor = SLATE, italic: bool = False, first: bool = False):
        if first:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        r = p.add_run()
        r.text = text
        r.font.name = "Aptos"
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color

    _line("Mode averages", size=14, bold=True, color=GOLD, first=True)
    _line("fixed_repair", size=12, bold=True, color=SLATE)
    _line("3.73 → 3.47 → 3.35", size=12, color=MUTED_SLATE)
    _line("self_repair", size=12, bold=True, color=SLATE)
    _line("3.73 → 3.47 → 3.41", size=12, color=MUTED_SLATE)


# ---------------------------------------------------------------------------
# Chart retheme (shared)
# ---------------------------------------------------------------------------
def _disable_invert_if_negative(series) -> None:
    """Force a chart series to keep its fill color on negative values.

    PowerPoint's "invertIfNegative" defaults to True for bar/column charts,
    which renders negative bars as outlines only. We patch the series XML
    to set <c:invertIfNegative val="0"/>, inserted at the correct schema
    position (after c:spPr, before c:cat/c:val/c:dPt).
    """
    from lxml import etree

    ser = series._element
    C_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"
    nsmap = {"c": C_NS}
    invert = ser.find("c:invertIfNegative", nsmap)
    if invert is not None:
        invert.set("val", "0")
        return

    # Need to insert in the right position. Per the c:ser schema:
    # idx, order, tx, spPr, invertIfNegative, dPt*, dLbls, trendline*, errBars, cat, val, smooth
    insert_before = None
    for tag in ("c:dPt", "c:dLbls", "c:trendline", "c:errBars", "c:cat", "c:val", "c:smooth"):
        nodes = ser.findall(tag, nsmap)
        if nodes:
            insert_before = nodes[0]
            break

    new_el = etree.SubElement(ser, qn("c:invertIfNegative"))
    new_el.set("val", "0")
    if insert_before is not None:
        ser.remove(new_el)
        insert_before.addprevious(new_el)


def _color_negative_data_points(series, negative_color: RGBColor) -> None:
    """Add per-data-point overrides so negative bars are filled with `negative_color`.

    Belt-and-suspenders alongside _disable_invert_if_negative: even if PowerPoint's
    invertIfNegative behavior gets re-enabled, the explicit dPt fills will still apply.
    """
    from lxml import etree

    ser = series._element
    C_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"
    A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
    values = list(series.values)

    # Remove any existing dPt to keep idempotent.
    for old in ser.findall("c:dPt", {"c": C_NS}):
        ser.remove(old)

    # Build dPt for each negative value
    insert_before = None
    for tag in ("c:dLbls", "c:cat", "c:val"):
        node = ser.find(tag, {"c": C_NS})
        if node is not None:
            insert_before = node
            break

    hex_color = f"{negative_color[0]:02X}{negative_color[1]:02X}{negative_color[2]:02X}"
    for idx, val in enumerate(values):
        if val is None or val >= 0:
            continue
        dpt_xml = (
            f'<c:dPt xmlns:c="{C_NS}" xmlns:a="{A_NS}">'
            f'<c:idx val="{idx}"/>'
            f'<c:invertIfNegative val="0"/>'
            f'<c:bubble3D val="0"/>'
            f'<c:spPr>'
            f'<a:solidFill><a:srgbClr val="{hex_color}"/></a:solidFill>'
            f'<a:ln><a:noFill/></a:ln>'
            f'</c:spPr>'
            f'</c:dPt>'
        )
        dpt = etree.fromstring(dpt_xml)
        if insert_before is not None:
            insert_before.addprevious(dpt)
        else:
            ser.append(dpt)


def _retheme_chart(chart, expected_categories: list[str] | None = None) -> None:
    """Apply palette colors to a chart's series and lighten gridlines."""
    # Hide chart title (slide already has one)
    try:
        chart.has_title = False
    except Exception:
        pass

    # Color series by name
    series_color_map = {
        "clean": TEAL,
        "noisy": MUTED_SLATE,
        "repaired": GOLD,
        "fixed_repair": TEAL,
        "self_repair": GOLD,
    }

    for ser in chart.series:
        name = (ser.name or "").strip().lower()
        color = series_color_map.get(name)
        if color is None:
            continue
        fill = ser.format.fill
        fill.solid()
        fill.fore_color.rgb = color
        try:
            ser.format.line.fill.background()
        except Exception:
            pass

    # Lighten gridlines and axis text
    try:
        cat_ax = chart.category_axis
        val_ax = chart.value_axis
        for ax in (cat_ax, val_ax):
            ax.tick_labels.font.size = Pt(11)
            ax.tick_labels.font.color.rgb = SLATE
            try:
                if ax.has_major_gridlines:
                    gl = ax.major_gridlines
                    gl.format.line.color.rgb = GRID
                    gl.format.line.width = Pt(0.5)
            except Exception:
                pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Slides 9, 10: per-model and per-noise — replace with horizontal sorted bars
# ---------------------------------------------------------------------------
def _replace_chart_with_horizontal_sorted(slide, *, source_chart_shape, sort_by_series: str = "self_repair") -> None:
    """Take an existing column chart, extract its data, sort by `sort_by_series` desc,
    and replace with a horizontal bar chart (XL_CHART_TYPE.BAR_CLUSTERED) at the same position.
    """
    from pptx.enum.chart import XL_CHART_TYPE

    chart = source_chart_shape.chart
    # Extract data
    plots = list(chart.plots)
    if not plots:
        return
    plot = plots[0]
    categories = list(plot.categories)
    series_data: dict[str, list[float]] = {}
    for ser in chart.series:
        series_data[(ser.name or "").strip()] = list(ser.values)

    if sort_by_series not in series_data:
        # fall back to the first series
        sort_by_series = next(iter(series_data.keys()))

    sort_vals = series_data[sort_by_series]
    order = sorted(range(len(categories)), key=lambda i: sort_vals[i], reverse=False)
    # Why reverse=False: in horizontal bar charts, the first category appears at the bottom.
    # We want the largest positive value at the TOP, so we sort ascending here.
    new_categories = [categories[i] for i in order]
    new_series = {name: [vals[i] for i in order] for name, vals in series_data.items()}

    # Position
    left, top, width, height = (
        source_chart_shape.left,
        source_chart_shape.top,
        source_chart_shape.width,
        source_chart_shape.height,
    )

    # Build new chart data
    chart_data = CategoryChartData()
    chart_data.categories = new_categories
    for name, vals in new_series.items():
        chart_data.add_series(name, vals)

    # Remove old chart shape
    sp = source_chart_shape._element
    sp.getparent().remove(sp)

    # Add horizontal bar chart
    new_chart_shape = slide.shapes.add_chart(
        XL_CHART_TYPE.BAR_CLUSTERED,
        left,
        top,
        width,
        height,
        chart_data,
    )
    new_chart = new_chart_shape.chart
    _retheme_chart(new_chart)
    # Color series and override per-point fills for negative values so the
    # winners-vs-losers reading is unambiguous: positive=series color, negative=cherry.
    for ser in new_chart.series:
        name = (ser.name or "").strip().lower()
        fill = ser.format.fill
        fill.solid()
        if name == "fixed_repair":
            fill.fore_color.rgb = TEAL
        elif name == "self_repair":
            fill.fore_color.rgb = GOLD
        _disable_invert_if_negative(ser)
        _color_negative_data_points(ser, CHERRY)


def style_per_model_or_noise_slide(slide) -> None:
    for shape in list(slide.shapes):
        if shape.has_chart:
            _replace_chart_with_horizontal_sorted(slide, source_chart_shape=shape, sort_by_series="self_repair")
            return


# ---------------------------------------------------------------------------
# Slide 19: thank you — dark bg + gold accent
# ---------------------------------------------------------------------------
def style_thank_you_slide(slide) -> None:
    _set_slide_background(slide, SLATE)
    for shape in slide.shapes:
        if shape.has_text_frame:
            for run in _iter_runs(shape.text_frame):
                run.font.color.rgb = WHITE
            txt = shape.text_frame.text.strip()
            if txt.lower().startswith("thank"):
                for run in _iter_runs(shape.text_frame):
                    run.font.size = Pt(54)
                    run.font.bold = True
            if "Repair improved question intent" in txt:
                # gold accent the key sentence
                for run in _iter_runs(shape.text_frame):
                    run.font.color.rgb = GOLD
                    run.font.size = Pt(16)
            if "pushback" in txt or "counter-evidence" in txt:
                for run in _iter_runs(shape.text_frame):
                    run.font.italic = True
                    run.font.color.rgb = GRID


# ---------------------------------------------------------------------------
# Per-slide table row striping (slides 5, 6, 17 — best-effort)
# ---------------------------------------------------------------------------
def stripe_table_like_rows(slide, *, left_min: float, left_max: float, top_min: float, top_max: float) -> None:
    """Find narrow rectangular shapes (table row backgrounds) and apply alternating fills."""
    rows = []
    for shape in slide.shapes:
        if shape.has_text_frame:
            continue
        if shape.left is None or shape.top is None:
            continue
        try:
            l = Emu(shape.left).inches
            t = Emu(shape.top).inches
            h = Emu(shape.height).inches
        except Exception:
            continue
        if left_min <= l <= left_max and top_min <= t <= top_max and 0.30 < h < 0.55:
            rows.append((t, shape))
    rows.sort()
    for idx, (_, shape) in enumerate(rows):
        if idx == 0:
            _set_solid_fill(shape, TEAL)  # header
            _set_no_line(shape)
            continue
        if idx % 2 == 0:
            _set_solid_fill(shape, SOFT_TEAL_BG)
        else:
            _set_solid_fill(shape, WHITE)
        _set_no_line(shape)


def style_setup_slide(slide) -> None:
    style_stat_numbers(slide)
    # Stripe the two adjacent tables on slide 5
    stripe_table_like_rows(slide, left_min=0.45, left_max=0.60, top_min=3.5, top_max=6.5)
    stripe_table_like_rows(slide, left_min=7.95, left_max=8.15, top_min=3.5, top_max=6.5)
    # Recolor header-row text to white if header background turned teal
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        if shape.top is None:
            continue
        top_in = Emu(shape.top).inches
        left_in = Emu(shape.left).inches
        if 3.65 <= top_in <= 3.78 and (0.55 <= left_in <= 8.20):
            for run in _iter_runs(shape.text_frame):
                run.font.color.rgb = WHITE
                run.font.bold = True

    # The right-side "Model families" column is too narrow at 12pt and "Open instruction"
    # wraps into the next row. Widen the family-name column from 1.45" to 1.85" by
    # shifting the "Examples" column 0.40" to the right (and shrinking it accordingly).
    for shape in slide.shapes:
        if shape.left is None:
            continue
        left_in = Emu(shape.left).inches
        top_in = Emu(shape.top).inches
        if 8.05 <= left_in <= 8.15 and 3.6 <= top_in <= 6.5:
            # this shape is a Model families column cell (text or row-bg)
            try:
                shape.width = Inches(max(Emu(shape.width).inches, 1.85))
            except Exception:
                pass
        if 9.60 <= left_in <= 9.70 and 3.6 <= top_in <= 6.5:
            # this shape is an Examples column cell - shift right
            try:
                shape.left = Inches(left_in + 0.40)
                shape.width = Inches(max(Emu(shape.width).inches - 0.40, 2.30))
            except Exception:
                pass


def style_metrics_slide(slide) -> None:
    """Slide 6: alternate row shading + 12pt floor."""
    stripe_table_like_rows(slide, left_min=0.45, left_max=0.60, top_min=1.7, top_max=6.5)
    # Header row text white
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        if shape.top is None:
            continue
        top_in = Emu(shape.top).inches
        if 1.78 <= top_in <= 1.92:
            for run in _iter_runs(shape.text_frame):
                run.font.color.rgb = WHITE
                run.font.bold = True


def style_limitations_slide(slide) -> None:
    """Slide 17: alternating row shade."""
    stripe_table_like_rows(slide, left_min=0.45, left_max=0.60, top_min=1.7, top_max=6.5)
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        if shape.top is None:
            continue
        top_in = Emu(shape.top).inches
        if 1.78 <= top_in <= 1.92:
            for run in _iter_runs(shape.text_frame):
                run.font.color.rgb = WHITE
                run.font.bold = True


# ---------------------------------------------------------------------------
# Evidence example slides 12-15: column header chips + stat bar
# ---------------------------------------------------------------------------
EVIDENCE_HEADER_CHIPS = {
    "CLEAN ANSWER": (TEAL, "CLEAN"),
    "NOISY ANSWER": (CHERRY, "NOISY"),
    "REPAIRED": (GOLD, "REPAIRED"),
}


def style_evidence_slide(slide) -> None:
    """Find the three column header text boxes and turn them into colored chips.

    Replaces text with shorter labels (CLEAN/NOISY/REPAIRED) and widens the
    text frame so the labels don't get truncated.
    """
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        txt = shape.text_frame.text.strip()
        if txt not in EVIDENCE_HEADER_CHIPS:
            continue
        color, short_label = EVIDENCE_HEADER_CHIPS[txt]
        _set_solid_fill(shape, color)
        _set_no_line(shape)
        # Replace text and reset formatting in one place
        tf = shape.text_frame
        tf.clear()
        tf.margin_left = Inches(0.12)
        tf.margin_right = Inches(0.12)
        tf.margin_top = Inches(0.03)
        tf.margin_bottom = Inches(0.03)
        tf.word_wrap = False
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = short_label
        r.font.name = "Aptos"
        r.font.color.rgb = WHITE
        r.font.bold = True
        r.font.size = Pt(12)
        # Size the chip tightly to its label so it doesn't bleed past the column card
        try:
            target_w = {"CLEAN": 0.95, "NOISY": 0.95, "REPAIRED": 1.30}[short_label]
            shape.width = Inches(target_w)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Slide 11: metric tension — turn improved/worse/flat counts into colored badges
# ---------------------------------------------------------------------------
TENSION_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*$")


def style_metric_tension_slide(slide) -> None:
    """Find shapes that match 'X/Y/Z' (improved/worse/flat) and split into 3 colored badges.

    Strategy: rather than re-laying out, just color each segment by replacing the run text
    with three runs and coloring them green/cherry/grey. Also shrink the slide-11 headline
    to 26pt and widen its frame so the long sentence stays on one line.
    """
    # Shrink the wrapping headline ("Repair cleans up intent more reliably than it improves answers.")
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        if shape.top is None:
            continue
        top_in = Emu(shape.top).inches
        if 0.6 < top_in < 1.3:
            txt = shape.text_frame.text.strip()
            if txt.startswith("Repair cleans up"):
                try:
                    shape.width = Inches(12.3)
                except Exception:
                    pass
                for run in _iter_runs(shape.text_frame):
                    if run.font.size and run.font.size.pt >= 16:
                        run.font.size = Pt(24)
                        run.font.bold = True

    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        txt = shape.text_frame.text.strip()
        m = TENSION_RE.match(txt)
        if not m:
            continue
        improved, worse, flat = m.group(1), m.group(2), m.group(3)
        tf = shape.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER

        def _run(text: str, color: RGBColor, *, bold=True):
            r = p.add_run()
            r.text = text
            r.font.name = "Aptos"
            r.font.size = Pt(13)
            r.font.bold = bold
            r.font.color.rgb = color

        _run(improved, GREEN)
        _run(" / ", MUTED_SLATE, bold=False)
        _run(worse, CHERRY)
        _run(" / ", MUTED_SLATE, bold=False)
        _run(flat, MUTED_SLATE)


# ---------------------------------------------------------------------------
# Slide 16: numbered insights get teal motif circle
# ---------------------------------------------------------------------------
def style_interpretation_slide(slide) -> None:
    """Find the '01', '02', '03' numbered marks and turn each into a teal motif badge.

    Also reduce headline size on the wrapping insights ("Question repair adds another
    failure mode.") to 14pt so they stay on one line above their description text.
    """
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        txt = shape.text_frame.text.strip()
        if txt in {"01", "02", "03"} and Emu(shape.width).inches < 1.5:
            _set_solid_fill(shape, TEAL)
            _set_no_line(shape)
            tf = shape.text_frame
            for run in _iter_runs(tf):
                run.font.color.rgb = WHITE
                run.font.bold = True
                run.font.size = Pt(16)
            for p in tf.paragraphs:
                p.alignment = PP_ALIGN.CENTER

    # Shrink each insight headline so it stays on one line
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        txt = shape.text_frame.text.strip()
        if txt in {
            "Repair is useful, but bounded.",
            "Question repair adds another failure mode.",
            "Metric choice changes the story.",
        }:
            try:
                shape.width = Inches(11.0)
            except Exception:
                pass
            for run in _iter_runs(shape.text_frame):
                run.font.size = Pt(14)
                run.font.bold = True
                run.font.color.rgb = SLATE


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)

    shutil.copy2(SRC, DST)
    prs = Presentation(DST)

    # Pass 1: global palette + font floors
    apply_global_palette_and_floors(prs)

    # Per-slide passes (1-indexed)
    slides = list(prs.slides)
    style_title_slide(slides[0])  # 1

    for idx in range(1, 18):  # slides 2..18
        add_section_motif(slides[idx])

    style_stat_numbers(slides[1])  # 2
    style_setup_slide(slides[4])   # 5
    style_metrics_slide(slides[5]) # 6
    style_headline_slide(slides[7])  # 8
    style_per_model_or_noise_slide(slides[8])  # 9
    style_per_model_or_noise_slide(slides[9])  # 10
    style_metric_tension_slide(slides[10])  # 11
    for ev_idx in (11, 12, 13, 14):  # 12..15
        style_evidence_slide(slides[ev_idx])
    style_interpretation_slide(slides[15])  # 16
    style_limitations_slide(slides[16])     # 17
    style_thank_you_slide(slides[18])       # 19

    prs.save(DST)
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
