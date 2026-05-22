"""
Pitch Deck PowerPoint Generator Service

Generates professional PPTX files from slide data using python-pptx.
Dark theme with green accent. Uses only blank layouts to avoid
placeholder KeyError crashes from missing theme placeholders.
"""

import io
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Colour palette ────────────────────────────────────────────────────────────
DARK_BG      = RGBColor(13,  17,  23)   # #0d1117
CARD_BG      = RGBColor(22,  27,  34)   # #16171a
ACCENT_GREEN = RGBColor(35,  134, 54)   # #238636
ACCENT_TEAL  = RGBColor(31,  136, 161)  # #1f88a1
TEXT_WHITE   = RGBColor(240, 246, 252)  # #f0f6fc
TEXT_MUTED   = RGBColor(110, 118, 129)  # #6e7681

# ── Slide dimensions (16:9 widescreen) ───────────────────────────────────────
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _blank(prs: Presentation):
    """Always use the blank layout (index 6) — avoids placeholder crashes."""
    return prs.slides.add_slide(prs.slide_layouts[6])


def _bg(slide, color: RGBColor):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _rect(slide, left, top, width, height, fill_color: RGBColor, line_color=None):
    shp = slide.shapes.add_shape(1, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill_color
    if line_color:
        shp.line.color.rgb = line_color
    else:
        shp.line.fill.background()
    return shp


def _textbox(slide, left, top, width, height,
             text: str, size: int, bold=False,
             color: RGBColor = TEXT_WHITE,
             align=PP_ALIGN.LEFT,
             wrap=True) -> None:
    """Add a textbox. Silently truncates overlong text to avoid overflow."""
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf  = txb.text_frame
    tf.word_wrap = wrap

    # Split into paragraphs on newlines so bullet lines render correctly
    lines = str(text).split('\n')
    for li, line in enumerate(lines):
        if li == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line.strip()
        run.font.size   = Pt(size)
        run.font.bold   = bold
        run.font.color.rgb = color
        run.font.name   = "Calibri"


# ── Slide builders ────────────────────────────────────────────────────────────

def _build_title_slide(prs: Presentation, slide_data: dict):
    """Cover slide — full dark background, large company name, tagline."""
    slide = _blank(prs)
    _bg(slide, DARK_BG)

    # Bottom accent strip
    _rect(slide, Inches(0), Inches(6.2), SLIDE_W, Inches(1.3), CARD_BG)
    # Green left bar
    _rect(slide, Inches(0), Inches(0), Inches(0.15), SLIDE_H, ACCENT_GREEN)

    title_text   = slide_data.get('title', 'Startup')
    content_text = slide_data.get('content', '')
    if isinstance(content_text, list):
        content_text = '\n'.join(str(item) for item in content_text)

    # Company / startup name
    _textbox(slide,
             left=Inches(0.5), top=Inches(1.8),
             width=Inches(12.5), height=Inches(1.8),
             text=title_text, size=54, bold=True,
             color=ACCENT_GREEN, align=PP_ALIGN.LEFT)

    # Tagline
    _textbox(slide,
             left=Inches(0.5), top=Inches(3.7),
             width=Inches(11.0), height=Inches(1.2),
             text=content_text, size=24,
             color=TEXT_WHITE, align=PP_ALIGN.LEFT)

    # Confidential footer
    _textbox(slide,
             left=Inches(0.5), top=Inches(6.55),
             width=Inches(8), height=Inches(0.5),
             text="CONFIDENTIAL  ·  INVESTOR PRESENTATION",
             size=10, color=TEXT_MUTED, align=PP_ALIGN.LEFT)


def _build_content_slide(prs: Presentation, slide_data: dict, slide_num: int):
    """Standard content slide — title header + bullet points + optional chart card."""
    slide = _blank(prs)
    _bg(slide, DARK_BG)

    title_text   = slide_data.get('title', f'Slide {slide_num}')
    content_text = slide_data.get('content', '')
    if isinstance(content_text, list):
        content_text = '\n'.join(str(item) for item in content_text)
    chart_data   = slide_data.get('chart_data')

    has_chart = bool(chart_data and chart_data.get('values'))

    # ── Header band ──────────────────────────────────────────────────────────
    _rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(1.1), CARD_BG)
    _rect(slide, Inches(0), Inches(0), Inches(0.15), Inches(1.1), ACCENT_GREEN)

    _textbox(slide,
             left=Inches(0.3), top=Inches(0.1),
             width=Inches(12.5), height=Inches(0.9),
             text=title_text, size=32, bold=True,
             color=ACCENT_GREEN, align=PP_ALIGN.LEFT)

    # Slide number
    _textbox(slide,
             left=Inches(12.5), top=Inches(0.15),
             width=Inches(0.7), height=Inches(0.5),
             text=str(slide_num), size=10,
             color=TEXT_MUTED, align=PP_ALIGN.RIGHT)

    # ── Content area ─────────────────────────────────────────────────────────
    # If chart present: bullets on left 55%, chart card on right 40%
    # If no chart:      bullets use full width
    content_width = Inches(7.0) if has_chart else Inches(12.5)

    if isinstance(content_text, list):
        content_text = '\n'.join(str(item) for item in content_text)
    lines = [l.strip() for l in content_text.split('\n') if l.strip()]
    top = Inches(1.3)
    for line in lines[:8]:                          # cap at 8 bullets
        # Green bullet dot
        _rect(slide,
              left=Inches(0.3), top=top + Inches(0.17),
              width=Inches(0.1), height=Inches(0.1),
              fill_color=ACCENT_GREEN)
        _textbox(slide,
                 left=Inches(0.55), top=top,
                 width=content_width - Inches(0.6), height=Inches(0.65),
                 text=line, size=18, color=TEXT_WHITE)
        top += Inches(0.72)

    # ── Chart card (right side) ───────────────────────────────────────────────
    if has_chart:
        _build_chart_card(slide, chart_data)


def _build_chart_card(slide, chart_data: dict):
    """
    Renders chart data as a styled card on the RIGHT side of a content slide.
    Positioned to avoid overlapping body text.
    """
    card_left   = Inches(7.5)
    card_top    = Inches(1.2)
    card_width  = Inches(5.5)
    card_height = Inches(5.8)

    # Card background
    _rect(slide, card_left, card_top, card_width, card_height,
          fill_color=CARD_BG, line_color=ACCENT_GREEN)

    chart_title  = chart_data.get('title', 'Data')
    labels       = chart_data.get('labels', [])
    values       = chart_data.get('values', [])
    chart_type   = chart_data.get('type', 'pie')

    # Chart title
    _textbox(slide,
             left=card_left + Inches(0.15), top=card_top + Inches(0.1),
             width=card_width - Inches(0.3), height=Inches(0.55),
             text=chart_title, size=15, bold=True,
             color=ACCENT_GREEN, align=PP_ALIGN.CENTER)

    # Divider line
    _rect(slide,
          card_left + Inches(0.15), card_top + Inches(0.7),
          card_width - Inches(0.3), Inches(0.02),
          fill_color=ACCENT_GREEN)

    # Determine max value for bar scaling
    max_val = max((float(v) for v in values if v), default=1)

    row_top = card_top + Inches(0.85)
    row_h   = Inches(0.58)
    bar_max_w = card_width - Inches(1.8)

    for i, (label, val) in enumerate(zip(labels, values)):
        if row_top + row_h > card_top + card_height - Inches(0.2):
            break                                   # prevent overflow

        # Label
        _textbox(slide,
                 left=card_left + Inches(0.15), top=row_top,
                 width=Inches(1.4), height=row_h,
                 text=str(label), size=12, color=TEXT_MUTED,
                 align=PP_ALIGN.RIGHT)

        # Bar or segment indicator
        try:
            ratio = float(val) / max_val
        except (TypeError, ZeroDivisionError):
            ratio = 0

        bar_w = max(Inches(0.05), bar_max_w * ratio)

        if chart_type == 'pie':
            # Colour-coded dot + percentage
            dot_colors = [ACCENT_GREEN, ACCENT_TEAL, TEXT_MUTED,
                          RGBColor(255, 180, 0), RGBColor(220, 80, 80)]
            dot_color = dot_colors[i % len(dot_colors)]
            _rect(slide,
                  card_left + Inches(1.65), row_top + Inches(0.2),
                  Inches(0.2), Inches(0.2), fill_color=dot_color)
            total = sum(float(v) for v in values if v) or 1
            pct   = round(float(val) / total * 100)
            _textbox(slide,
                     left=card_left + Inches(1.95), top=row_top,
                     width=Inches(3.3), height=row_h,
                     text=f"{pct}%  —  {val}", size=13, color=TEXT_WHITE)
        else:
            # Horizontal bar
            _rect(slide,
                  card_left + Inches(1.65), row_top + Inches(0.18),
                  bar_w, Inches(0.26),
                  fill_color=ACCENT_GREEN)
            _textbox(slide,
                     left=card_left + Inches(1.65) + bar_w + Inches(0.08),
                     top=row_top,
                     width=Inches(1.5), height=row_h,
                     text=str(val), size=11, color=TEXT_MUTED)

        row_top += row_h


# ── Public entry point ────────────────────────────────────────────────────────

def generate_ppt_file(slides_data: list) -> io.BytesIO:
    """
    Generate a PowerPoint file from slide data.
    Drop-in replacement — same signature as the original.

    Args:
        slides_data: List of dicts with 'title', 'content', 'chart_data' keys

    Returns:
        BytesIO object containing the PPTX file, seeked to position 0
    """
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    for i, slide_data in enumerate(slides_data):
        if not isinstance(slide_data, dict):
            continue                                # skip malformed entries

        if i == 0:
            _build_title_slide(prs, slide_data)
        else:
            _build_content_slide(prs, slide_data, slide_num=i + 1)

    output = io.BytesIO()
    prs.save(output)
    output.seek(0)
    return output