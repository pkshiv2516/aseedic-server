"""
Pitch Deck PowerPoint Generator Service

Generates professional PPTX files from slide data using python-pptx.
Dark theme with green accent, supports charts and structured content.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import io

# Define Colors - Dark Theme
DARK_BG = RGBColor(13, 17, 23)       # #0d1117 (Dark Blue/Grey)
TEXT_WHITE = RGBColor(240, 246, 252)  # #f0f6fc
ACCENT_GREEN = RGBColor(35, 134, 54)  # #238636


def apply_slide_theme(slide):
    """Applies a dark background to the slide."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = DARK_BG


def generate_ppt_file(slides_data: list) -> io.BytesIO:
    """
    Generate a PowerPoint file from slide data.
    
    Args:
        slides_data: List of dictionaries with 'title', 'content', 'chart_data' keys
        
    Returns:
        BytesIO object containing the PPTX file
    """
    prs = Presentation()
    
    # Set slide dimensions (16:9 aspect ratio)
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    for i, slide_data in enumerate(slides_data):
        title_text = slide_data.get('title', 'Slide')
        content_text = slide_data.get('content', '')
        chart_data = slide_data.get('chart_data')
        
        if i == 0:
            # --- TITLE SLIDE ---
            slide_layout = prs.slide_layouts[0]  # Title Slide
            slide = prs.slides.add_slide(slide_layout)
            apply_slide_theme(slide)
            
            title = slide.shapes.title
            subtitle = slide.placeholders[1]
            
            # Style Title
            title.text = title_text
            title.text_frame.paragraphs[0].font.color.rgb = ACCENT_GREEN
            title.text_frame.paragraphs[0].font.bold = True
            title.text_frame.paragraphs[0].font.size = Pt(54)
            
            # Style Subtitle
            subtitle.text = content_text
            subtitle.text_frame.paragraphs[0].font.color.rgb = TEXT_WHITE
            subtitle.text_frame.paragraphs[0].font.size = Pt(24)

        else:
            # --- CONTENT SLIDE ---
            slide_layout = prs.slide_layouts[1]  # Title and Content
            slide = prs.slides.add_slide(slide_layout)
            apply_slide_theme(slide)
            
            # Title Shape
            title = slide.shapes.title
            title.text = title_text
            title.text_frame.paragraphs[0].font.color.rgb = ACCENT_GREEN
            title.text_frame.paragraphs[0].font.size = Pt(40)
            title.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
            
            # Content Body
            body_shape = slide.placeholders[1]
            tf = body_shape.text_frame
            tf.text = content_text
            
            # Format Paragraphs
            for p in tf.paragraphs:
                p.font.color.rgb = TEXT_WHITE
                p.font.size = Pt(20)
                p.space_after = Pt(10)
            
            # Add Chart Data Box if present
            if chart_data and chart_data.get('values'):
                left = Inches(8)
                top = Inches(2)
                width = Inches(4.5)
                height = Inches(4)
                
                shape = slide.shapes.add_shape(
                    1,  # msoShapeRectangle
                    left, top, width, height
                )
                
                # Style the box
                fill = shape.fill
                fill.solid()
                fill.fore_color.rgb = RGBColor(22, 27, 34)  # Lighter card bg
                shape.line.color.rgb = ACCENT_GREEN
                
                # Add text to the box
                tf_box = shape.text_frame
                tf_box.text = f"{chart_data.get('title', 'Data')}\n"
                
                title_p = tf_box.paragraphs[0]
                title_p.font.bold = True
                title_p.font.size = Pt(18)
                title_p.font.color.rgb = ACCENT_GREEN
                title_p.alignment = PP_ALIGN.CENTER
                
                # List values
                labels = chart_data.get('labels', [])
                values = chart_data.get('values', [])
                
                for idx, label in enumerate(labels):
                    p = tf_box.add_paragraph()
                    val = values[idx] if idx < len(values) else ''
                    p.text = f"{label}: {val}"
                    p.font.size = Pt(14)
                    p.font.color.rgb = TEXT_WHITE
                    p.alignment = PP_ALIGN.LEFT

    output = io.BytesIO()
    prs.save(output)
    output.seek(0)
    return output
