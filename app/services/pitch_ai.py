"""
Pitch Deck AI Content Generator Service

Uses Google Gemini to generate professional pitch deck content
for each slide section with structured JSON output.
"""

import os
import json
import google.generativeai as genai
from typing import Optional

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize model
model = genai.GenerativeModel('gemini-2.0-flash')

# Slide sections for a complete pitch deck
SLIDE_SECTIONS = [
    'cover',
    'problem',
    'solution',
    'market',
    'product',
    'business_model',
    'competition',
    'team',
    'traction',
    'funding_needs'
]


def generate_pitch_deck_section(
    section: str, 
    context: dict, 
    current_content: Optional[str] = None
) -> dict:
    """
    Generate content for a specific pitch deck section using Gemini AI.
    
    Args:
        section: The slide section name (e.g., 'cover', 'problem', 'market')
        context: Dictionary containing startup information
        current_content: Optional existing content for regeneration
        
    Returns:
        Dictionary with 'title', 'content', and optional 'chart_data'
    """
    startup_name = context.get('startupName', 'Startup')
    
    # Base instructions
    prompt = f"""
    Act as an expert startup consultant. Create content for the "{section}" section of a pitch deck.
    Startup Name: {startup_name}
    Context: {json.dumps(context)}
    
    Requirements:
    - Be creative, compelling, and professional.
    - Focus on key points.
    """

    # Section-specific logic
    if section == 'cover':
        prompt += """
        Create a catchy tagline.
        Return JSON structure: { "title": "Startup Name", "content": "Tagline", "chart_data": null }
        """
    elif section == 'market':
        prompt += """
        Include Market Size data (TAM/SAM/SOM).
        Return JSON structure:
        {
            "title": "Market Opportunity",
            "content": "Text description...",
            "chart_data": {
                "type": "pie",
                "title": "Market Size",
                "labels": ["TAM", "SAM", "SOM"],
                "values": [value1, value2, value3] (numbers representing market size in billions)
            }
        }
        """
    elif section == 'traction':
        prompt += """
        Include fictional but realistic growth metrics.
        Return JSON structure:
        {
            "title": "Traction & Growth",
            "content": "Text description...",
            "chart_data": {
                "type": "bar",
                "title": "User Growth",
                "labels": ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5"],
                "values": [100, 500, 2000, 5000, 10000]
            }
        }
        """
    elif section == 'business_model':
        prompt += """
        Include revenue streams.
        Return JSON structure:
        {
            "title": "Business Model",
            "content": "Text description...",
            "chart_data": {
                "type": "pie",
                "title": "Revenue Split",
                "labels": ["Stream A", "Stream B"],
                "values": [60, 40]
            }
        }
        """
    elif current_content:
        # For regeneration
        prompt += f"""
        Refine this content:
        {current_content}
        
        Return JSON structure: {{ "title": "Section Title", "content": "Improved content...", "chart_data": null }}
        """
    else:
        # Generic sections
        prompt += f"""
        Return JSON structure: {{ "title": "{section.replace('_', ' ').title()}", "content": "Bullet points...", "chart_data": null }}
        """

    prompt += "\nRETURN ONLY RAW JSON. No markdown formatting."

    try:
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON parse error for {section}: {e}")
        return {
            "title": section.replace('_', ' ').title(),
            "content": "Generation failed. Please try again.",
            "chart_data": None
        }
    except Exception as e:
        print(f"Error generating {section}: {e}")
        return {
            "title": section.replace('_', ' ').title(),
            "content": "Generation failed. Please try again.",
            "chart_data": None
        }


def generate_full_deck(context: dict) -> list:
    """
    Generate content for all 10 slides of a pitch deck.
    
    Args:
        context: Dictionary containing startup information
        
    Returns:
        List of slide dictionaries with 'title', 'content', 'chart_data'
    """
    deck_slides = []
    
    for section in SLIDE_SECTIONS:
        slide_data = generate_pitch_deck_section(section, context)
        deck_slides.append(slide_data)
    
    return deck_slides
