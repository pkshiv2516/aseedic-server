"""
Pitch Deck Generation API Routes

Endpoints for generating AI-powered pitch deck content and PowerPoint files.
"""

from flask import Blueprint, request, jsonify, send_file
from app.services.pitch_ai import generate_full_deck, generate_pitch_deck_section
from app.services.pitch_generator import generate_ppt_file

bp = Blueprint("pitch", __name__)


@bp.route("/pitch/generate-full-deck", methods=["POST"])
def api_generate_full_deck():
    """
    Generate content for all 10 slides of a pitch deck.
    
    Request Body:
        {
            "startupName": "QuantumFAI",
            "problem": "Description of the problem...",
            "solution": "Your solution...",
            "targetMarket": "Target market...",
            "businessModel": "Business model...",
            ...any additional context
        }
    
    Returns:
        {
            "slides": [
                {"title": "...", "content": "...", "chart_data": {...}},
                ...
            ]
        }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    if not data.get("startupName"):
        return jsonify({"error": "startupName is required"}), 400
    
    deck_slides = generate_full_deck(data)
    
    return jsonify({"slides": deck_slides})


@bp.route("/pitch/generate-slide", methods=["POST"])
def api_generate_slide():
    """
    Regenerate content for a specific slide.
    
    Request Body:
        {
            "sectionTitle": "Market",
            "currentContent": "Old content to improve...",
            "context": { ...startup data }
        }
    
    Returns:
        {"title": "...", "content": "...", "chart_data": {...}}
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    section = data.get('sectionTitle', '').lower().replace(' ', '_')
    context = data.get('context', {})
    current_content = data.get('currentContent')
    
    if not section:
        return jsonify({"error": "sectionTitle is required"}), 400
    
    slide_data = generate_pitch_deck_section(section, context, current_content)
    
    return jsonify(slide_data)


@bp.route("/pitch/generate-ppt", methods=["POST"])
def api_generate_ppt():
    """
    Generate and download a PowerPoint file from existing slide data.
    
    Request Body:
        {
            "startupName": "QuantumFAI",
            "deck": {
                "slides": [
                    {"title": "...", "content": "...", "chart_data": {...}},
                    ...
                ]
            }
        }
    
    Returns:
        Binary PPTX file download
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    deck = data.get('deck')
    if not deck or not deck.get('slides'):
        return jsonify({"error": "deck.slides is required"}), 400
    
    startup_name = data.get('startupName', 'Startup')
    
    ppt_file = generate_ppt_file(deck['slides'])
    
    return send_file(
        ppt_file,
        as_attachment=True,
        download_name=f"{startup_name.replace(' ', '_')}_Pitch_Deck.pptx",
        mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation'
    )


@bp.route("/pitch/generate-and-download", methods=["POST"])
def api_generate_and_download():
    """
    Generate pitch deck content and immediately return the PowerPoint file.
    Combines generate-full-deck and generate-ppt into one call.
    
    Request Body:
        {
            "startupName": "QuantumFAI",
            "problem": "...",
            "solution": "...",
            ...
        }
    
    Returns:
        Binary PPTX file download
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    if not data.get("startupName"):
        return jsonify({"error": "startupName is required"}), 400
    
    # Generate content for all sections
    deck_slides = generate_full_deck(data)
    
    # Generate PPT from the content
    ppt_file = generate_ppt_file(deck_slides)
    startup_name = data.get('startupName', 'Startup')

    return send_file(
        ppt_file,
        as_attachment=True,
        download_name=f"{startup_name.replace(' ', '_')}_Pitch_Deck.pptx",
        mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation'
    )
