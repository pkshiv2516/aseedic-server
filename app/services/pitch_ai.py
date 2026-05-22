"""
Pitch Deck AI Content Generator Service

Uses Google Gemini to generate professional pitch deck content
for each slide section with structured JSON output.
"""

import os
import json
import re
import requests
from typing import Optional

# Configure Gemini

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3"

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


def _clean_json(text: str) -> str:
    """
    Strip markdown fences and extract the first valid JSON object or array.
    Handles cases where Gemini wraps output in ```json ... ``` blocks.
    """
    # Remove markdown code fences
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    # Extract first {...} block in case of extra commentary
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text


def _fallback(section: str) -> dict:
    """Return a safe fallback slide dict when generation fails."""
    return {
        "title": section.replace('_', ' ').title(),
        "content": "Content generation failed. Please regenerate this slide.",
        "chart_data": None
    }


def _build_prompt(section: str, context: dict, current_content: Optional[str]) -> str:
    startup_name = context.get('startupName', 'Startup')
    # Trim context to avoid token bloat — only pass relevant keys
    safe_context = {k: v for k, v in context.items()
                    if k in ('startupName', 'problem', 'solution', 'targetMarket',
                             'businessModel', 'teamInfo', 'traction', 'fundingNeeds',
                             'industry', 'location', 'stage')}

    base = f"""You are an expert startup pitch consultant.
Generate content for the "{section}" slide of a professional investor pitch deck.
Startup Name: {startup_name}
Startup Context: {json.dumps(safe_context)}

Rules:
- Be specific, compelling, and data-driven.
- Use bullet points for content (one point per line, no dashes or asterisks).
- Return ONLY a raw JSON object. No markdown. No explanation. No extra text.
"""

    if section == 'cover':
        return base + """
Return exactly:
{"title": "<Startup Name>", "content": "<One powerful tagline under 12 words>", "chart_data": null}
"""

    if section == 'problem':
        return base + """
Return exactly:
{"title": "The Problem", "content": "<3-4 bullet points describing the problem, one per line>", "chart_data": null}
"""

    if section == 'solution':
        return base + """
Return exactly:
{"title": "Our Solution", "content": "<3-4 bullet points describing the solution, one per line>", "chart_data": null}
"""

    if section == 'market':
        return base + """
Include realistic TAM/SAM/SOM figures in billions USD.
Return exactly:
{
  "title": "Market Opportunity",
  "content": "<2-3 sentences about the market opportunity>",
  "chart_data": {
    "type": "pie",
    "title": "Market Size (USD Billions)",
    "labels": ["TAM", "SAM", "SOM"],
    "values": [<tam_number>, <sam_number>, <som_number>]
  }
}
"""

    if section == 'product':
        return base + """
Return exactly:
{"title": "Product", "content": "<4-5 bullet points on key product features, one per line>", "chart_data": null}
"""

    if section == 'business_model':
        return base + """
Include 2-3 realistic revenue streams with percentage split.
Return exactly:
{
  "title": "Business Model",
  "content": "<3-4 bullet points on how the company makes money, one per line>",
  "chart_data": {
    "type": "pie",
    "title": "Revenue Streams",
    "labels": ["<Stream 1>", "<Stream 2>", "<Stream 3>"],
    "values": [<pct1>, <pct2>, <pct3>]
  }
}
"""

    if section == 'competition':
        return base + """
Return exactly:
{"title": "Competitive Landscape", "content": "<3-4 bullet points on competitive advantages and key competitors, one per line>", "chart_data": null}
"""

    if section == 'team':
        return base + """
Return exactly:
{"title": "Our Team", "content": "<3-4 bullet points on key team members and their relevant experience, one per line>", "chart_data": null}
"""

    if section == 'traction':
        return base + """
Include realistic monthly growth metrics for 6 months.
Return exactly:
{
  "title": "Traction & Growth",
  "content": "<2-3 sentences summarising key milestones achieved>",
  "chart_data": {
    "type": "bar",
    "title": "Monthly User Growth",
    "labels": ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"],
    "values": [<v1>, <v2>, <v3>, <v4>, <v5>, <v6>]
  }
}
"""

    if section == 'funding_needs':
        return base + """
Return exactly:
{"title": "Funding Ask", "content": "<3-4 bullet points on funding amount, use of funds breakdown, and expected milestones, one per line>", "chart_data": null}
"""

    # Fallback for regeneration or unknown sections
    if current_content:
        return base + f"""
Refine and improve this existing content:
{current_content}

Return exactly:
{{"title": "<Section Title>", "content": "<Improved bullet points, one per line>", "chart_data": null}}
"""

    return base + f"""
Return exactly:
{{"title": "{section.replace('_', ' ').title()}", "content": "<3-4 relevant bullet points, one per line>", "chart_data": null}}
"""


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
    prompt = _build_prompt(section, context, current_content)

    # Try up to 2 times in case Gemini returns malformed JSON first attempt
    for attempt in range(2):
        try:
            response = requests.post(OLLAMA_URL, json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False
            })
            cleaned = _clean_json(response.json()["response"])
            result = json.loads(cleaned)

            # Validate required keys are present
            if 'title' not in result or 'content' not in result:
                raise ValueError("Missing required keys in Gemini response")

            # Ensure chart_data key always exists
            result.setdefault('chart_data', None)
            return result

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[pitch_ai] Attempt {attempt + 1} failed for '{section}': {e}")
            if attempt == 0:
                # On first failure, append stricter instruction and retry
                prompt += "\n\nCRITICAL: Return ONLY the JSON object. Absolutely no other text."
                continue

        except Exception as e:
            print(f"[pitch_ai] Gemini error for '{section}': {e}")
            break

    return _fallback(section)


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
        try:
            slide_data = generate_pitch_deck_section(section, context)
        except Exception as e:
            # Isolate failures — one bad slide must not kill the whole deck
            print(f"[pitch_ai] Fatal error on section '{section}': {e}")
            slide_data = _fallback(section)

        deck_slides.append(slide_data)

    return deck_slides