from __future__ import annotations
"""
LLM Recommendation Service

Uses Google Gemini to generate actionable, context-aware recommendations
on top of the existing prediction model outputs — funding predictions,
QFS scores, investor matches, and location analysis.
"""

import os
import re
import json
import time
from typing import Optional

from google import genai


# Fallback model list — tried in order until one succeeds
_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-001",
]


# ----------------------------- Helpers ----------------------------- #

def _get_client():
    """Build Gemini client on demand so the key is always read fresh."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _call_gemini(prompt: str) -> dict:
    """
    Send a prompt to Gemini and parse the JSON response.
    Tries multiple models in order with retry on 503 — returns None on total failure.
    """
    client = _get_client()
    if not client:
        return {"error": "GEMINI_API_KEY is not configured."}

    env_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    models_to_try = [env_model] + [m for m in _FALLBACK_MODELS if m != env_model]

    last_error = None
    for model in models_to_try:
        # Retry up to 2 times on 503 (overloaded) with a short wait
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                text = response.text

                match = re.search(r'\{.*\}', text, re.DOTALL)
                if not match:
                    print(f"[llm_recommender] No JSON found in response from {model}: {text[:200]}")
                    break  # try next model

                print(f"[llm_recommender] Success with model: {model}")
                return json.loads(match.group())

            except json.JSONDecodeError as e:
                print(f"[llm_recommender] JSON parse error with {model}: {e}")
                last_error = str(e)
                break  # try next model
            except Exception as e:
                err_str = str(e)
                last_error = err_str
                if "503" in err_str and attempt == 0:
                    # Model overloaded — wait 3s and retry once
                    print(f"[llm_recommender] {model} overloaded, retrying in 3s...")
                    time.sleep(3)
                    continue
                print(f"[llm_recommender] {model} failed: {e}")
                break  # try next model

    print(f"[llm_recommender] All models exhausted. Last error: {last_error}")
    return None


def _format_usd(amount: float) -> str:
    """Format a raw USD float into a readable string like $4.2M or $850K."""
    if amount >= 1_000_000_000:
        return f"${amount / 1_000_000_000:.1f}B"
    if amount >= 1_000_000:
        return f"${amount / 1_000_000:.1f}M"
    if amount >= 1_000:
        return f"${amount / 1_000:.0f}K"
    return f"${amount:.0f}"


# ----------------------------- Core Functions ----------------------------- #

def recommend_from_funding_prediction(
    startup_profile: dict,
    predicted_funding_usd: float,
) -> dict:
    """
    Generate recommendations based on the funding prediction model output.

    Args:
        startup_profile: The original request payload (founded_date, industries, etc.)
        predicted_funding_usd: The raw prediction from the TF-DF model

    Returns:
        Structured recommendation dict with summary, strengths, gaps, and next steps.
    """
    formatted = _format_usd(predicted_funding_usd)
    industries = startup_profile.get("industries", [])
    industry_str = ", ".join(industries) if industries else "Not specified"

    prompt = f"""
You are a senior startup funding analyst. A startup has been evaluated by a machine learning model
that predicted their total fundable amount based on historical data from thousands of companies.

Startup Profile:
- Industries: {industry_str}
- Founded: {startup_profile.get("founded_date", "N/A")}
- Headquarters: {startup_profile.get("headquarters_location", "N/A")}
- Number of Founders: {startup_profile.get("number_of_founders", "N/A")}
- Number of Investors: {startup_profile.get("number_of_investors", "N/A")}
- Funding Rounds So Far: {startup_profile.get("number_of_funding_rounds", "N/A")}
- Employee Range: {startup_profile.get("employees_label", "N/A")}
- Revenue Range: {startup_profile.get("revenue_label", "N/A")}
- Last Funding Type: {startup_profile.get("last_funding_type", "N/A")}
- Patents Granted: {startup_profile.get("patents_granted", 0)}

ML Model Prediction:
- Predicted Total Funding Potential: {formatted}

Based on this profile and prediction, provide a concise, actionable analysis.
Be direct and specific — avoid generic advice.

Return ONLY this JSON structure, no markdown:
{{
  "summary": "2-3 sentence overview of the startup's funding position",
  "predicted_funding_readable": "{formatted}",
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "gaps": ["gap or risk 1", "gap or risk 2"],
  "recommendations": ["specific action 1", "specific action 2", "specific action 3"],
  "next_steps": ["immediate step 1", "immediate step 2"]
}}
"""
    return _call_gemini(prompt)


def recommend_from_qfs_score(
    startup_data: dict,
    qfs: float,
    tokens: int,
    valuation_usd: float,
) -> dict:
    """
    Generate recommendations based on the QuantumFAI Score output.

    Args:
        startup_data: Single startup row dict (same fields used in /api/score)
        qfs: The computed QFS score (0-100)
        tokens: Allocated tokens from the scoring engine
        valuation_usd: Estimated current valuation in USD

    Returns:
        Structured recommendation dict tailored to the QFS result.
    """
    valuation_str = _format_usd(valuation_usd)

    # Determine score band for context
    if qfs >= 75:
        band = "strong"
    elif qfs >= 50:
        band = "moderate"
    elif qfs >= 30:
        band = "developing"
    else:
        band = "early-stage with significant gaps"

    prompt = f"""
You are a startup investment analyst working with the QuantumFAI scoring system.
A startup has been scored across multiple dimensions including revenue, team size,
funding history, IP assets, web traction, and sector/region factors.

Startup Details:
- Industry: {startup_data.get("Industry", "N/A")}
- Headquarters: {startup_data.get("Headquarters Location", "N/A")}
- Founded: {startup_data.get("Founded Date", "N/A")}
- Company Type: {startup_data.get("Company Type", "N/A")}
- Employees: {startup_data.get("Number of Employees", "N/A")}
- Founders: {startup_data.get("Number of Founders", "N/A")}
- Revenue Range: {startup_data.get("Annual Revenue Range", "N/A")}
- Funding Status: {startup_data.get("Funding Status", "N/A")}
- Funding Rounds: {startup_data.get("Number of Funding Rounds", "N/A")}
- Monthly Website Visits: {startup_data.get("Monthly Website Visits", "N/A")}
- Currently Hiring: {startup_data.get("Currently Hiring?", "N/A")}
- Patents Granted: {startup_data.get("Patents Granted", 0)}

QuantumFAI Score Results:
- QFS Score: {round(qfs, 2)} / 100 ({band})
- Token Allocation: {tokens:,}
- Estimated Valuation: {valuation_str}

Provide a focused, honest analysis. Call out what's actually holding the score back
and what would move the needle most. Be specific to this startup's profile.

Return ONLY this JSON structure, no markdown:
{{
  "summary": "2-3 sentence assessment of the QFS result and what it means",
  "score_band": "{band}",
  "key_drivers": ["what pushed the score up 1", "what pushed the score up 2"],
  "score_drags": ["what pulled the score down 1", "what pulled the score down 2"],
  "recommendations": ["specific improvement 1", "specific improvement 2", "specific improvement 3"],
  "valuation_note": "brief note on the estimated valuation and what affects it",
  "next_steps": ["immediate action 1", "immediate action 2"]
}}
"""
    return _call_gemini(prompt)


def recommend_from_investor_matches(
    startup_profile: dict,
    top_investors: list,
) -> dict:
    """
    Generate outreach and positioning recommendations based on investor match results.

    Args:
        startup_profile: The startup's industry, stage, and region info
        top_investors: List of top matched investor dicts (from /api/match)

    Returns:
        Structured recommendation dict with outreach strategy and positioning tips.
    """
    # Summarize top 5 investors for the prompt — avoid bloating context
    top5 = top_investors[:5]
    investor_summary = []
    for inv in top5:
        name = inv.get("investor_name", "Unknown")
        score = inv.get("final_score", 0)
        focus = inv.get("focus", "N/A")
        stage = inv.get("stage", "N/A")
        geo = inv.get("geo_focus", "N/A")
        investor_summary.append(
            f"- {name} (match score: {score}) | Focus: {focus} | Stage: {stage} | Geo: {geo}"
        )

    investors_text = "\n".join(investor_summary) if investor_summary else "No matches found."

    prompt = f"""
You are a startup fundraising strategist. A startup has run an investor matching algorithm
that uses TF-IDF similarity across industry, funding stage, location, and portfolio data.

Startup Profile:
- Industry: {startup_profile.get("industry", "N/A")}
- Funding Stage: {startup_profile.get("funding_stage", "N/A")}
- Region: {startup_profile.get("region", "N/A")}

Top Matched Investors:
{investors_text}

Total matches found: {len(top_investors)}

Based on these matches, provide strategic advice on how to approach these investors,
what to emphasize in outreach, and how to position the startup effectively.
Be specific — reference the investor focus areas and stage preferences.

Return ONLY this JSON structure, no markdown:
{{
  "summary": "2-3 sentence overview of the match quality and investor landscape",
  "outreach_strategy": ["strategy point 1", "strategy point 2", "strategy point 3"],
  "positioning_tips": ["tip 1", "tip 2"],
  "top_investor_to_prioritize": "name and one-line reason why",
  "red_flags_to_address": ["potential concern 1", "potential concern 2"],
  "next_steps": ["immediate action 1", "immediate action 2"]
}}
"""
    return _call_gemini(prompt)


def recommend_from_location_analysis(
    industry: str,
    geographic_level: str,
    top_locations: list,
) -> dict:
    """
    Generate location strategy recommendations based on the location recommender output.

    Args:
        industry: The startup's industry vertical
        geographic_level: "City", "State", or "Country"
        top_locations: List of top location dicts from /api/location/recommend

    Returns:
        Structured recommendation dict with location strategy and expansion advice.
    """
    top5 = top_locations[:5]
    location_summary = []
    for loc in top5:
        name = loc.get("Location", "Unknown")
        score = round(loc.get("score", loc.get("Composite Score", 0)), 3)
        success = round(loc.get("SuccessRate", 0), 3)
        density = round(loc.get("DensityLog", 0), 3)
        confidence = loc.get("Confidence", "N/A")
        location_summary.append(
            f"- {name} | Composite Score: {score} | Success Rate: {success} | "
            f"Ecosystem Density: {density} | Confidence: {confidence}"
        )

    locations_text = "\n".join(location_summary) if location_summary else "No locations found."

    prompt = f"""
You are a startup ecosystem strategist. A startup in the {industry} space has run a
location recommendation analysis that scores geographic areas based on startup density,
funding availability, success rates, and recent growth trends.

Analysis Parameters:
- Industry: {industry}
- Geographic Level Analyzed: {geographic_level}

Top Recommended Locations:
{locations_text}

Based on this data, provide strategic advice on where to establish or expand operations,
what makes the top location compelling, and what trade-offs to consider.
Ground your advice in the actual scores — don't be generic.

Return ONLY this JSON structure, no markdown:
{{
  "summary": "2-3 sentence overview of the location landscape for this industry",
  "top_location_rationale": "why the #1 location stands out based on the data",
  "expansion_strategy": ["strategic point 1", "strategic point 2", "strategic point 3"],
  "trade_offs": ["trade-off to consider 1", "trade-off to consider 2"],
  "recommendations": ["specific recommendation 1", "specific recommendation 2"],
  "next_steps": ["immediate action 1", "immediate action 2"]
}}
"""
    return _call_gemini(prompt)
