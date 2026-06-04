from __future__ import annotations
"""
Geographic Distribution Route

Returns the geographic distribution of startups in the same industry
as the given startup profile — designed for dashboard bottom-panel display.

Provides:
  - Country-level distribution (% share + counts) → pie/bar chart
  - City hotspots (top cities by startup count) → map/bubble chart
  - Regional breakdown (continental/macro-region %) → donut chart
  - LLM-powered strategic geo insights
"""

from flask import Blueprint, request, jsonify
from app.services.geo_distribution_agent import run_geo_distribution_agent

bp = Blueprint("geo", __name__)


@bp.post("/geo/distribution")
def api_geo_distribution():
    """
    POST /api/geo/distribution

    Analyzes geographic distribution of startups in the same industry
    as the given startup profile. Uses real dataset + Gemini insights.

    Request Body:
    {
      "industry": "FinTech",
      "headquarters_location": "Bangalore, Karnataka, India",
      "region": "South Asia",
      "stage": "Seed",
      "company_name": "PaySwift"         (optional)
    }

    Minimum required: "industry"

    Returns:
    {
      "industry": "FinTech",
      "total_startups_in_industry": 1234,
      "top_country": "United States",
      "top_city": "San Francisco",

      "country_distribution": [
        { "country": "United States", "count": 450, "percentage": 36.5 },
        { "country": "India", "count": 210, "percentage": 17.0 },
        ...
      ],

      "city_hotspots": [
        { "city": "San Francisco", "count": 120, "percentage": 9.7 },
        { "city": "New York", "count": 95, "percentage": 7.7 },
        ...
      ],

      "regional_breakdown": [
        { "region": "North America", "count": 480, "percentage": 42.1 },
        { "region": "Asia-Pacific (APAC)", "count": 310, "percentage": 27.2 },
        ...
      ],

      "geo_insights": {
        "distribution_summary": "...",
        "dominant_hub_analysis": "...",
        "startup_geo_position": "...",
        "geographic_advantages": [...],
        "geographic_challenges": [...],
        "strategic_recommendations": [...],
        "expansion_targets": [...],
        "verdict": "..."
      },

      "cache_hit": false,
      "agent_steps_completed": 2
    }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request body is required"}), 400

        if not data.get("industry"):
            return jsonify({"error": "industry is required"}), 400

        # Optional CSV path override (for testing)
        csv_path = data.pop("csv_path", None)

        result = run_geo_distribution_agent(
            startup_profile=data,
            csv_path=csv_path,
        )

        steps = result.get("agent_steps_completed", 0)

        if steps == 0:
            return jsonify({
                "error": "agent_failed",
                "message": result.get("error", "No distribution data found for this industry."),
                "industry": data.get("industry"),
            }), 404

        response = result.copy()
        if steps < 2:
            response.setdefault(
                "warning",
                "Distribution data computed but LLM insights unavailable."
            )

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": "server_error", "message": str(e)}), 500
