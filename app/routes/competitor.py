from __future__ import annotations
"""
Competitor Analysis Agent Route

Exposes the 5-step Competitor Analysis Agent as a single REST endpoint.
The agent reasons through the competitive landscape in structured phases,
building context from each step into the next.
"""

from flask import Blueprint, request, jsonify
from app.services.competitor_agent import run_competitor_analysis_agent

bp = Blueprint("competitor", __name__)


@bp.post("/competitor/analyze")
def api_competitor_analyze():
    """
    POST /api/competitor/analyze

    Runs the full 3-step Competitor Analysis Agent for a startup.

    The agent executes these steps sequentially:
      1. Market Scan         — identifies top 5 real competitors
      2. Intelligence Bundle — deep dive + positioning + gap analysis
      3. Battle Plan         — delivers a concrete competitive strategy

    Request Body:
    {
      "company_name": "MyStartup",
      "industry": "FinTech",
      "niche": "B2B payments infrastructure",
      "target_market": "SMEs in Southeast Asia",
      "region": "Southeast Asia",
      "stage": "Seed",
      "product_description": "API-first payment gateway for SMEs",
      "usp": "Lowest transaction fees with instant settlement"
    }

    Returns:
    {
      "startup": "MyStartup",
      "industry": "FinTech",
      "agent_steps_completed": 5,
      "market_summary": "...",
      "competitors_identified": [...],
      "competitor_analysis": [...],
      "positioning": {...},
      "gap_intelligence": {...},
      "battle_plan": {...}
    }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request body is required"}), 400

        # Validate required fields
        if not data.get("industry"):
            return jsonify({"error": "industry is required"}), 400

        if not data.get("product_description"):
            return jsonify({"error": "product_description is required"}), 400

        result = run_competitor_analysis_agent(data)

        steps = result.get("agent_steps_completed", 0)
        if steps == 0:
            return jsonify({
                "error": "agent_failed",
                "message": result.get("error", "Agent could not complete analysis."),
                "partial_result": result,
            }), 503

        # Return 200 even on partial — prediction data is still valuable
        response = result.copy()
        if steps < 3:
            response.setdefault("warning", f"Agent completed {steps}/3 steps. Partial results returned.")

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": "server_error", "message": str(e)}), 500
