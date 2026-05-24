from __future__ import annotations
"""
Market Analysis Agent Route

Exposes the 3-step Market Analysis Agent as a single REST endpoint.
The agent analyzes the market landscape, entry barriers, customer profile,
demand signals, and delivers a go-to-market strategy.
"""

from flask import Blueprint, request, jsonify
from app.services.market_agent import run_market_analysis_agent

bp = Blueprint("market", __name__)


@bp.post("/market/analyze")
def api_market_analyze():
    """
    POST /api/market/analyze

    Runs the full 3-step Market Analysis Agent for a startup.

    The agent executes these steps sequentially:
      1. Market Landscape    — TAM/SAM/SOM, CAGR, segments, trends
      2. Intelligence Bundle — Entry barriers, customer profile, demand signals
      3. Market Strategy     — GTM approach, entry plan, expansion roadmap, verdict

    Request Body:
    {
      "company_name": "MyStartup",
      "industry": "FinTech",
      "niche": "B2B payments infrastructure",
      "target_market": "SMEs in Southeast Asia",
      "region": "Southeast Asia",
      "stage": "Seed",
      "product_description": "API-first payment gateway for SMEs"
    }

    Returns:
    {
      "startup": "MyStartup",
      "industry": "FinTech",
      "cache_hit": false,
      "agent_steps_completed": 3,
      "tam": {...},
      "sam": {...},
      "som": {...},
      "cagr": "...",
      "market_maturity": "...",
      "key_segments": [...],
      "top_trends": [...],
      "market_summary": "...",
      "entry_barriers": [...],
      "customer_profile": {...},
      "demand_signals": {...},
      "gtm_strategy": {...},
      "market_entry_plan": [...],
      "expansion_roadmap": [...],
      "key_risks": [...],
      "success_metrics": [...],
      "verdict": "..."
    }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request body is required"}), 400

        if not data.get("industry"):
            return jsonify({"error": "industry is required"}), 400

        if not data.get("product_description"):
            return jsonify({"error": "product_description is required"}), 400

        result = run_market_analysis_agent(data)

        steps = result.get("agent_steps_completed", 0)
        if steps == 0:
            return jsonify({
                "error": "agent_failed",
                "message": result.get("error", "Agent could not complete analysis."),
                "partial_result": result,
            }), 503

        response = result.copy()
        if steps < 3:
            response.setdefault("warning", f"Agent completed {steps}/3 steps. Partial results returned.")

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": "server_error", "message": str(e)}), 500
