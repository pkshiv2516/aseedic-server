from __future__ import annotations
from flask import Blueprint, request, jsonify

from app.services.investor_matcher import load_investor_data, match_investors

bp = Blueprint("investor_match", __name__)


@bp.post("/match")
def api_investors_match():
    """
    POST /api/match
    Body example:
    {
      "industry": "FinTech",
      "funding_stage": "Seed",
      "region": "India",
      "traction_score": 78,              # optional, not used in baseline model
      "topk": 100,                        # optional
      "csv_path": "investor_db.csv"      # optional custom CSV path
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        csv_path = data.get("csv_path", "investor_db.csv")
        try:
            topk = int(data.get("topk", 100))
        except Exception:
            topk = 100

        df = load_investor_data(csv_path)
        if df is None:
            return jsonify({
                "error": "investor_db_missing",
                "message": f"Investor database not found at '{csv_path}'."
            }), 500

        results_df = match_investors(
            startup_profile=data,
            df=df,
            topk=topk,
        )
        return jsonify(results_df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": "server_error", "message": str(e)}), 500


