from __future__ import annotations
from flask import Blueprint, request, jsonify
import pandas as pd

from app.services.scoring import score_startups

bp = Blueprint("score", __name__)


@bp.post("/score")
def api_score():
    """
    POST /api/score
    JSON Body:
    {
      "startups": [ { ...fields... }, {...} ],
      "s_pool": 1000000,      # optional; default 1,000,000
      "gamma": 1.2            # optional; default 1.2
    }
    Returns QFS, Tokens, V_current_USD for each startup.
    Token allocation is proportional within THIS payload (the cohort you send).
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "startups" not in data or not isinstance(data["startups"], list):
            return jsonify({
                "error": "Invalid payload. Expect JSON with key 'startups' as a list of startup objects."
            }), 400

        startups = data["startups"]
        if len(startups) == 0:
            return jsonify({"error": "The 'startups' list is empty."}), 400

        # Optional knobs
        try:
            s_pool = int(data.get("s_pool", 1_000_000))
            gamma = float(data.get("gamma", 1.2))
        except (ValueError, TypeError) as e:
            return jsonify({"error": f"Invalid 's_pool' or 'gamma' parameter types: {str(e)}"}), 400

        # Build DataFrame and score
        df = pd.DataFrame(startups)
        try:
            scored = score_startups(df, s_pool=s_pool, gamma=gamma)
        except Exception as e:
            return jsonify({
                "error": "Scoring failed.",
                "detail": str(e)
            }), 500

        # Convert to plain JSON
        results = scored.to_dict(orient="records")

        return jsonify({
            "meta": {
                "cohort_size": len(results),
                "s_pool": s_pool,
                "gamma": gamma
            },
            "results": results
        }), 200
    except Exception as e:
        return jsonify({
            "error": "Unexpected error occurred.",
            "detail": str(e)
        }), 500

