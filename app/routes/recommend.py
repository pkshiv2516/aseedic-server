from __future__ import annotations
"""
LLM Recommendation Routes

Wraps the four prediction model outputs with Gemini-powered recommendations.
Each endpoint accepts the same input as its corresponding prediction endpoint,
runs the model, then passes the output through the LLM for actionable insights.
"""

from flask import Blueprint, request, jsonify
from pydantic import ValidationError
import pandas as pd
import time
import warnings

warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

from app.schemas import FundingRequest
from app.services.preprocess import predict as run_funding_predict
from app.services.scoring import score_startups
from app.services.investor_matcher import load_investor_data, match_investors
from app.services.location_reco import recommend as run_location_recommend, DEFAULT_OUTDIR
from app.services.llm_recommender import (
    recommend_from_funding_prediction,
    recommend_from_qfs_score,
    recommend_from_investor_matches,
    recommend_from_location_analysis,
)


bp = Blueprint("recommend", __name__)


@bp.post("/recommend/funding")
def api_recommend_funding():
    """
    POST /api/recommend/funding

    Runs the funding prediction model and returns both the raw prediction
    and an LLM-generated recommendation on top of it.

    Request Body: same as POST /api/funding/predict
    {
      "founded_date": "2021-03-01",
      "number_of_founders": 2,
      "number_of_investors": 4,
      "number_of_funding_rounds": 1,
      "patents_granted": 0,
      "employees_label": "11 – 50",
      "revenue_label": "$1M – $10M",
      "headquarters_location": "Bangalore, Karnataka, India",
      "last_funding_type": "Seed",
      "industries": ["AI", "SaaS"],
      "model_type": "combined"   # optional
    }

    Returns:
    {
      "prediction": { ...raw model output... },
      "recommendation": { ...LLM analysis... }
    }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Request body is required"}), 400

        clean = data.copy()
        model_type = clean.pop("model_type", None)

        try:
            req = FundingRequest(**clean)
        except ValidationError as ve:
            return jsonify({"error": "validation_error", "details": ve.errors()}), 422

        predicted_usd, meta = run_funding_predict(req.model_dump(), model_type)

        prediction = {
            "predicted_total_funding_usd": predicted_usd,
            "log_transform": meta["LOG_TRANSFORM"],
            "safe_target": meta["safe_target"],
            "n_features_fed": meta["n_features_fed"],
        }

        recommendation = recommend_from_funding_prediction(
            startup_profile=req.model_dump(),
            predicted_funding_usd=predicted_usd,
        )

        response = {"prediction": prediction, "recommendation": recommendation}
        if recommendation is None:
            response["recommendation"] = None
            response["warning"] = "LLM recommendation unavailable. Prediction succeeded."

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": "server_error", "message": str(e)}), 500


@bp.post("/recommend/score")
def api_recommend_score():
    """
    POST /api/recommend/score

    Scores a single startup using the QFS engine and returns both the
    raw scores and an LLM-generated recommendation based on the result.

    Note: scoring works on a cohort (batch), but recommendation is per-startup.
    Send one startup at a time for a focused recommendation.

    Request Body: same shape as a single item in POST /api/score startups list
    {
      "Industry": "FinTech",
      "Headquarters Location": "Mumbai, Maharashtra, India",
      "Founded Date": "2020-06-15",
      "Company Type": "B2B",
      "Number of Employees": 45,
      "Number of Founders": 3,
      "Annual Revenue Range": "$1M – $10M",
      "Funding Status": "Seed",
      "Number of Funding Rounds": 2,
      "Monthly Website Visits": 12000,
      "Currently Hiring?": "Yes",
      "Patents Granted": 1,
      "Trademarks Registered": 0
    }

    Returns:
    {
      "scores": { "QFS": ..., "Tokens": ..., "V_current_USD": ... },
      "recommendation": { ...LLM analysis... }
    }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request body must be a single startup object"}), 400

        df = pd.DataFrame([data])

        try:
            scored = score_startups(df, s_pool=1_000_000, gamma=1.2)
        except Exception as e:
            return jsonify({"error": "Scoring failed.", "detail": str(e)}), 500

        row = scored.iloc[0]
        qfs = float(row.get("QFS", 0))
        tokens = int(row.get("Tokens", 0))
        valuation = float(row.get("V_current_USD", 0))

        scores = {
            "QFS": round(qfs, 2),
            "Tokens": tokens,
            "V_current_USD": int(valuation),
        }

        # Small delay to avoid hitting Gemini rate limit back-to-back
        time.sleep(2)

        recommendation = recommend_from_qfs_score(
            startup_data=data,
            qfs=qfs,
            tokens=tokens,
            valuation_usd=valuation,
        )

        response = {"scores": scores, "recommendation": recommendation}
        if recommendation is None:
            response["recommendation"] = None
            response["warning"] = "LLM recommendation unavailable. Scoring succeeded."

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": "server_error", "message": str(e)}), 500


@bp.post("/recommend/investors")
def api_recommend_investors():
    """
    POST /api/recommend/investors

    Runs investor matching and returns both the matched investor list
    and an LLM-generated outreach and positioning strategy.

    Request Body: same as POST /api/match
    {
      "industry": "FinTech",
      "funding_stage": "Seed",
      "region": "India",
      "topk": 10,
      "csv_path": "investor_db.csv"   # optional
    }

    Returns:
    {
      "matches": [ ...investor list... ],
      "recommendation": { ...LLM outreach strategy... }
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}

        csv_path = data.get("csv_path", "investor_db.csv")
        try:
            topk = int(data.get("topk", 10))
        except Exception:
            topk = 10

        df = load_investor_data(csv_path)
        if df is None:
            return jsonify({
                "error": "investor_db_missing",
                "message": f"Investor database not found at '{csv_path}'.",
            }), 500

        results_df = match_investors(startup_profile=data, df=df, topk=topk)
        matches = results_df.to_dict(orient="records")

        recommendation = recommend_from_investor_matches(
            startup_profile=data,
            top_investors=matches,
        )

        response = {"matches": matches, "recommendation": recommendation}
        if recommendation is None:
            response["recommendation"] = None
            response["warning"] = "LLM recommendation unavailable. Investor matching succeeded."

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": "server_error", "message": str(e)}), 500


@bp.post("/recommend/location")
def api_recommend_location():
    """
    POST /api/recommend/location

    Runs the location recommender and returns both the ranked locations
    and an LLM-generated location strategy for the startup.

    Requires /api/location/prep to have been run at least once beforehand
    so that the aggregated features CSV exists.

    Request Body: same as POST /api/location/recommend
    {
      "industry": "FinTech",
      "level": "City",
      "topk": 5,
      "outdir": "reco_artifacts",   # optional
      "use_model": true             # optional
    }

    Returns:
    {
      "locations": [ ...ranked location list... ],
      "recommendation": { ...LLM strategy... }
    }
    """
    try:
        data = request.get_json(force=True, silent=True) or {}

        outdir = data.get("outdir", DEFAULT_OUTDIR)
        level = data.get("level", "City")
        industry = data.get("industry", "")
        topk = int(data.get("topk", 5))
        use_model = bool(data.get("use_model", True))

        if not industry:
            return jsonify({"error": "industry is required"}), 400

        try:
            loc_df = run_location_recommend(
                outdir=outdir,
                level=level,
                industry=industry,
                topk=topk,
                use_model=use_model,
            )
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 404
        except Exception as e:
            return jsonify({"error": "Location recommendation failed.", "detail": str(e)}), 500

        locations = loc_df.to_dict(orient="records")

        recommendation = recommend_from_location_analysis(
            industry=industry,
            geographic_level=level,
            top_locations=locations,
        )

        response = {"locations": locations, "recommendation": recommendation}
        if recommendation is None:
            response["recommendation"] = None
            response["warning"] = "LLM recommendation unavailable. Location ranking succeeded."

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": "server_error", "message": str(e)}), 500
