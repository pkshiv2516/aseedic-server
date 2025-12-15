from __future__ import annotations
from flask import Blueprint, request, jsonify
from pathlib import Path
import pandas as pd

from app.services.location_reco import (
    prep, generate_candidates, build_ltr, train_regressor, recommend,
    DEFAULT_INPUT, DEFAULT_OUTDIR
)

bp = Blueprint("location_recommender", __name__)


@bp.post("/location/prep")
def api_prep():
    try:
        data = request.get_json(force=True, silent=False) or {}
        input_csv = data.get("input_csv", DEFAULT_INPUT)
        outdir = data.get("outdir", DEFAULT_OUTDIR)
        recent_years = int(data.get("recent_years", 5))
        topk_prep = int(data.get("topk_prep", 10))

        paths = prep(input_csv=input_csv, outdir=outdir, recent_years=recent_years)
        cand_p = generate_candidates(outdir=outdir, topk=topk_prep)
        ltr_p  = build_ltr(outdir=outdir)

        return jsonify({
            "status": "ok",
            "agg_features_csv": paths["agg_features_csv"],
            "candidates_csv": cand_p,
            "ltr_train_csv": ltr_p,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.post("/location/train")
def api_train():
    try:
        data = request.get_json(force=True, silent=False) or {}
        outdir = data.get("outdir", DEFAULT_OUTDIR)
        result = train_regressor(outdir=outdir)
        return jsonify({"status": "ok", **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.post("/location/recommend")
def api_recommend():
    try:
        data = request.get_json(force=True, silent=False) or {}
        outdir = data.get("outdir", DEFAULT_OUTDIR)
        level = data.get("level", "City")
        industry = data.get("industry", "FinTech")
        topk = int(data.get("topk", 10))
        use_model = bool(data.get("use_model", True))

        df = recommend(outdir=outdir, level=level, industry=industry, topk=topk, use_model=use_model)
        recs = df.to_dict(orient="records")
        return jsonify({
            "status": "ok",
            "recommendations": recs,
            "count": len(recs),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.get("/location/industries")
def api_industries():
    try:
        outdir = request.args.get("outdir", DEFAULT_OUTDIR)
        agg_path = Path(outdir) / "agg_features.csv"
        if not agg_path.exists():
            return jsonify({"industries": []})
        df = pd.read_csv(agg_path, usecols=["Industry"]).dropna()
        industries = sorted(df["Industry"].astype(str).unique().tolist())
        return jsonify({"industries": industries})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



