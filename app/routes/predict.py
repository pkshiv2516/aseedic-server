from __future__ import annotations
from flask import Blueprint, request, jsonify
from pydantic import ValidationError

from app.schemas import FundingRequest, FundingResponse
from app.services.preprocess import predict
import warnings, pandas as pd
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
bp = Blueprint("predict", __name__)


@bp.post("/funding/predict")
def funding_predict():
    try:
        data = request.get_json(force=True, silent=False)
        clean = data.copy()
        model=clean.pop("model_type", None) # safely remove if present
        req = FundingRequest(**clean)
        # req = FundingRequest(**data)
    except ValidationError as ve:
        return jsonify({"error": "validation_error", "details": ve.errors()}), 422
    except Exception as e:
        return jsonify({"error": "bad_request", "message": str(e)}), 400

    value, meta = predict(req.model_dump(),model)

    resp = FundingResponse(
        predicted_total_funding_usd=value,
        log_transform=meta["LOG_TRANSFORM"],
        safe_target=meta["safe_target"],
        safe_target_log=meta["safe_target_log"],
        n_features_fed=meta["n_features_fed"],
    )
    return jsonify(resp.model_dump())
