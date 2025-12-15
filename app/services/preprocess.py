from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow_decision_forests as tfdf

from app.config import Config
from app.services.model_loader import load_artifacts, load_model, tf_to_np_dtype


# Buckets must mirror the Streamlit app exactly
EMP_BUCKETS: List[Tuple[str, int, int | None]] = [
    ("1 – 10", 1, 10),
    ("11 – 50", 11, 50),
    ("51 – 200", 51, 200),
    ("201 – 500", 201, 500),
    ("501 – 1,000", 501, 1000),
    ("1,001 – 5,000", 1001, 5000),
    ("5,001+", 5001, None),
]

REV_BUCKETS: List[Tuple[str, int, int | None]] = [
    ("0–250k", 0, 250_000),
    ("250k–500k", 250_000, 500_000),
    ("500k–750k", 500_000, 750_000),
    ("750k–1M", 750_000, 1_000_000),
    ("< $1M", 0, 1_000_000),
    ("$1M – $10M", 1_000_000, 10_000_000),
    ("$10M – $50M", 10_000_000, 50_000_000),
    ("$50M – $100M", 50_000_000, 100_000_000),
    ("$100M – $500M", 100_000_000, 500_000_000),
    ("$500M – $1B", 500_000_000, 1_000_000_000),
    ("$1B+", 1_000_000_000, 100_000_000_000),
]


def _bucket_lookup(label: str, buckets: List[Tuple[str, int, int | None]]):
    for lab, mn, mx in buckets:
        if lab == label:
            return mn, mx
    raise ValueError(f"Unknown bucket label: {label}")


def split_industries_to_lists(s: pd.Series) -> pd.Series:
    def _split(x):
        if pd.isna(x):
            return []
        parts = [p.strip() for p in str(x).replace("|", ",").split(",")]
        return [p for p in parts if p]
    return s.apply(_split)


def encode_with_known_classes(series: pd.Series, classes: list, unknown_bucket="__UNKNOWN__"):
    idx_map = {v: i for i, v in enumerate(classes)}
    if unknown_bucket not in idx_map:
        classes = classes + [unknown_bucket]
        idx_map[unknown_bucket] = len(classes) - 1
    return series.map(lambda v: idx_map.get(v, idx_map[unknown_bucket]))


def build_features(payload: dict,model_type) -> tuple[pd.DataFrame, dict]:
    """Recreate training-time preprocessing and align to serving signature.
    Returns the final feature DataFrame X and a meta dict.
    """
    mapping, feature_order, meta, industries_classes, cat_maps = load_artifacts(model_type)

    LOG_TRANSFORM = bool(meta.get("log_transform", True))
    SAFE_TARGET = meta.get("safe_target")
    SAFE_TARGET_LOG = meta.get("safe_target_log", (SAFE_TARGET or "target") + "_log")

    founded_date = pd.to_datetime(payload["founded_date"], errors="coerce")

    # Buckets
    emp_min, emp_max = _bucket_lookup(payload["employees_label"], EMP_BUCKETS)
    rev_min, rev_max = _bucket_lookup(payload["revenue_label"], REV_BUCKETS)

    if emp_max is None:
        emp_max = emp_min * 2
    if rev_max is None:
        rev_max = rev_min * 2

    raw_dict = {
        Config.FOUNDED_COL: founded_date,
        Config.IND_COL: ", ".join(payload.get("industries", [])),
        "Headquarters Location": payload["headquarters_location"],
        "Last Funding Type": payload["last_funding_type"],
        "Number of Founders": float(payload["number_of_founders"]),
        "Number of Investors": float(payload["number_of_investors"]),
        "Number of Funding Rounds": float(payload["number_of_funding_rounds"]),
        "IPqwery - Patents Granted": float(payload["patents_granted"]),
        "revenue_min_usd": float(rev_min),
        "revenue_max_usd": float(rev_max),
        "employees_min": int(emp_min),
        "employees_max": int(emp_max),
    }

    raw = pd.DataFrame([raw_dict])

    # Date features
    dt = pd.to_datetime(raw[Config.FOUNDED_COL], errors="coerce")
    today = pd.Timestamp.today().normalize()
    raw["company_age_years"] = (today - dt).dt.days / 365.25
    raw["Founded Date_Timestamp"] = dt.astype("int64", copy=False) // 10**9

    # Industries one-hot
    lists = split_industries_to_lists(raw[Config.IND_COL])
    for lab in industries_classes:
        raw[f"IND__{lab}"] = lists.apply(lambda arr: 1 if lab in arr else 0)

    # Encode categoricals
    for orig_col in ["Headquarters Location", "Last Funding Type"]:
        classes = cat_maps.get(orig_col)
        if not classes:
            raise RuntimeError(f"Missing classes artifact for {orig_col}.")
        raw[orig_col] = encode_with_known_classes(raw[orig_col].astype(str), classes)

    # Drop exploded cols
    raw = raw.drop(columns=[Config.IND_COL, Config.FOUNDED_COL])

    # Safe rename mapping
    renamed = raw.rename(columns={k: v for k, v in mapping.items() if k in raw.columns})

    # Ensure target placeholders (harmless for inference)
    safe_target = SAFE_TARGET
    safe_target_log = SAFE_TARGET_LOG
    if safe_target and (safe_target not in renamed.columns):
        renamed[safe_target] = 0.0
    if safe_target_log and (safe_target_log not in renamed.columns):
        renamed[safe_target_log] = 0.0

    # Add missing training columns
    for col in feature_order:
        if col not in renamed.columns:
            renamed[col] = 0

    # Order columns
    renamed = renamed[feature_order].copy()

    # Build log target column if required
    if LOG_TRANSFORM and safe_target in renamed.columns:
        renamed[safe_target_log] = np.log1p(renamed[safe_target])
    else:
        renamed[safe_target_log] = 0.0

    # Align to model signature
    model = load_model(model_type)
    sig = model.signatures["serving_default"]
    expected_specs = sig.structured_input_signature[1]  # dict: name -> TensorSpec

    n = len(renamed)
    cols_in_expected_order = []
    data_dict = {}
    for feat_name, spec in expected_specs.items():
        np_dtype = tf_to_np_dtype(spec.dtype)
        if feat_name in renamed.columns:
            data_dict[feat_name] = renamed[feat_name].astype(np_dtype, copy=False).values
        else:
            data_dict[feat_name] = np.zeros(n, dtype=np_dtype)
        cols_in_expected_order.append(feat_name)

    X = pd.DataFrame(data_dict, columns=cols_in_expected_order)

    # Final NA cleanup
    for c in X.columns:
        if np.issubdtype(X[c].dtype, np.floating):
            X[c] = X[c].fillna(0.0).astype(X[c].dtype)
        else:
            X[c] = X[c].fillna(0).astype(X[c].dtype)

    meta_out = {
        "LOG_TRANSFORM": LOG_TRANSFORM,
        "safe_target": safe_target,
        "safe_target_log": safe_target_log,
        "n_features_fed": X.shape[1],
    }

    return X, meta_out


def predict(payload: dict,model_type: str) -> tuple[float, dict]:
    
    X, meta = build_features(payload,model_type)
    model = load_model(model_type)
    infer_ds = tfdf.keras.pd_dataframe_to_tf_dataset(X, task=tfdf.keras.Task.REGRESSION)
    pred_log = model.predict(infer_ds, verbose=0).reshape(-1)
    if meta["LOG_TRANSFORM"]:
        pred = np.expm1(pred_log)
    else:
        pred = pred_log
    return float(pred[0]), meta