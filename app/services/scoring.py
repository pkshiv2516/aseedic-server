# quantumfai/scoring.py
"""
QuantumFAI — Startup scoring, token allocation, and valuation
Inputs (per startup row):
    Industry
    Headquarters Location
    Founded Date
    Company Type
    Number of Employees
    Number of Founders
    Annual Revenue Range
    Funding Status
    Total Funding Raised ($)
    Number of Funding Rounds
    Monthly Website Visits
    Currently Hiring?
    Patents Granted
    Trademarks Registered

Outputs (per startup row):
    QFS (0-100), Tokens, V_current (USD), plus debug columns.

Usage:
    import pandas as pd
    from app.services.scoring import score_startups

    df = pd.read_csv("startups.csv")  # columns as above
    out = score_startups(df, s_pool=1_000_000, gamma=1.2)
    out.to_csv("startups_scored.csv", index=False)
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------- Admin Settings ----------------------------- #

DEFAULT_SECTOR_MULTIPLIER: Dict[str, float] = {
    # Set sensible initial bands; tune later with backtests
    "SaaS": 8.0,
    "AI": 8.5,
    "AI/ML": 8.5,
    "HealthTech": 6.5,
    "Fintech": 7.0,
    "Biotech": 10.0,
    "EdTech": 5.0,
    "E-commerce": 4.0,
    "Marketplace": 4.5,
    "DeepTech": 9.0,
    "_default": 6.0,
}

DEFAULT_REGION_FACTOR: Dict[str, float] = {
    # Keep neutral unless you choose to tilt for ecosystem maturity
    "United States": 1.05,
    "USA": 1.05,
    "India": 1.00,
    "United Kingdom": 1.03,
    "UK": 1.03,
    "Singapore": 1.03,
    "Germany": 1.02,
    "Canada": 1.02,
    "_default": 1.00,
}

# Stage floor (USD) and nominal dilution (used in other variants; kept for completeness)
DEFAULT_STAGE_FLOOR_USD: Dict[str, float] = {
    "Idea": 1_000_000.0,
    "Pre-seed": 1_500_000.0,
    "Preseed": 1_500_000.0,
    "Seed": 5_000_000.0,
    "Series A": 15_000_000.0,
    "Series B": 50_000_000.0,
    "_default": 3_000_000.0,
}

# Stage index for token bias (lower = earlier)
DEFAULT_STAGE_INDEX: Dict[str, int] = {
    "Idea": 0,
    "Pre-seed": 1,
    "Preseed": 1,
    "Seed": 2,
    "Series A": 3,
    "Series B": 4,
    "_default": 2,
}

# Optional company type tilt (very light)
COMPANY_TYPE_TILT: Dict[str, float] = {
    "B2B": 0.15,
    "B2C": 0.00,
    "Marketplace": 0.10,
    "DeepTech": 0.25,
    "SaaS": 0.10,
    "_default": 0.00,
}

# QFS feature weights (start simple; tune later)
DEFAULT_WEIGHTS = {
    "z_ln_rev": 1.0,           # Annual Revenue (midpoint, ln1p, z)
    "z_emp": 0.8,              # Employees (z)
    "z_rounds": 0.6,           # Funding rounds (z)
    "g_age": 0.7,              # U-curve age sweet spot
    "t_founders": 0.5,         # Founders sweet spot (2-3 founders)
    "z_visits": 0.7,           # ln1p(website visits), z
    "z_ip": 0.6,               # ln1p(patents + trademarks), z
    "hiring": 0.3,             # currently hiring flag
    "sector_tilt": 0.5,        # (M_sector/6 - 1)
    "region_tilt": 0.2,        # (M_region - 1)
    "ctype_tilt": 0.2,         # Company Type small tilt
}

# Revenue-multiple blend
VAL_LAMBDA_PF = 0.25    # fundability lift on multiple (we use proxies, so keep modest)
VAL_KAPPA_QFS = 0.35    # QFS lift around stage floor
VAL_W_REV_RAW = 0.60    # raw weight for revenue anchor when revenue present
VAL_W_SCORE_RAW = 0.40  # raw weight for score-floor anchor (always present)


# ----------------------------- Utility Parsers ---------------------------- #

RANGE_RE = re.compile(
    r"""
    ^\s*
    (?P<lo>[\d\.,]+)?\s*
    (?:-|to|–|—)\s*
    (?P<hi>[\d\.,]+)\s*
    (?P<Unit>[kKmMbB])?
    \s*$
    """,
    re.VERBOSE,
)

AMOUNT_RE = re.compile(
    r"""
    ^\s*[\$₹]?
    (?P<num>[\d\.,]+)
    \s*(?P<Unit>[kKmMbB])?
    \s*$
    """,
    re.VERBOSE,
)

def _unit_to_multiplier(unit: Optional[str]) -> float:
    if not unit:
        return 1.0
    u = unit.lower()
    return {"k": 1e3, "m": 1e6, "b": 1e9}.get(u, 1.0)

def parse_amount(s: Optional[str]) -> Optional[float]:
    """Parse single amount like '1.2M', '500k', '3000000' -> float USD."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    m = AMOUNT_RE.match(str(s).strip())
    if not m:
        return None
    num = float(m.group("num").replace(",", ""))
    mult = _unit_to_multiplier(m.group("Unit"))
    return num * mult

def parse_revenue_range(s: Optional[str]) -> Optional[float]:
    """
    Parse 'Annual Revenue Range' like '0-1M', '1M–5M', '500k-2M'.
    Returns midpoint in USD.
    """
    if s is None:
        return None
    s_ = str(s).strip()
    # If it's a single amount, just return it
    single = parse_amount(s_)
    if single is not None:
        return single
    m = RANGE_RE.match(s_)
    if not m:
        return None
    lo_str, hi_str, unit = m.group("lo"), m.group("hi"), m.group("Unit")
    mult = _unit_to_multiplier(unit)
    if lo_str:
        lo = float(lo_str.replace(",", "")) * (1.0 if unit else 1.0)
    else:
        lo = 0.0
    hi = float(hi_str.replace(",", "")) * (1.0 if unit else 1.0)
    # If unit is present, apply to both lo/hi that lack explicit units
    if unit:
        lo *= mult
        hi *= mult
    return (lo + hi) / 2.0

def extract_country(headquarters: Optional[str]) -> str:
    """
    Try to extract country as the last token after comma.
    Examples:
        'Hyderabad, India' -> 'India'
        'San Francisco, CA, United States' -> 'United States'
    """
    if not headquarters or not isinstance(headquarters, str):
        return ""
    parts = [p.strip() for p in headquarters.split(",") if p.strip()]
    if not parts:
        return ""
    return parts[-1]


# -------------------------- Core Scoring Machinery ------------------------ #

def ln1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, a_min=0, a_max=None))

def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd == 0 or np.isnan(sd):
        return np.zeros_like(x)
    return (x - mu) / sd

def g_age(age_years: np.ndarray, mu: float = 3.5, sigma: float = 2.0) -> np.ndarray:
    """U-curve sweet spot around ~3.5y; returns 0..1"""
    with np.errstate(invalid="ignore"):
        return np.exp(-((age_years - mu) / sigma) ** 2)

def t_founders(nf: np.ndarray, sweet: float = 2.5) -> np.ndarray:
    """Founders sweet spot around 2-3; returns 0..1"""
    with np.errstate(invalid="ignore"):
        return np.clip(1.0 - np.abs(nf - sweet) / sweet, 0.0, 1.0)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def stage_index(stage: Optional[str]) -> int:
    s = (stage or "").strip()
    return DEFAULT_STAGE_INDEX.get(s, DEFAULT_STAGE_INDEX["_default"])

def stage_floor_usd(stage: Optional[str]) -> float:
    s = (stage or "").strip()
    return DEFAULT_STAGE_FLOOR_USD.get(s, DEFAULT_STAGE_FLOOR_USD["_default"])

def sector_multiple(industry: Optional[str]) -> float:
    if not industry:
        return DEFAULT_SECTOR_MULTIPLIER["_default"]
    # pick first known key in text
    txt = industry.strip().lower()
    for key, mult in DEFAULT_SECTOR_MULTIPLIER.items():
        if key == "_default":
            continue
        if key.lower() in txt:
            return mult
    return DEFAULT_SECTOR_MULTIPLIER["_default"]

def region_factor_from_hq(hq: Optional[str]) -> float:
    country = extract_country(hq)
    if not country:
        return DEFAULT_REGION_FACTOR["_default"]
    return DEFAULT_REGION_FACTOR.get(country, DEFAULT_REGION_FACTOR["_default"])

def company_type_tilt(ctype: Optional[str]) -> float:
    if not ctype:
        return COMPANY_TYPE_TILT["_default"]
    for k, v in COMPANY_TYPE_TILT.items():
        if k == "_default":
            continue
        if k.lower() in ctype.strip().lower():
            return v
    return COMPANY_TYPE_TILT["_default"]

def hiring_flag(v: Optional[str]) -> float:
    if isinstance(v, str):
        return 1.0 if v.strip().lower() in {"yes", "y", "true", "currently hiring", "open"} else 0.0
    if isinstance(v, (int, float, bool)):
        return 1.0 if bool(v) else 0.0
    return 0.0

def age_years_from_date(s: Optional[str]) -> float:
    if not s or not isinstance(s, str):
        return np.nan
    try:
        # Allow various formats
        dt = None
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y", "%Y"):
            try:
                dt = datetime.strptime(s.strip(), fmt)
                break
            except Exception:
                continue
        if dt is None:
            # last resort: just year at end
            year_m = re.search(r"(\d{4})", s)
            if year_m:
                dt = datetime(int(year_m.group(1)), 7, 1)
        if dt is None:
            return np.nan
        today = datetime.now(timezone.utc)
        return max(0.0, (today - dt.replace(tzinfo=timezone.utc)).days / 365.25)
    except Exception:
        return np.nan


@dataclass
class QFAISettings:
    sector_multiplier: Dict[str, float] = field(default_factory=lambda: DEFAULT_SECTOR_MULTIPLIER)
    region_factor: Dict[str, float] = field(default_factory=lambda: DEFAULT_REGION_FACTOR)
    weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS)
    stage_floor_usd: Dict[str, float] = field(default_factory=lambda: DEFAULT_STAGE_FLOOR_USD)
    stage_index_map: Dict[str, int] = field(default_factory=lambda: DEFAULT_STAGE_INDEX)
    # Valuation knobs
    lambda_pf: float = VAL_LAMBDA_PF
    kappa_qfs: float = VAL_KAPPA_QFS
    w_rev_raw: float = VAL_W_REV_RAW
    w_score_raw: float = VAL_W_SCORE_RAW


def compute_qfs(df: pd.DataFrame, settings: Optional[QFAISettings] = None) -> pd.Series:
    """Compute QuantumFAI Score (0..100) for each row."""
    S = settings or QFAISettings()

    # Derived numeric features
    rev_mid = df.get("Annual Revenue Range", pd.Series([None] * len(df))).apply(parse_revenue_range).astype("float64")
    emp = pd.to_numeric(df.get("Number of Employees", np.nan), errors="coerce")
    rounds = pd.to_numeric(df.get("Number of Funding Rounds", np.nan), errors="coerce")
    nf = pd.to_numeric(df.get("Number of Founders", np.nan), errors="coerce")
    visits = pd.to_numeric(df.get("Monthly Website Visits", np.nan), errors="coerce")
    patents = pd.to_numeric(df.get("Patents Granted", np.nan), errors="coerce")
    tmarks = pd.to_numeric(df.get("Trademarks Registered", np.nan), errors="coerce")
    ip = (patents.fillna(0) + tmarks.fillna(0)).astype("float64")
    hiring = df.get("Currently Hiring?", pd.Series([None] * len(df))).apply(hiring_flag).astype("float64")
    age = df.get("Founded Date", pd.Series([None] * len(df))).apply(age_years_from_date).astype("float64")

    industry = df.get("Industry", pd.Series([""] * len(df))).astype(str)
    hq = df.get("Headquarters Location", pd.Series([""] * len(df))).astype(str)
    ctype = df.get("Company Type", pd.Series([""] * len(df))).astype(str)

    m_sector = industry.apply(sector_multiple).astype("float64")
    m_region = hq.apply(region_factor_from_hq).astype("float64")
    ctype_t = ctype.apply(company_type_tilt).astype("float64")

    # z-scores
    z_ln_rev = zscore(ln1p(rev_mid.fillna(0).values))
    z_emp = zscore(emp.fillna(0).values)
    z_rounds = zscore(rounds.fillna(0).values)
    z_visits = zscore(ln1p(visits.fillna(0).values))
    z_ip = zscore(ln1p(ip.fillna(0).values))

    # shape functions 0..1
    gA = g_age(age.fillna(np.nan).values)
    tF = t_founders(nf.fillna(np.nan).values)

    # tilts (centered)
    sector_tilt = (m_sector.values / 6.0) - 1.0
    region_tilt = (m_region.values) - 1.0

    # Weighted sum
    W = S.weights
    X = (
        W["z_ln_rev"] * z_ln_rev
        + W["z_emp"] * z_emp
        + W["z_rounds"] * z_rounds
        + W["g_age"] * gA
        + W["t_founders"] * tF
        + W["z_visits"] * z_visits
        + W["z_ip"] * z_ip
        + W["hiring"] * hiring.values
        + W["sector_tilt"] * sector_tilt
        + W["region_tilt"] * region_tilt
        + W["ctype_tilt"] * ctype_t.values
    )

    # Logistic to 0..100
    qfs = 100.0 * sigmoid(X)
    return pd.Series(qfs, index=df.index, name="QFS")


def allocate_tokens(
    df: pd.DataFrame,
    qfs: pd.Series,
    s_pool: int = 1_000_000,
    gamma: float = 1.2,
    tokens_min: int = 500,
    tokens_max: int = 100_000,
) -> pd.Series:
    """Allocate tokens proportionally across this cohort."""
    stages = df.get("Funding Status", pd.Series([""] * len(df))).astype(str)
    stage_idx = stages.apply(stage_index).astype("int64")
    # Early-stage bias (1/(1+s)) — adjust if you want the opposite
    W_stage = 1.0 / (1.0 + stage_idx.values)

    # Basic data completeness bump (KYC-like) — fields present
    fields = [
        "Industry",
        "Headquarters Location",
        "Founded Date",
        "Number of Employees",
        "Number of Founders",
        "Annual Revenue Range",
        "Funding Status",
        "Number of Funding Rounds",
        "Monthly Website Visits",
    ]
    completeness = df[fields].notna().sum(axis=1) / len(fields)
    W_verif = 1.0 + 0.05 * (completeness.values >= 0.8)  # +5% if good completeness

    score = (np.power(qfs.values, gamma)) * W_stage * W_verif
    total = np.sum(score)
    if total <= 0:
        # Fallback to equal split
        raw = np.ones_like(score) / len(score)
    else:
        raw = score / total

    tokens = np.floor(s_pool * raw).astype(int)
    tokens = np.clip(tokens, tokens_min, tokens_max)
    return pd.Series(tokens, index=df.index, name="Tokens")


def estimate_valuation(
    df: pd.DataFrame,
    qfs: pd.Series,
    settings: Optional[QFAISettings] = None
) -> pd.Series:
    """Two-anchor valuation (USD): Revenue multiple + Stage floor (QFS-lift)."""
    S = settings or QFAISettings()

    # Anchors
    rev_mid = df.get("Annual Revenue Range", pd.Series([None] * len(df))).apply(parse_revenue_range).astype("float64")
    industry = df.get("Industry", pd.Series([""] * len(df))).astype(str)
    hq = df.get("Headquarters Location", pd.Series([""] * len(df))).astype(str)
    stage = df.get("Funding Status", pd.Series([""] * len(df))).astype(str)

    m_sector = industry.apply(sector_multiple).astype("float64")
    m_region = hq.apply(region_factor_from_hq).astype("float64")

    # Revenue anchor (if revenue present)
    z_qfs = zscore(qfs.values)  # cohort-relative lift
    has_rev = ~rev_mid.isna()
    V_rev = rev_mid.fillna(0.0) * m_sector * m_region * (1.0 + S.lambda_pf * zscore(rev_mid.fillna(0.0).values))
    # Score-floor anchor (always present)
    base_floor = stage.apply(stage_floor_usd).astype("float64")
    V_score = base_floor * np.exp(S.kappa_qfs * z_qfs)

    # Blend
    w_rev_raw = np.where(has_rev.values, S.w_rev_raw, 0.0)
    w_score_raw = np.full(len(df), S.w_score_raw)
    w_sum = w_rev_raw + w_score_raw
    w_rev = np.divide(w_rev_raw, w_sum, out=np.zeros_like(w_rev_raw), where=(w_sum > 0))
    w_score = np.divide(w_score_raw, w_sum, out=np.ones_like(w_score_raw), where=(w_sum > 0))

    lnV = w_rev * np.log(np.clip(V_rev.values, 1.0, None)) + w_score * np.log(np.clip(V_score.values, 1.0, None))
    V_current = np.exp(lnV)

    # Safety rails per stage
    V_min = 0.5 * base_floor.values
    V_max = 3.0 * base_floor.values
    V_current = np.clip(V_current, V_min, V_max)

    return pd.Series(V_current, index=df.index, name="V_current_USD")


def score_startups(
    df: pd.DataFrame,
    s_pool: int = 1_000_000,
    gamma: float = 1.2,
    settings: Optional[QFAISettings] = None,
) -> pd.DataFrame:
    """
    Main entrypoint: compute QFS, Tokens, V_current (USD).
    Returns original columns + outputs + useful debug columns.
    """
    # Defensive copy
    data = df.copy()

    qfs = compute_qfs(data, settings=settings)
    tokens = allocate_tokens(data, qfs, s_pool=s_pool, gamma=gamma)
    vcur = estimate_valuation(data, qfs, settings=settings)

    # Debug/explain columns (optional but helpful)
    data_out = data.assign(
        QFS=qfs.round(2),
        Tokens=tokens.astype(int),
        V_current_USD=np.round(vcur, 0).astype(int),
    )

    return data_out

