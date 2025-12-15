from __future__ import annotations
import os
import re
import math
from datetime import datetime
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

# Defaults (can be overridden by route params)
DEFAULT_INPUT = "filled_tf_df.csv"
DEFAULT_OUTDIR = "reco_artifacts"
RANDOM_SEED = 42
RECENT_YEARS = 5
WEIGHTS = dict(density=0.25, funding=0.25, success=0.30, growth=0.20)

POSITIVE_STATUS = ["active","operating","acquired","ipo","public","merged","subsidiary"]
NEGATIVE_STATUS = ["closed","defunct","shutdown","shut down","out of business","deadpooled","inactive","dissolved"]

POSSIBLE_INDUSTRY_COLS = ["Industries", "Industry", "Categories"]
POSSIBLE_STATUS_COLS   = ["Operating Status", "Status", "Company Status"]
POSSIBLE_FOUNDED_COLS  = ["Founded Year", "Founded Date", "Founded", "Year Founded"]
POSSIBLE_FUNDING_COLS  = ["Total Funding", "Total Funding USD", "Total Funding ($)", "Total Funding Amount"]
POSSIBLE_CITY_COLS     = ["City"]
POSSIBLE_STATE_COLS    = ["State", "State/Province"]
POSSIBLE_COUNTRY_COLS  = ["Country"]
POSSIBLE_HQ_COLS       = ["Headquarters Location", "HQ Location", "Location"]

FEATURE_COLS = [
    "DensityLog","FundingMedian","SuccessRate","RecentGrowth",
    "N","Success_n","Growth_n"
]


def find_col(df: pd.DataFrame, exact_candidates: List[str], substr_candidates: List[str]) -> Optional[str]:
    for c in exact_candidates:
        for col in df.columns:
            if col.strip().lower() == c.strip().lower():
                return col
    for col in df.columns:
        low = col.lower()
        if all(s.lower() in low for s in substr_candidates):
            return col
    for col in df.columns:
        low = col.lower()
        if any(s.lower() in low for s in substr_candidates):
            return col
    return None


def parse_hq_location(s: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not isinstance(s, str) or not s.strip():
        return (None, None, None)
    parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
    city = parts[0] if len(parts) >= 1 else None
    state = parts[1] if len(parts) >= 2 else None
    country = parts[2] if len(parts) >= 3 else (parts[1] if len(parts) == 2 else None)
    return (city, state, country)


def split_industries(val) -> List[str]:
    if not isinstance(val, str):
        return []
    s = val.replace("&", " and ").replace("/", ",").replace("|", ",")
    s = re.sub(r"\s+", " ", s)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    seen, out = set(), []
    for p in parts:
        k = p.lower()
        if k not in seen:
            seen.add(k); out.append(p)
    return out


def parse_money(x) -> float:
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip().replace(",", "").replace("₹", "").replace("$", "").replace("€", "").replace("£", "")
    m = re.match(r"^\s*([+-]?\d*\.?\d+)\s*([KkMmBb])?\s*$", s)
    if m:
        v = float(m.group(1)); suf = m.group(2)
        if suf:
            if suf.lower()=="k": v *= 1_000
            elif suf.lower()=="m": v *= 1_000_000
            elif suf.lower()=="b": v *= 1_000_000_000
        return v
    try: return float(s)
    except: return np.nan


def to_year(x) -> Optional[int]:
    if pd.isna(x): return None
    if isinstance(x,(int,float)) and not np.isnan(x):
        y=int(x)
        return y if 1900<=y<=datetime.now().year else None
    s=str(x).strip()
    m=re.search(r"(19|20)\d{2}", s)
    if m:
        y=int(m.group(0))
        return y if 1900<=y<=datetime.now().year else None
    for fmt in ("%Y-%m-%d","%d-%m-%Y","%m/%d/%Y","%d/%m/%Y"):
        try: return datetime.strptime(s, fmt).year
        except: pass
    return None


def classify_success(s) -> float:
    if pd.isna(s): return np.nan
    low=str(s).strip().lower()
    if any(k in low for k in NEGATIVE_STATUS): return 0.0
    if any(k in low for k in POSITIVE_STATUS): return 1.0
    return np.nan


def wilson_ci(k: float, n: float, z: float = 1.96):
    if n <= 0: return (np.nan, np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2/(2*n)) / denom
    margin = (z * math.sqrt((phat*(1-phat) + z**2/(4*n)) / n)) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return (phat, lo, hi)


def confidence_label(n: int) -> str:
    if n >= 50: return "High"
    if n >= 20: return "Medium"
    return "Low"


def robust_minmax_block(x: pd.Series) -> pd.Series:
    lo = x.quantile(0.05)
    hi = x.quantile(0.95)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series([0.5]*len(x), index=x.index)
    return (x.clip(lo,hi) - lo) / (hi - lo)


def prep(input_csv: str = DEFAULT_INPUT, outdir: str = DEFAULT_OUTDIR, recent_years: int = RECENT_YEARS) -> Dict[str, str]:
    outdir = str(outdir)
    os.makedirs(outdir, exist_ok=True)
    agg_path = f"{outdir}/agg_features.csv"
    cand_path = f"{outdir}/candidates_topk.csv"  # filled later
    df = pd.read_csv(input_csv, low_memory=False)

    industry_col = find_col(df, POSSIBLE_INDUSTRY_COLS, ["industr","categor"]) or "__Industries"
    if industry_col == "__Industries":
        df[industry_col] = "Unknown"
    status_col   = find_col(df, POSSIBLE_STATUS_COLS, ["status"])
    founded_col  = find_col(df, POSSIBLE_FOUNDED_COLS, ["found"])
    funding_col  = find_col(df, POSSIBLE_FUNDING_COLS, ["fund"])
    city_col     = find_col(df, POSSIBLE_CITY_COLS, ["city"])
    state_col    = find_col(df, POSSIBLE_STATE_COLS, ["state","provin","region"])
    country_col  = find_col(df, POSSIBLE_COUNTRY_COLS, ["country"])
    hq_col       = find_col(df, POSSIBLE_HQ_COLS, ["headquarters","location"])

    if (city_col is None or state_col is None or country_col is None) and hq_col is not None:
        parsed = df[hq_col].apply(parse_hq_location)
        if city_col is None:    df["__City"]    = parsed.apply(lambda x: x[0]); city_col="__City"
        if state_col is None:   df["__State"]   = parsed.apply(lambda x: x[1]); state_col="__State"
        if country_col is None: df["__Country"] = parsed.apply(lambda x: x[2]); country_col="__Country"

    for src, dst in [(city_col,"City"), (state_col,"State"), (country_col,"Country")]:
        df[dst] = df[src] if src is not None else None
        df[dst] = df[dst].astype(str).where(df[dst].notna(), None)
        df[dst] = df[dst].apply(lambda x: x.strip().title() if isinstance(x,str) else x)

    df["__IndustriesList"] = df[industry_col].apply(split_industries)
    df = df.explode("__IndustriesList").rename(columns={"__IndustriesList":"Industry"})
    df["Industry"] = df["Industry"].fillna("Unknown").astype(str).str.strip()
    df = df[df["Industry"]!=""]

    df["FundingNumeric"] = df[funding_col].apply(parse_money) if funding_col else np.nan

    CURRENT_YEAR = datetime.now().year
    RECENT_START = CURRENT_YEAR - recent_years
    df["FoundedYear"] = df[founded_col].apply(to_year) if founded_col else None
    df["RecentFlag"] = df["FoundedYear"].apply(lambda y: 1.0 if (isinstance(y,int) and y>=RECENT_START) else np.nan)

    df["SuccessBinary"] = df[status_col].apply(classify_success).astype("float") if status_col else np.nan

    df["SuccessSumHelper"]  = df["SuccessBinary"].fillna(0.0).astype(float)
    df["SuccessValidHelper"]= np.isfinite(df["SuccessBinary"]).astype(int)
    df["RecentSumHelper"]   = df["RecentFlag"].fillna(0.0).astype(float)
    df["RecentValidHelper"] = np.isfinite(df["RecentFlag"]).astype(int)

    def group_level(level_name: str) -> pd.DataFrame:
        sub = df.copy()
        sub = sub[(sub[level_name].astype(str).str.strip().str.lower()!="none") & (sub[level_name].astype(str).str.strip()!="")]
        g = sub.groupby(["Industry", level_name], observed=True).agg(
            N=("Industry","size"),
            FundingMedian=("FundingNumeric","median"),
            SuccessSum=("SuccessSumHelper","sum"),
            SuccessN=("SuccessValidHelper","sum"),
            RecentSum=("RecentSumHelper","sum"),
            RecentN=("RecentValidHelper","sum")
        ).reset_index().rename(columns={level_name:"Location"})
        g["Geographic Level"] = level_name

        z=1.96
        g["DensityCount"] = g["N"].astype(float)
        g["DensityLog"]   = np.log1p(g["DensityCount"])
        se = np.sqrt(g["DensityCount"])
        lo = np.maximum(0.0, g["DensityCount"] - z*se)
        hi = g["DensityCount"] + z*se
        g["DensityLog_CI_Lo"] = np.log1p(lo)
        g["DensityLog_CI_Hi"] = np.log1p(hi)

        s_r = g.apply(lambda r: wilson_ci(r["SuccessSum"], r["SuccessN"]), axis=1, result_type="expand")
        g["SuccessRate"], g["SuccessRate_CI_Lo"], g["SuccessRate_CI_Hi"] = s_r[0], s_r[1], s_r[2]
        g_r = g.apply(lambda r: wilson_ci(r["RecentSum"], r["RecentN"]), axis=1, result_type="expand")
        g["RecentGrowth"], g["RecentGrowth_CI_Lo"], g["RecentGrowth_CI_Hi"] = g_r[0], g_r[1], g_r[2]

        g["Confidence"] = g["N"].apply(confidence_label)
        g["Success_n"] = g["SuccessN"]; g["Growth_n"] = g["RecentN"]
        return g

    city_df = group_level("City")
    state_df = group_level("State")
    country_df = group_level("Country")
    agg = pd.concat([city_df, state_df, country_df], ignore_index=True)

    agg["Density Score"] = agg.groupby(["Geographic Level","Industry"], observed=True)["DensityLog"].transform(robust_minmax_block)
    agg["Funding Score"] = agg.groupby(["Geographic Level","Industry"], observed=True)["FundingMedian"].transform(robust_minmax_block)
    agg["Success Score"] = agg.groupby(["Geographic Level","Industry"], observed=True)["SuccessRate"].transform(robust_minmax_block)
    agg["Growth Score"]  = agg.groupby(["Geographic Level","Industry"], observed=True)["RecentGrowth"].transform(robust_minmax_block)

    agg["Composite Score"] = (
        WEIGHTS["density"] * agg["Density Score"].fillna(0) +
        WEIGHTS["funding"] * agg["Funding Score"].fillna(0) +
        WEIGHTS["success"] * agg["Success Score"].fillna(0) +
        WEIGHTS["growth"]  * agg["Growth Score"].fillna(0)
    )

    agg.sort_values(["Geographic Level","Industry","Composite Score"], ascending=[True,True,False]).to_csv(agg_path, index=False)

    return {"agg_features_csv": agg_path, "candidates_csv": cand_path}


def generate_candidates(outdir: str, topk: int = 10) -> str:
    agg_path = f"{outdir}/agg_features.csv"
    cand_path = f"{outdir}/candidates_topk.csv"
    agg = pd.read_csv(agg_path)
    agg["rank_within_industry"] = agg.groupby(["Geographic Level","Industry"])["Composite Score"].rank(method="first", ascending=False)
    top = agg[agg["rank_within_industry"]<=topk].sort_values(["Geographic Level","Industry","Composite Score"], ascending=[True,True,False])
    top.to_csv(cand_path, index=False)
    return cand_path


def build_ltr(outdir: str) -> str:
    import joblib
    from sklearn.preprocessing import RobustScaler

    agg_path = f"{outdir}/agg_features.csv"
    ltr_csv  = f"{outdir}/ltr_train.csv"
    scaler_p = f"{outdir}/scaler.pkl"

    df = pd.read_csv(agg_path)
    X = df[FEATURE_COLS].fillna(0.0).copy()
    y = df["Composite Score"].astype(float).values
    qid = (df["Geographic Level"].astype(str) + " :: " + df["Industry"].astype(str)).astype("category").cat.codes

    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X)
    joblib.dump(scaler, scaler_p)

    out = df[["Geographic Level","Industry","Location"]].copy()
    out["qid"] = qid
    out["label"] = y
    for i, col in enumerate(FEATURE_COLS):
        out[col] = Xs[:, i]
    out.to_csv(ltr_csv, index=False)
    return ltr_csv


def train_regressor(outdir: str) -> Dict[str, str]:
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import ndcg_score
    import numpy as np
    from scipy.stats import spearmanr

    agg_path = f"{outdir}/agg_features.csv"
    model_p  = f"{outdir}/regressor.joblib"
    scaler_p = f"{outdir}/scaler.pkl"

    df = pd.read_csv(agg_path)
    X = df[FEATURE_COLS].fillna(0.0).copy()
    y = df["Composite Score"].astype(float).values
    qid = (df["Geographic Level"].astype(str) + " :: " + df["Industry"].astype(str)).astype("category").cat.codes
    uq = np.unique(qid)
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(uq)
    split = int(0.8 * len(uq))
    train_q = set(uq[:split]); val_q = set(uq[split:])

    scaler = RobustScaler().fit(X)
    Xs = scaler.transform(X)
    joblib.dump(scaler, scaler_p)

    model = RandomForestRegressor(
        n_estimators=500, max_depth=None, random_state=RANDOM_SEED, n_jobs=-1
    )
    model.fit(Xs[np.isin(qid, list(train_q))], y[np.isin(qid, list(train_q))])
    joblib.dump(model, model_p)

    df["pred_reg"] = model.predict(Xs)

    def group_ndcg(d, pred_col, k=10):
        scores = []
        for (_, _), g in d.groupby(["Geographic Level", "Industry"]):
            m = len(g)
            if m < 2:
                continue
            y_true = g["Composite Score"].to_numpy().reshape(1, m)
            y_pred = g[pred_col].to_numpy().reshape(1, m)
            k_eff = min(k, m)
            scores.append(ndcg_score(y_true, y_pred, k=k_eff))
        return float(np.mean(scores)) if scores else float("nan")

    def group_spearman(d, pred_col):
        corrs = []
        for (_, _), g in d.groupby(["Geographic Level", "Industry"]):
            if len(g) >= 3:
                corrs.append(spearmanr(g["Composite Score"].values, g[pred_col].values).correlation)
        return float(np.nanmean(corrs)) if corrs else float("nan")

    def _nan_to_none(x):
        return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)

    val = df[np.isin(qid, list(val_q))]
    val_ndcg = val.groupby(["Geographic Level","Industry"]).filter(lambda g: len(g) >= 2)

    ndcg5  = group_ndcg(val_ndcg, "pred_reg", k=5)
    ndcg10 = group_ndcg(val_ndcg, "pred_reg", k=10)
    sp     = group_spearman(val, "pred_reg")

    metrics = dict(
        ndcg5=_nan_to_none(ndcg5),
        ndcg10=_nan_to_none(ndcg10),
        spearman=_nan_to_none(sp),
        val_groups_total=int(val.groupby(["Geographic Level","Industry"]).ngroups),
        val_groups_used_for_ndcg=int(val_ndcg.groupby(["Geographic Level","Industry"]).ngroups)
    )
    return {"model_path": model_p, "scaler_path": scaler_p, "metrics": metrics}


def recommend(outdir: str, level: str, industry: str, topk: int = 10, use_model: bool = True) -> pd.DataFrame:
    import joblib
    agg_path = f"{outdir}/agg_features.csv"
    model_p  = f"{outdir}/regressor.joblib"
    scaler_p = f"{outdir}/scaler.pkl"

    df = pd.read_csv(agg_path)
    sub = df[(df["Geographic Level"].str.lower()==level.lower())
             & (df["Industry"].str.lower()==industry.lower())].copy()
    if sub.empty:
        raise ValueError(f"No rows for Industry='{industry}' at Level='{level}'.")

    if use_model and joblib is not None and os.path.exists(model_p) and os.path.exists(scaler_p):
        try:
            scaler = joblib.load(scaler_p)
            model  = joblib.load(model_p)
            Xs = scaler.transform(sub[FEATURE_COLS].fillna(0.0))
            sub["score"] = model.predict(Xs)
        except Exception:
            sub["score"] = sub["Composite Score"]
    else:
        sub["score"] = sub["Composite Score"]

    cols = ["Geographic Level","Industry","Location","score","Composite Score",
            "DensityLog","FundingMedian","SuccessRate","N","Confidence"]
    return sub.sort_values("score", ascending=False).head(topk)[cols]



