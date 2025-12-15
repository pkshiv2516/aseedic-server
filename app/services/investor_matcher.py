from __future__ import annotations
from pathlib import Path
from typing import Optional, List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random


def load_investor_data(csv_path: str | Path = "investor_db.csv") -> Optional[pd.DataFrame]:
    """Load investor CSV. Returns None if file does not exist."""
    path = Path(csv_path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    random.seed(42)
    if "response_likelihood" not in df.columns:
        df["response_likelihood"] = [round(random.uniform(0.4, 0.95), 2) for _ in range(len(df))]
    return df


def _preprocess_profiles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Tolerate missing columns by filling with empty strings
    out["industry"] = out.get("industry", "").fillna("")
    out["funding_stage"] = out.get("funding_stage", "").fillna("")
    out["location"] = out.get("location", "").fillna("")
    out["portfolio"] = out.get("portfolio", "").fillna("")
    out["profile_text"] = (
        out["industry"].astype(str) + " " +
        out["funding_stage"].astype(str) + " " +
        out["location"].astype(str) + " " +
        out["portfolio"].astype(str)
    )
    return out


def _extract_overlap(portfolio: str, keywords: List[str]) -> str:
    if pd.isna(portfolio):
        return ""
    matches = [p.strip() for p in str(portfolio).split(",") if any(k.lower() in p.lower() for k in keywords)]
    return ", ".join(matches)


def match_investors(
    startup_profile: dict,
    df: pd.DataFrame,
    topk: int = 100,
) -> pd.DataFrame:
    """Compute investor matches and return topk results as a DataFrame."""
    dfp = _preprocess_profiles(df)

    industry = str(startup_profile.get("industry", ""))
    funding_stage = str(startup_profile.get("funding_stage", ""))
    region = str(startup_profile.get("region", ""))

    startup_text = f"{industry} {funding_stage} {region}".strip()
    corpus = dfp["profile_text"].tolist() + [startup_text]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    similarities = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1]).flatten()
    dfp["match_score"] = (similarities * 100).round(2)

    keywords = startup_text.lower().split()
    dfp["portfolio_overlap"] = dfp["portfolio"].apply(lambda p: _extract_overlap(p, keywords))

    dfp = dfp[dfp["match_score"] > 0].copy()

    # Weighted score: adjust weights if desired
    dfp["final_score"] = (
        (dfp["match_score"] * 0.6) +
        (dfp["response_likelihood"].fillna(0) * 100 * 0.2)
    ).round(2)

    dfp = dfp.sort_values(by="final_score", ascending=False).reset_index(drop=True)

    out = dfp[[
        "investor_name",
        "final_score",
        "match_score",
        "response_likelihood",
        "industry",
        "funding_stage",
        "location",
        "tag_line",
        "portfolio_overlap",
    ]].rename(columns={
        "industry": "focus",
        "funding_stage": "stage",
        "location": "geo_focus",
    })

    return out.head(topk)


