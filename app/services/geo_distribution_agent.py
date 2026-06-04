from __future__ import annotations
"""
Geographic Distribution Agent

Analyzes the geographic distribution of startups in the same industry/field
as the given startup profile using real data from the dataset.

Agent Steps:
  Step 1 — Data Crunch  : Compute actual distribution from filled_tf_df.csv
                          (country breakdown, city hotspots, regional share %)
  Step 2 — Gemini Layer : Strategic analysis — why startups cluster where they do,
                          where this startup fits geographically, positioning insights

Output is suitable for rendering charts + insights at the bottom of a dashboard.

Redis Cache:
  - Cache key: SHA256 hash of normalized profile
  - TTL: 24 hours (configurable via GEO_CACHE_TTL env var)
  - Cache hit  → ~50ms
  - Cache miss → ~5-8s (1 data crunch + 1 Gemini call)
"""

import os
import re
import json
import hashlib
from typing import Optional

import numpy as np
import pandas as pd
from google import genai
from google.genai import types

# ----------------------------- Cache Setup ----------------------------- #

try:
    import redis
    _redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True,
    )
    _redis_client.ping()
    REDIS_AVAILABLE = True
    print("[geo_agent] ✅ Redis connected")
except Exception as e:
    _redis_client = None
    REDIS_AVAILABLE = False
    print(f"[geo_agent] ⚠️ Redis unavailable, using in-memory cache: {e}")

_memory_cache: dict = {}
_CACHE_TTL = int(os.getenv("GEO_CACHE_TTL", 86400))
_FALLBACK_MODELS = ["gemini-2.5-flash"]

# Dataset path
_DEFAULT_CSV = os.getenv("GEO_DATA_CSV", "filled_tf_df.csv")


# ----------------------------- Cache Helpers ----------------------------- #

def _make_cache_key(profile: dict) -> str:
    normalized = json.dumps(profile, sort_keys=True, default=str).lower()
    return "geo:" + hashlib.sha256(normalized.encode()).hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    if REDIS_AVAILABLE:
        try:
            val = _redis_client.get(key)
            if val:
                print(f"[geo_agent] 🎯 Cache HIT (Redis): {key[:30]}...")
                return json.loads(val)
        except Exception as e:
            print(f"[geo_agent] Cache get error: {e}")
    if key in _memory_cache:
        print(f"[geo_agent] 🎯 Cache HIT (memory): {key[:30]}...")
        return _memory_cache[key]
    return None


def _cache_set(key: str, value: dict) -> None:
    if REDIS_AVAILABLE:
        try:
            _redis_client.setex(key, _CACHE_TTL, json.dumps(value, default=str))
            print(f"[geo_agent] 💾 Cached in Redis for {_CACHE_TTL}s")
            return
        except Exception as e:
            print(f"[geo_agent] Cache set error: {e}")
    _memory_cache[key] = value
    print(f"[geo_agent] 💾 Cached in memory")


# ----------------------------- Gemini Helpers ----------------------------- #

def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _call_gemini(prompt: str) -> Optional[dict]:
    client = _get_client()
    if not client:
        return None

    env_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    models_to_try = [env_model] + [m for m in _FALLBACK_MODELS if m != env_model]

    for model in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                    ),
                )
                text = response.text or ""
                print(f"[geo_agent] Response len={len(text)} model={model}")
                if not text.strip():
                    print(f"[geo_agent] Empty response from {model}")
                    break
                # Extract JSON — find outermost { } block
                start = text.find('{')
                end = text.rfind('}')
                if start == -1 or end == -1 or end <= start:
                    print(f"[geo_agent] Truncated JSON, retrying..." if attempt == 0 else f"[geo_agent] No JSON after retry.")
                    if attempt == 0:
                        import time; time.sleep(2)
                        continue
                    break
                json_str = text[start:end+1]
                try:
                    parsed = json.loads(json_str)
                    print(f"[geo_agent] ✓ Gemini: {model}")
                    return parsed
                except json.JSONDecodeError as je:
                    print(f"[geo_agent] JSON parse failed: {je}")
                    break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    print(f"[geo_agent] {model} quota/rate limited, skipping")
                    break
                if "503" in err_str and attempt == 0:
                    print(f"[geo_agent] {model} overloaded, retrying in 3s...")
                    import time; time.sleep(3)
                    continue
                print(f"[geo_agent] {model} failed: {e}")
                break

    print(f"[geo_agent] All models exhausted.")
    return None


# ----------------------------- Data Helpers ----------------------------- #

def _parse_hq(s) -> tuple:
    """Parse 'City, State, Country' → (city, country)"""
    if not isinstance(s, str) or not s.strip():
        return (None, None)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    city = parts[0] if parts else None
    country = parts[-1] if len(parts) >= 2 else None
    return (city, country)


def _split_industries(val) -> list:
    """Split comma/pipe separated industries into a list."""
    if not isinstance(val, str):
        return []
    val = val.replace("|", ",").replace("/", ",")
    return [p.strip() for p in val.split(",") if p.strip()]


def _fuzzy_match_industry(target: str, industries: list) -> bool:
    """Check if any industry in the list fuzzy-matches the target."""
    target_lower = str(target).lower().strip()
    target_words = set(target_lower.split())
    for ind in industries:
        # Guard against float/None values in the column
        if ind is None or (isinstance(ind, float) and not isinstance(ind, str)):
            continue
        ind_lower = str(ind).lower().strip()
        if not ind_lower or ind_lower == "nan":
            continue
        # Exact match
        if target_lower == ind_lower:
            return True
        # Substring match
        if target_lower in ind_lower or ind_lower in target_lower:
            return True
        # Word overlap (at least 1 word in common)
        ind_words = set(ind_lower.split())
        if target_words & ind_words:
            return True
    return False


# ----------------------------- Step 1: Data Crunch ----------------------------- #

def _step_data_crunch(startup_profile: dict, csv_path: str) -> Optional[dict]:
    """
    Load filled_tf_df.csv, filter by industry, compute geographic distribution.
    Returns structured data: country breakdown, city hotspots, regional share.
    """
    print(f"[geo_agent] Step 1/2 — Data Crunch")

    industry = startup_profile.get("industry", "")
    if not industry:
        return None

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"[geo_agent] Failed to load CSV: {e}")
        return None

    # Find industry column
    ind_col = None
    for col in df.columns:
        if col.lower() in ["industries", "industry", "categories"]:
            ind_col = col
            break

    if ind_col is None:
        print(f"[geo_agent] No industry column found")
        return None

    # Find HQ column
    hq_col = None
    for col in df.columns:
        if "headquarters" in col.lower() or (
            "hq" in col.lower() and "location" in col.lower()
        ):
            hq_col = col
            break
    if hq_col is None:
        for col in df.columns:
            if col.lower() == "location":
                hq_col = col
                break

    if hq_col is None:
        print(f"[geo_agent] No HQ location column found")
        return None

    # Explode industries and filter by target industry
    df["__industries_list"] = df[ind_col].apply(_split_industries)
    df_exp = df.explode("__industries_list")
    df_exp["__industries_list"] = df_exp["__industries_list"].astype(str).str.strip()

    # Filter to matching industry rows
    mask = df_exp["__industries_list"].apply(
        lambda x: _fuzzy_match_industry(industry, [str(x)])
    )
    filtered = df_exp[mask].copy()

    total_in_industry = len(filtered)
    if total_in_industry == 0:
        print(f"[geo_agent] No startups found for industry: {industry}")
        return None

    print(f"[geo_agent] Found {total_in_industry} startups in '{industry}'")

    # Parse HQ → city, country
    parsed = filtered[hq_col].apply(_parse_hq)
    filtered = filtered.copy()
    filtered["__city"] = parsed.apply(lambda x: x[0])
    filtered["__country"] = parsed.apply(lambda x: x[1])

    # ── Country distribution ──
    country_counts = (
        filtered["__country"]
        .dropna()
        .value_counts()
        .head(15)
        .reset_index()
    )
    country_counts.columns = ["country", "count"]
    country_counts["percentage"] = (
        (country_counts["count"] / total_in_industry * 100).round(1)
    )
    country_list = country_counts.to_dict(orient="records")

    # ── City hotspots ──
    city_counts = (
        filtered["__city"]
        .dropna()
        .value_counts()
        .head(20)
        .reset_index()
    )
    city_counts.columns = ["city", "count"]
    city_counts["percentage"] = (
        (city_counts["count"] / total_in_industry * 100).round(1)
    )
    city_list = city_counts.to_dict(orient="records")

    # ── Regional breakdown using Headquarters Regions if available ──
    region_list = []
    region_col = None
    for col in df.columns:
        if "region" in col.lower() and "hq" in col.lower():
            region_col = col
            break
    if region_col is None:
        for col in df.columns:
            if col.lower() in ["headquarters regions", "hq regions", "region"]:
                region_col = col
                break

    if region_col is not None:
        # Re-filter original df for region (before explode)
        df["__industries_list2"] = df[ind_col].apply(_split_industries)
        df_exp2 = df.explode("__industries_list2")
        mask2 = df_exp2["__industries_list2"].apply(
            lambda x: _fuzzy_match_industry(industry, [str(x)])
        )
        filtered2 = df_exp2[mask2].copy()

        # Split multi-region entries (e.g. "Asia-Pacific, Europe")
        filtered2["__region_list"] = filtered2[region_col].apply(
            lambda x: [r.strip() for r in str(x).split(",") if r.strip()] if isinstance(x, str) else []
        )
        region_exp = filtered2.explode("__region_list")
        region_exp = region_exp[region_exp["__region_list"].astype(str).str.len() > 2]

        region_counts = (
            region_exp["__region_list"]
            .value_counts()
            .head(10)
            .reset_index()
        )
        region_counts.columns = ["region", "count"]
        total_with_region = region_counts["count"].sum()
        if total_with_region > 0:
            region_counts["percentage"] = (
                (region_counts["count"] / total_with_region * 100).round(1)
            )
        else:
            region_counts["percentage"] = 0.0
        region_list = region_counts.to_dict(orient="records")

    # ── Top country for context ──
    top_country = country_list[0]["country"] if country_list else "Unknown"
    top_city = city_list[0]["city"] if city_list else "Unknown"

    return {
        "industry": industry,
        "total_startups_in_industry": int(total_in_industry),
        "top_country": top_country,
        "top_city": top_city,
        "country_distribution": country_list,
        "city_hotspots": city_list,
        "regional_breakdown": region_list,
    }


# ----------------------------- Step 2: Gemini Analysis ----------------------------- #

def _step_gemini_analysis(startup_profile: dict, distribution_data: dict) -> Optional[dict]:
    """Gemini analysis on top of the real distribution data."""
    print(f"[geo_agent] Step 2/2 — Gemini Analysis")

    top_countries = distribution_data.get("country_distribution", [])[:5]
    top_cities = distribution_data.get("city_hotspots", [])[:5]
    total = distribution_data.get("total_startups_in_industry", 0)
    industry = distribution_data.get("industry", "N/A")
    top_country = distribution_data.get("top_country", "N/A")
    top_city = distribution_data.get("top_city", "N/A")

    startup_hq = startup_profile.get("headquarters_location", startup_profile.get("region", "N/A"))
    startup_stage = startup_profile.get("stage", startup_profile.get("last_funding_type", "N/A"))

    country_summary = ", ".join([f"{c['country']} ({c['percentage']}%)" for c in top_countries])
    city_summary = ", ".join([f"{c['city']} ({c['count']} startups)" for c in top_cities])

    prompt = f"""
You are a startup ecosystem geographer. Analyze the geographic distribution of {industry} startups
based on real data and provide strategic insights for the given startup.

Real Data Summary:
- Total {industry} startups in dataset: {total}
- Top countries: {country_summary}
- Top cities: {city_summary}
- Dominant hub: {top_city}, {top_country}

This Startup:
- Industry: {industry}
- HQ / Region: {startup_hq}
- Stage: {startup_stage}

Based on this real distribution data, provide strategic geographic insights.
Be specific — reference the actual concentrations and percentages.

Return ONLY this JSON, no markdown:
{{
  "distribution_summary": "2-3 sentence overview of where {industry} startups are concentrated and why",
  "dominant_hub_analysis": "why {top_city} / {top_country} dominates this space",
  "startup_geo_position": "how this startup's location compares to the main clusters",
  "geographic_advantages": ["advantage of being in startup's location 1", "advantage 2"],
  "geographic_challenges": ["challenge of being outside main cluster 1", "challenge 2"],
  "strategic_recommendations": ["geo strategy 1", "geo strategy 2", "geo strategy 3"],
  "expansion_targets": ["target location 1 with reason", "target location 2 with reason"],
  "verdict": "one bold sentence on the startup's geographic positioning"
}}
"""
    return _call_gemini(prompt)


# ----------------------------- Main Agent Entry Point ----------------------------- #

def run_geo_distribution_agent(startup_profile: dict, csv_path: str = None) -> dict:
    """
    Run the 2-step Geographic Distribution Agent.

    Step 1: Crunch real data from filled_tf_df.csv
    Step 2: Gemini strategic analysis on top of the distribution

    Args:
        startup_profile: dict with at minimum 'industry'. Also uses
                         'headquarters_location', 'region', 'stage'.
        csv_path: optional override for dataset path

    Returns:
        Full geographic distribution report with chart-ready data + insights.
    """
    if csv_path is None:
        csv_path = _DEFAULT_CSV

    cache_key = _make_cache_key({**startup_profile, "_csv": csv_path})

    # Cache check
    cached = _cache_get(cache_key)
    if cached:
        cached["cache_hit"] = True
        return cached

    industry = startup_profile.get("industry", "Unknown")
    print(f"\n[geo_agent] 🚀 Starting geo distribution for: {industry}")

    report = {
        "cache_hit": False,
        "industry": industry,
        "agent_steps_completed": 0,
    }

    # Step 1 — Data crunch
    distribution = _step_data_crunch(startup_profile, csv_path)
    if not distribution:
        report["error"] = f"No data found for industry '{industry}' in dataset."
        return report

    report.update(distribution)
    report["agent_steps_completed"] = 1

    # Step 2 — Gemini analysis
    analysis = _step_gemini_analysis(startup_profile, distribution)
    if analysis:
        report["geo_insights"] = analysis
        report["agent_steps_completed"] = 2
    else:
        report["geo_insights"] = None
        report["warning"] = "Gemini analysis unavailable. Distribution data returned."

    print(f"[geo_agent] ✅ Complete for: {industry}")
    _cache_set(cache_key, report)
    return report
