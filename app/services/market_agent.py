from __future__ import annotations
"""
Market Analysis Agent

A 3-step agentic service with Redis caching for fast repeated lookups.

Agent Loop:
  Step 1 — Market Landscape  : TAM/SAM/SOM, growth rate, key segments, trends
  Step 2 — Intelligence Bundle: Entry barriers + customer profiling + demand signals
  Step 3 — Market Strategy   : Go-to-market positioning, timing, and expansion roadmap

Redis Cache:
  - Cache key: SHA256 hash of normalized startup profile
  - TTL: 24 hours (configurable via MARKET_CACHE_TTL env var)
  - Cache hit  → ~50ms response
  - Cache miss → ~10-15s (3 Gemini calls)
"""

import os
import re
import json
import time
import hashlib
from typing import Optional

from google import genai
from google.genai import types

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
    print("[market_agent] ✅ Redis connected")
except Exception as e:
    _redis_client = None
    REDIS_AVAILABLE = False
    print(f"[market_agent] ⚠️ Redis unavailable, using in-memory cache: {e}")

# In-memory fallback cache when Redis is unavailable
_memory_cache: dict = {}

_CACHE_TTL = int(os.getenv("MARKET_CACHE_TTL", 86400))

# Gemini model fallback list
_FALLBACK_MODELS = ["gemini-2.5-flash"]


# ----------------------------- Cache Helpers ----------------------------- #

def _make_cache_key(startup_profile: dict) -> str:
    normalized = json.dumps(startup_profile, sort_keys=True, default=str).lower()
    return "market:" + hashlib.sha256(normalized.encode()).hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    if REDIS_AVAILABLE:
        try:
            val = _redis_client.get(key)
            if val:
                print(f"[market_agent] 🎯 Cache HIT (Redis): {key[:30]}...")
                return json.loads(val)
        except Exception as e:
            print(f"[market_agent] Cache get error: {e}")
    # Fallback to in-memory
    if key in _memory_cache:
        print(f"[market_agent] 🎯 Cache HIT (memory): {key[:30]}...")
        return _memory_cache[key]
    return None


def _cache_set(key: str, value: dict) -> None:
    if REDIS_AVAILABLE:
        try:
            _redis_client.setex(key, _CACHE_TTL, json.dumps(value, default=str))
            print(f"[market_agent] 💾 Cached in Redis for {_CACHE_TTL}s")
            return
        except Exception as e:
            print(f"[market_agent] Cache set error: {e}")
    # Fallback to in-memory
    _memory_cache[key] = value
    print(f"[market_agent] 💾 Cached in memory")


# ----------------------------- Gemini Helpers ----------------------------- #

def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _call_agent_step(prompt: str, step_name: str) -> Optional[dict]:
    """Execute a single agent step with smart fallback — skips daily-exhausted models instantly."""
    client = _get_client()
    if not client:
        print(f"[market_agent:{step_name}] No API key configured.")
        return None

    env_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    models_to_try = [env_model] + [m for m in _FALLBACK_MODELS if m != env_model]

    last_error = None
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
                text = response.text
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if not match:
                    print(f"[market_agent:{step_name}] No JSON in response from {model}")
                    break
                print(f"[market_agent:{step_name}] ✓ {model}")
                return json.loads(match.group())

            except json.JSONDecodeError as e:
                print(f"[market_agent:{step_name}] JSON parse error with {model}: {e}")
                last_error = str(e)
                break
            except Exception as e:
                err_str = str(e)
                last_error = err_str
                # Daily quota exhausted — skip immediately, no retry
                if "429" in err_str and "PerDay" in err_str:
                    print(f"[market_agent:{step_name}] {model} daily quota exhausted, skipping")
                    break
                # Any 429 — skip immediately, don't retry
                if "429" in err_str:
                    print(f"[market_agent:{step_name}] {model} rate limited, skipping")
                    break
                # Overloaded — retry once
                if "503" in err_str and attempt == 0:
                    print(f"[market_agent:{step_name}] {model} overloaded, retrying in 3s...")
                    time.sleep(3)
                    continue
                print(f"[market_agent:{step_name}] {model} failed: {e}")
                break

    print(f"[market_agent:{step_name}] All models exhausted. Last: {last_error}")
    return None


# ----------------------------- Agent Steps ----------------------------- #

def _step_market_landscape(startup: dict) -> Optional[dict]:
    """Step 1 — TAM/SAM/SOM, growth rate, key segments, trends."""
    prompt = f"""
Market analyst. Quick market sizing for this startup.

{startup.get("company_name","Unnamed")} | {startup.get("industry","N/A")} | {startup.get("niche","N/A")} | {startup.get("region","Global")} | {startup.get("product_description","N/A")}

Return ONLY this JSON:
{{
  "tam": {{"value": "$Xb", "description": "one line"}},
  "sam": {{"value": "$Xb", "description": "one line"}},
  "som": {{"value": "$Xm", "description": "one line"}},
  "cagr": "X% (2024-2029)",
  "market_maturity": "Emerging/Growing/Mature/Declining",
  "key_segments": [{{"segment": "name", "size": "$X", "growth": "fast/moderate/slow"}}],
  "top_trends": ["trend1", "trend2", "trend3"],
  "market_summary": "2 sentences"
}}
"""
    return _call_agent_step(prompt, "market_landscape")


def _step_intelligence_bundle(startup: dict, landscape: dict) -> Optional[dict]:
    """Step 2 — Entry barriers + customer profiling + demand signals in one call."""
    prompt = f"""
Market analyst. Analyze entry barriers, customer profile, demand signals.

{startup.get("company_name","Unnamed")} | {startup.get("industry","N/A")} | {startup.get("product_description","N/A")} | Stage: {startup.get("stage","N/A")} | Target: {startup.get("target_market","N/A")}
TAM: {landscape.get("tam",{}).get("value","N/A")} | CAGR: {landscape.get("cagr","N/A")} | {landscape.get("market_maturity","N/A")}

Return ONLY this JSON:
{{
  "entry_barriers": [{{"barrier": "one line", "severity": "High/Medium/Low", "mitigation": "one line"}}],
  "customer_profile": {{
    "primary_persona": "one line",
    "pain_points": ["p1","p2","p3"],
    "buying_triggers": ["t1","t2"],
    "willingness_to_pay": "$X/month"
  }},
  "demand_signals": {{
    "current_demand": "High/Medium/Low",
    "demand_drivers": ["d1","d2"],
    "demand_risks": ["r1","r2"],
    "timing_assessment": "one line"
  }}
}}
"""
    return _call_agent_step(prompt, "intelligence_bundle")


def _step_market_strategy(startup: dict, landscape: dict, bundle: dict) -> Optional[dict]:
    """Step 3 — GTM positioning, timing, and expansion roadmap."""
    prompt = f"""
Market strategist. Concise GTM strategy.

{startup.get("company_name","Unnamed")} | {startup.get("industry","N/A")} | Stage: {startup.get("stage","N/A")} | {startup.get("region","Global")}
TAM: {landscape.get("tam",{}).get("value","N/A")} | SOM: {landscape.get("som",{}).get("value","N/A")} | Demand: {bundle.get("demand_signals",{}).get("current_demand","N/A")}
Customer: {bundle.get("customer_profile",{}).get("primary_persona","N/A")}

Return ONLY this JSON:
{{
  "gtm_strategy": {{
    "recommended_approach": "e.g. PLG/Direct Sales/Channel",
    "primary_channel": "one line",
    "beachhead_segment": "one line",
    "beachhead_rationale": "one line"
  }},
  "market_entry_plan": ["step1","step2","step3"],
  "expansion_roadmap": [
    {{"phase": "0-6 months", "focus": "one line", "target_revenue": "$X"}},
    {{"phase": "6-18 months", "focus": "one line", "target_revenue": "$X"}},
    {{"phase": "18-36 months", "focus": "one line", "target_revenue": "$X"}}
  ],
  "key_risks": ["risk1","risk2"],
  "success_metrics": ["metric1","metric2","metric3"],
  "verdict": "one bold sentence"
}}
"""
    return _call_agent_step(prompt, "market_strategy")


# ----------------------------- Main Agent Entry Point ----------------------------- #

def run_market_analysis_agent(startup_profile: dict) -> dict:
    """
    Run the 3-step Market Analysis Agent with Redis caching.

    Cache hit  → returns instantly from Redis (~50ms)
    Cache miss → runs 3 Gemini calls (~10-15s), then caches result

    Args:
        startup_profile: dict with company_name, industry, niche,
                         target_market, region, stage, product_description

    Returns:
        Full market intelligence report.
    """
    cache_key = _make_cache_key(startup_profile)

    # ── Cache Check ──
    cached = _cache_get(cache_key)
    if cached:
        cached["cache_hit"] = True
        return cached

    print(f"\n[market_agent] 🚀 Starting analysis for: {startup_profile.get('company_name', 'Unknown')}")
    report = {
        "agent_steps_completed": 0,
        "cache_hit": False,
        "startup": startup_profile.get("company_name", "Unknown"),
        "industry": startup_profile.get("industry", "N/A"),
    }

    # ── Step 1: Market Landscape ──
    print("[market_agent] Step 1/3 — Market Landscape")
    landscape = _step_market_landscape(startup_profile)
    if not landscape:
        report["error"] = "Agent failed at Step 1 (Market Landscape). Check API key and quota."
        return report

    report["tam"] = landscape.get("tam")
    report["sam"] = landscape.get("sam")
    report["som"] = landscape.get("som")
    report["cagr"] = landscape.get("cagr")
    report["market_maturity"] = landscape.get("market_maturity")
    report["key_segments"] = landscape.get("key_segments", [])
    report["top_trends"] = landscape.get("top_trends", [])
    report["market_summary"] = landscape.get("market_summary")
    report["agent_steps_completed"] = 1

    # ── Step 2: Intelligence Bundle ──
    print("[market_agent] Step 2/3 — Intelligence Bundle")
    bundle = _step_intelligence_bundle(startup_profile, landscape)
    if not bundle:
        report["warning"] = "Agent stopped at Step 2 (Intelligence Bundle). Partial results returned."
        return report

    report["entry_barriers"] = bundle.get("entry_barriers", [])
    report["customer_profile"] = bundle.get("customer_profile", {})
    report["demand_signals"] = bundle.get("demand_signals", {})
    report["agent_steps_completed"] = 2

    # ── Step 3: Market Strategy ──
    print("[market_agent] Step 3/3 — Market Strategy")
    strategy = _step_market_strategy(startup_profile, landscape, bundle)
    if not strategy:
        report["warning"] = "Agent stopped at Step 3 (Market Strategy). Partial results returned."
        return report

    report["gtm_strategy"] = strategy.get("gtm_strategy", {})
    report["market_entry_plan"] = strategy.get("market_entry_plan", [])
    report["expansion_roadmap"] = strategy.get("expansion_roadmap", [])
    report["key_risks"] = strategy.get("key_risks", [])
    report["success_metrics"] = strategy.get("success_metrics", [])
    report["verdict"] = strategy.get("verdict")
    report["agent_steps_completed"] = 3
    print(f"[market_agent] ✅ Complete for: {startup_profile.get('company_name', 'Unknown')}")

    # ── Cache the result ──
    _cache_set(cache_key, report)

    return report
