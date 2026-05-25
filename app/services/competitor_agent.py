from __future__ import annotations
"""
Competitor Analysis Agent

A 3-step agentic service with Redis caching for fast repeated lookups.

Agent Loop:
  Step 1 — Market Scan         : Identify top competitors in the space
  Step 2 — Intelligence Bundle : Deep dive + positioning + gap analysis in one call
  Step 3 — Battle Plan         : Synthesize everything into a concrete strategy

Redis Cache:
  - Cache key: SHA256 hash of normalized startup profile
  - TTL: 24 hours (configurable via COMPETITOR_CACHE_TTL env var)
  - Cache hit  → ~50ms response
  - Cache miss → ~10-12s (3 Gemini calls)
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
    print("[competitor_agent] ✅ Redis connected")
except Exception as e:
    _redis_client = None
    REDIS_AVAILABLE = False
    print(f"[competitor_agent] ⚠️ Redis unavailable, using in-memory cache: {e}")

# In-memory fallback cache when Redis is unavailable
_memory_cache: dict = {}

# Cache TTL in seconds — default 24 hours
_CACHE_TTL = int(os.getenv("COMPETITOR_CACHE_TTL", 86400))

# Gemini model fallback list
_FALLBACK_MODELS = ["gemini-2.5-flash"]


# ----------------------------- Cache Helpers ----------------------------- #

def _make_cache_key(startup_profile: dict) -> str:
    """Generate a stable cache key from the startup profile."""
    normalized = json.dumps(startup_profile, sort_keys=True, default=str).lower()
    return "competitor:" + hashlib.sha256(normalized.encode()).hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    if REDIS_AVAILABLE:
        try:
            val = _redis_client.get(key)
            if val:
                print(f"[competitor_agent] 🎯 Cache HIT (Redis): {key[:30]}...")
                return json.loads(val)
        except Exception as e:
            print(f"[competitor_agent] Cache get error: {e}")
    # Fallback to in-memory
    if key in _memory_cache:
        print(f"[competitor_agent] 🎯 Cache HIT (memory): {key[:30]}...")
        return _memory_cache[key]
    return None


def _cache_set(key: str, value: dict) -> None:
    if REDIS_AVAILABLE:
        try:
            _redis_client.setex(key, _CACHE_TTL, json.dumps(value, default=str))
            print(f"[competitor_agent] 💾 Cached in Redis for {_CACHE_TTL}s")
            return
        except Exception as e:
            print(f"[competitor_agent] Cache set error: {e}")
    # Fallback to in-memory
    _memory_cache[key] = value
    print(f"[competitor_agent] 💾 Cached in memory")


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
        print(f"[competitor_agent:{step_name}] No API key configured.")
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
                        max_output_tokens=2048,
                        temperature=0.3,
                    ),
                )
                text = response.text
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if not match:
                    print(f"[competitor_agent:{step_name}] No JSON in response from {model}")
                    break
                print(f"[competitor_agent:{step_name}] ✓ {model}")
                return json.loads(match.group())

            except json.JSONDecodeError as e:
                print(f"[competitor_agent:{step_name}] JSON parse error with {model}: {e}")
                last_error = str(e)
                break
            except Exception as e:
                err_str = str(e)
                last_error = err_str
                # Daily quota exhausted — skip this model immediately, no retry
                if "429" in err_str and "PerDay" in err_str:
                    print(f"[competitor_agent:{step_name}] {model} daily quota exhausted, skipping")
                    break
                # Any 429 — skip immediately, don't retry
                if "429" in err_str:
                    print(f"[competitor_agent:{step_name}] {model} rate limited, skipping")
                    break
                # Overloaded — retry once
                if "503" in err_str and attempt == 0:
                    print(f"[competitor_agent:{step_name}] {model} overloaded, retrying in 3s...")
                    time.sleep(3)
                    continue
                print(f"[competitor_agent:{step_name}] {model} failed: {e}")
                break

    print(f"[competitor_agent:{step_name}] All models exhausted. Last: {last_error}")
    return None


# ----------------------------- Agent Steps ----------------------------- #

def _step_market_scan(startup: dict) -> Optional[dict]:
    """Step 1 — Identify top 5 real competitors."""
    prompt = f"""
Competitive intelligence analyst. Identify top 5 real competitors for this startup.

Startup: {startup.get("company_name", "Unnamed")} | Industry: {startup.get("industry", "N/A")} | Niche: {startup.get("niche", "N/A")} | Region: {startup.get("region", "Global")} | Product: {startup.get("product_description", "N/A")}

Use real company names. Be concise.

Return ONLY this JSON:
{{
  "competitors": [
    {{"name": "Name", "hq": "City, Country", "stage": "Series/Public", "core_product": "one line", "why_relevant": "one line"}}
  ],
  "market_summary": "2-sentence overview"
}}
"""
    return _call_agent_step(prompt, "market_scan")


def _step_intelligence_bundle(startup: dict, competitors: list) -> Optional[dict]:
    """Step 2 — Deep dive + positioning + gaps in one call. Concise output."""
    comp_list = " | ".join([f"{c.get('name')} ({c.get('stage','N/A')})" for c in competitors])

    prompt = f"""
Senior competitive analyst. Analyze this startup vs its competitors. Be concise and specific.

Startup: {startup.get("company_name","Unnamed")} | {startup.get("industry","N/A")} | {startup.get("product_description","N/A")} | USP: {startup.get("usp","N/A")} | Stage: {startup.get("stage","N/A")}
Competitors: {comp_list}

Return ONLY this JSON:
{{
  "competitor_analysis": [
    {{"name": "Name", "strengths": ["s1","s2"], "weaknesses": ["w1","w2"], "estimated_funding": "e.g. $50M", "threat_level": "High/Medium/Low", "key_differentiator": "one line"}}
  ],
  "positioning": {{
    "current_position": "one line",
    "positioning_archetype": "e.g. Challenger",
    "differentiation_score": "1-10",
    "white_spaces": ["gap1","gap2"],
    "positioning_statement": "one sentence"
  }},
  "gap_intelligence": {{
    "exploitable_gaps": [
      {{"gap": "description", "source": "competitor name", "opportunity_size": "Large/Medium/Small", "action": "specific action"}}
    ],
    "biggest_opportunity": "one line"
  }}
}}
"""
    return _call_agent_step(prompt, "intelligence_bundle")


def _step_battle_plan(startup: dict, bundle: dict) -> Optional[dict]:
    """Step 3 — Concise battle plan."""
    analysis = bundle.get("competitor_analysis", [])
    positioning = bundle.get("positioning", {})
    gaps = bundle.get("gap_intelligence", {})

    top_threats = [c.get("name") for c in analysis if c.get("threat_level") == "High"]
    top_gap = gaps.get("biggest_opportunity", "N/A")

    prompt = f"""
Startup strategy advisor. Deliver a concise competitive battle plan.

Startup: {startup.get("company_name","Unnamed")} | {startup.get("industry","N/A")} | Stage: {startup.get("stage","N/A")}
High threats: {", ".join(top_threats) if top_threats else "None"}
Biggest opportunity: {top_gap}
Positioning: {positioning.get("positioning_archetype","N/A")} | Score: {positioning.get("differentiation_score","N/A")}/10

Return ONLY this JSON:
{{
  "executive_summary": "2-3 sentences",
  "immediate_moves": ["action1","action2","action3"],
  "30_day_plan": ["milestone1","milestone2"],
  "90_day_plan": ["goal1","goal2"],
  "defend_against": [{{"competitor": "name", "defense_tactic": "one line"}}],
  "win_condition": "one sentence"
}}
"""
    return _call_agent_step(prompt, "battle_plan")


# ----------------------------- Main Agent Entry Point ----------------------------- #

def run_competitor_analysis_agent(startup_profile: dict) -> dict:
    """
    Run the 3-step Competitor Analysis Agent with Redis caching.

    Cache hit  → returns instantly from Redis (~50ms)
    Cache miss → runs 3 Gemini calls (~10-12s), then caches result

    Args:
        startup_profile: dict with company_name, industry, niche,
                         target_market, region, stage, product_description, usp

    Returns:
        Full competitive intelligence report.
    """
    cache_key = _make_cache_key(startup_profile)

    # ── Cache Check ──
    cached = _cache_get(cache_key)
    if cached:
        cached["cache_hit"] = True
        return cached

    print(f"\n[competitor_agent] 🚀 Starting analysis for: {startup_profile.get('company_name', 'Unknown')}")
    report = {
        "agent_steps_completed": 0,
        "cache_hit": False,
        "startup": startup_profile.get("company_name", "Unknown"),
        "industry": startup_profile.get("industry", "N/A"),
    }

    # ── Step 1: Market Scan ──
    print("[competitor_agent] Step 1/3 — Market Scan")
    scan = _step_market_scan(startup_profile)
    if not scan:
        report["error"] = "Agent failed at Step 1 (Market Scan). Check API key and quota."
        return report

    report["market_summary"] = scan.get("market_summary")
    report["competitors_identified"] = scan.get("competitors", [])
    report["agent_steps_completed"] = 1

    # ── Step 2: Intelligence Bundle (deep dive + positioning + gaps) ──
    print("[competitor_agent] Step 2/3 — Intelligence Bundle")
    bundle = _step_intelligence_bundle(startup_profile, scan.get("competitors", []))
    if not bundle:
        report["warning"] = "Agent stopped at Step 2 (Intelligence Bundle). Partial results returned."
        return report

    report["competitor_analysis"] = bundle.get("competitor_analysis", [])
    report["positioning"] = bundle.get("positioning", {})
    report["gap_intelligence"] = bundle.get("gap_intelligence", {})
    report["agent_steps_completed"] = 2

    # ── Step 3: Battle Plan ──
    print("[competitor_agent] Step 3/3 — Battle Plan")
    battle_plan = _step_battle_plan(startup_profile, bundle)
    if not battle_plan:
        report["warning"] = "Agent stopped at Step 3 (Battle Plan). Partial results returned."
        return report

    report["battle_plan"] = battle_plan
    report["agent_steps_completed"] = 3
    print(f"[competitor_agent] ✅ Complete for: {startup_profile.get('company_name', 'Unknown')}")

    # ── Cache the result ──
    _cache_set(cache_key, report)

    return report
