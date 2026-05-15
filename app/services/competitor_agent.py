from __future__ import annotations
"""
Competitor Analysis Agent

A multi-step agentic service that reasons through competitive landscape analysis
in structured phases — just like a real analyst would think through the problem.

Agent Loop:
  Step 1 — Market Scan       : Identify top competitors in the space
  Step 2 — Deep Dive         : Analyze each competitor's strengths, weaknesses, funding
  Step 3 — Positioning Map   : Map where the startup sits vs competitors
  Step 4 — Gap Intelligence  : Surface exploitable gaps and blind spots
  Step 5 — Battle Plan       : Generate a concrete competitive strategy

Steps 2, 3, 4 run in parallel after Step 1 — then Step 5 synthesizes everything.
"""

import os
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from google import genai


# ----------------------------- Shared Helpers ----------------------------- #

_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-001",
]


def _get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _call_agent_step(prompt: str, step_name: str) -> Optional[dict]:
    """
    Execute a single agent step — calls Gemini with fallback and retry.
    Returns parsed JSON dict or None on failure.
    """
    client = _get_client()
    if not client:
        print(f"[competitor_agent:{step_name}] No API key configured.")
        return None

    env_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    models_to_try = [env_model] + [m for m in _FALLBACK_MODELS if m != env_model]

    last_error = None
    for model in models_to_try:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
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
    """
    Step 1 — Scan the market and identify the top 5 real competitors.
    """
    prompt = f"""
You are a competitive intelligence analyst. Your task is to identify the top 5 real-world
competitors for a startup based on its profile.

Startup Profile:
- Company Name: {startup.get("company_name", "Unnamed Startup")}
- Industry: {startup.get("industry", "N/A")}
- Sub-sector / Niche: {startup.get("niche", "N/A")}
- Target Market: {startup.get("target_market", "N/A")}
- Region: {startup.get("region", "Global")}
- Stage: {startup.get("stage", "N/A")}
- Core Product/Service: {startup.get("product_description", "N/A")}

Identify the 5 most relevant competitors this startup would face TODAY.
Be specific — use real company names. Avoid generic placeholders.

Return ONLY this JSON, no markdown:
{{
  "competitors": [
    {{
      "name": "Competitor name",
      "hq": "City, Country",
      "founded": "Year",
      "stage": "Series X / Public / etc",
      "core_product": "One line description",
      "why_relevant": "Why this is a direct competitor"
    }}
  ],
  "market_summary": "2-sentence overview of the competitive landscape"
}}
"""
    return _call_agent_step(prompt, "market_scan")


def _step_deep_dive(startup: dict, competitors: list) -> Optional[dict]:
    """
    Step 2 — Deep dive into each competitor's strengths, weaknesses, and funding.
    """
    comp_list = "\n".join([
        f"- {c.get('name')} ({c.get('stage', 'N/A')}) — {c.get('core_product', 'N/A')}"
        for c in competitors
    ])

    prompt = f"""
You are a senior startup analyst performing a deep competitive analysis.

Target Startup:
- Name: {startup.get("company_name", "Unnamed Startup")}
- Industry: {startup.get("industry", "N/A")}
- Product: {startup.get("product_description", "N/A")}

Competitors to analyze:
{comp_list}

For each competitor, provide a detailed breakdown.
Be honest, specific, and data-grounded. Avoid vague statements.

Return ONLY this JSON, no markdown:
{{
  "analysis": [
    {{
      "name": "Competitor name",
      "strengths": ["strength 1", "strength 2", "strength 3"],
      "weaknesses": ["weakness 1", "weakness 2"],
      "estimated_funding": "e.g. $50M Series B",
      "market_share_signal": "dominant / growing / niche / declining",
      "key_differentiator": "What makes them stand out",
      "threat_level": "High / Medium / Low",
      "threat_reason": "Why this threat level"
    }}
  ]
}}
"""
    return _call_agent_step(prompt, "deep_dive")


def _step_positioning_map(startup: dict, competitor_analysis: list) -> Optional[dict]:
    """
    Step 3 — Map the startup's position relative to competitors.
    """
    comp_summary = "\n".join([
        f"- {c.get('name')}: threat={c.get('threat_level')} | differentiator={c.get('key_differentiator')}"
        for c in competitor_analysis
    ])

    prompt = f"""
You are a market positioning strategist. Based on the competitive analysis below,
map where the startup currently sits in the market and where it should position itself.

Startup:
- Name: {startup.get("company_name", "Unnamed Startup")}
- Product: {startup.get("product_description", "N/A")}
- USP (if any): {startup.get("usp", "Not specified")}
- Target Market: {startup.get("target_market", "N/A")}

Competitor Landscape:
{comp_summary}

Provide a clear positioning assessment.

Return ONLY this JSON, no markdown:
{{
  "current_position": "Where the startup sits today in the market",
  "positioning_archetype": "e.g. Challenger / Niche Disruptor / Fast Follower / Category Creator",
  "differentiation_score": "1-10 rating of how differentiated the startup is",
  "differentiation_rationale": "Why this score",
  "white_spaces": ["underserved segment 1", "underserved segment 2"],
  "positioning_statement": "A crisp 1-sentence positioning statement for the startup"
}}
"""
    return _call_agent_step(prompt, "positioning_map")


def _step_gap_intelligence(startup: dict, competitor_analysis: list, positioning: dict) -> Optional[dict]:
    """
    Step 4 — Surface exploitable gaps and competitor blind spots.
    """
    weaknesses = []
    for c in competitor_analysis:
        for w in c.get("weaknesses", []):
            weaknesses.append(f"- {c.get('name')}: {w}")
    weakness_text = "\n".join(weaknesses) if weaknesses else "No weaknesses identified."

    white_spaces = positioning.get("white_spaces", []) if positioning else []
    white_space_text = "\n".join([f"- {w}" for w in white_spaces]) if white_spaces else "None identified."

    prompt = f"""
You are a competitive intelligence specialist hunting for exploitable gaps.

Startup:
- Name: {startup.get("company_name", "Unnamed Startup")}
- Industry: {startup.get("industry", "N/A")}
- Stage: {startup.get("stage", "N/A")}

Competitor Weaknesses Identified:
{weakness_text}

Market White Spaces:
{white_space_text}

Based on this intelligence, identify the most actionable gaps the startup can exploit
RIGHT NOW to gain competitive advantage. Be ruthlessly specific.

Return ONLY this JSON, no markdown:
{{
  "exploitable_gaps": [
    {{
      "gap": "Description of the gap",
      "source": "Which competitor's weakness or market void",
      "opportunity_size": "Large / Medium / Small",
      "time_to_exploit": "Immediate / 3-6 months / 6-12 months",
      "action": "Specific action to exploit this gap"
    }}
  ],
  "biggest_opportunity": "The single most impactful gap to go after first",
  "risk_of_inaction": "What happens if the startup ignores these gaps"
}}
"""
    return _call_agent_step(prompt, "gap_intelligence")


def _step_battle_plan(startup: dict, all_context: dict) -> Optional[dict]:
    """
    Step 5 — Synthesize everything into a concrete competitive battle plan.
    """
    top_threats = [
        c.get("name") for c in all_context.get("competitor_analysis", [])
        if c.get("threat_level") == "High"
    ]
    top_gaps = [
        g.get("gap") for g in all_context.get("gap_intelligence", {}).get("exploitable_gaps", [])
    ][:3]

    prompt = f"""
You are a startup strategy advisor delivering the final competitive battle plan.
Synthesize all intelligence gathered into a concrete, prioritized action plan.

Startup:
- Name: {startup.get("company_name", "Unnamed Startup")}
- Industry: {startup.get("industry", "N/A")}
- Stage: {startup.get("stage", "N/A")}
- Region: {startup.get("region", "Global")}

High-Threat Competitors: {", ".join(top_threats) if top_threats else "None identified"}
Top Exploitable Gaps: {", ".join(top_gaps) if top_gaps else "None identified"}
Positioning: {all_context.get("positioning", {}).get("positioning_archetype", "N/A")}
Differentiation Score: {all_context.get("positioning", {}).get("differentiation_score", "N/A")}/10

Deliver a battle plan that is bold, specific, and immediately actionable.
No fluff. No generic advice. Real moves.

Return ONLY this JSON, no markdown:
{{
  "executive_summary": "3-sentence competitive situation summary",
  "immediate_moves": ["action to take this week 1", "action to take this week 2", "action to take this week 3"],
  "30_day_plan": ["milestone 1", "milestone 2", "milestone 3"],
  "90_day_plan": ["strategic goal 1", "strategic goal 2"],
  "defend_against": [
    {{
      "competitor": "name",
      "defense_tactic": "how to neutralize their threat"
    }}
  ],
  "win_condition": "What does winning look like in 12 months for this startup"
}}
"""
    return _call_agent_step(prompt, "battle_plan")


# ----------------------------- Main Agent Entry Point ----------------------------- #

def run_competitor_analysis_agent(startup_profile: dict) -> dict:
    """
    Run the full 5-step Competitor Analysis Agent.

    Step 1 runs first (market scan).
    Steps 2, 3, 4 run in PARALLEL after Step 1 for speed.
    Step 5 synthesizes everything.

    Partial results are returned even if later steps fail.

    Args:
        startup_profile: Startup info dict with keys:
            company_name, industry, niche, target_market,
            region, stage, product_description, usp

    Returns:
        Full competitive intelligence report with all 5 agent steps.
    """
    print(f"\n[competitor_agent] 🚀 Starting analysis for: {startup_profile.get('company_name', 'Unknown')}")
    report = {
        "agent_steps_completed": 0,
        "startup": startup_profile.get("company_name", "Unknown"),
        "industry": startup_profile.get("industry", "N/A"),
    }

    # ── Step 1: Market Scan (must run first — others depend on it) ──
    print("[competitor_agent] Step 1/5 — Market Scan")
    scan = _step_market_scan(startup_profile)
    if not scan:
        report["error"] = "Agent failed at Step 1 (Market Scan). Check API key and quota."
        return report

    report["market_summary"] = scan.get("market_summary")
    report["competitors_identified"] = scan.get("competitors", [])
    report["agent_steps_completed"] = 1
    time.sleep(1)  # brief buffer before parallel burst

    # ── Steps 2, 3, 4: Run in PARALLEL ──
    print("[competitor_agent] Steps 2-4 — Running in parallel...")
    competitors = scan.get("competitors", [])

    dive_result = None
    positioning_result = None
    gaps_result = None

    def run_deep_dive():
        return _step_deep_dive(startup_profile, competitors)

    def run_positioning():
        # Positioning uses competitor names from scan — available now
        comp_summary = [
            {"name": c.get("name"), "key_differentiator": c.get("why_relevant", "N/A"), "threat_level": "Unknown"}
            for c in competitors
        ]
        return _step_positioning_map(startup_profile, comp_summary)

    def run_gaps():
        # Gap intelligence uses white spaces — run independently with scan context
        weaknesses_placeholder = []
        white_spaces_placeholder = {"white_spaces": []}
        return _step_gap_intelligence(startup_profile, weaknesses_placeholder, white_spaces_placeholder)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_deep_dive): "deep_dive",
            executor.submit(run_positioning): "positioning",
            executor.submit(run_gaps): "gaps_initial",
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                result = future.result()
                if label == "deep_dive":
                    dive_result = result
                elif label == "positioning":
                    positioning_result = result
                elif label == "gaps_initial":
                    gaps_result = result
            except Exception as e:
                print(f"[competitor_agent] Parallel step {label} raised: {e}")

    if not dive_result:
        report["warning"] = "Agent stopped at Step 2 (Deep Dive). Partial results returned."
        return report
    report["competitor_analysis"] = dive_result.get("analysis", [])
    report["agent_steps_completed"] = 2

    if not positioning_result:
        report["warning"] = "Agent stopped at Step 3 (Positioning). Partial results returned."
        return report
    report["positioning"] = positioning_result
    report["agent_steps_completed"] = 3

    if not gaps_result:
        report["warning"] = "Agent stopped at Step 4 (Gap Intelligence). Partial results returned."
        return report

    # Enrich gaps with actual deep dive weaknesses now that we have them
    enriched_gaps = _step_gap_intelligence(
        startup_profile,
        dive_result.get("analysis", []),
        positioning_result,
    )
    report["gap_intelligence"] = enriched_gaps or gaps_result
    report["agent_steps_completed"] = 4
    time.sleep(1)

    # ── Step 5: Battle Plan (synthesizes everything) ──
    print("[competitor_agent] Step 5/5 — Battle Plan")
    battle_plan = _step_battle_plan(startup_profile, {
        "competitor_analysis": dive_result.get("analysis", []),
        "positioning": positioning_result,
        "gap_intelligence": report["gap_intelligence"],
    })
    if not battle_plan:
        report["warning"] = "Agent stopped at Step 5 (Battle Plan). Partial results returned."
        return report

    report["battle_plan"] = battle_plan
    report["agent_steps_completed"] = 5
    print(f"[competitor_agent] ✅ All 5 steps complete for: {startup_profile.get('company_name', 'Unknown')}")

    return report
