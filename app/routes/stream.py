from __future__ import annotations
"""
SSE Streaming Routes for Competitor and Market Analysis Agents

Instead of waiting 10-15s for the full response, the client receives
each agent step as it completes — feels instant from the user's perspective.

NOTE: These are POST endpoints (not GET). Use fetch() with a ReadableStream
reader on the frontend — NOT EventSource (which only supports GET).

Frontend usage example:
  const response = await fetch('/api/stream/competitor/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(startupProfile)
  });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const lines = decoder.decode(value).split('\\n');
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const { step, data, done } = JSON.parse(line.slice(6));
        // render step data progressively
      }
    }
  }
"""

import json
from flask import Blueprint, request, jsonify, Response, stream_with_context

from app.services.competitor_agent import (
    _step_market_scan,
    _step_intelligence_bundle,
    _step_battle_plan,
    _make_cache_key as _competitor_cache_key,
    _cache_get as _competitor_cache_get,
    _cache_set as _competitor_cache_set,
)
from app.services.market_agent import (
    _step_market_landscape,
    _step_intelligence_bundle as _market_intelligence_bundle,
    _step_market_strategy,
    _make_cache_key as _market_cache_key,
    _cache_get as _market_cache_get,
    _cache_set as _market_cache_set,
)

bp = Blueprint("stream", __name__)


def _sse(data: dict) -> str:
    """Format a dict as an SSE message."""
    return f"data: {json.dumps(data, default=str)}\n\n"


# ─────────────────────────────────────────────────────────────
# Competitor Analysis — SSE Stream
# ─────────────────────────────────────────────────────────────

@bp.post("/stream/competitor/analyze")
def stream_competitor_analyze():
    """
    POST /api/stream/competitor/analyze

    Same input as /api/competitor/analyze.
    Streams each step as SSE events — client sees results progressively.

    Event format:
      data: {"step": 1, "label": "Market Scan", "data": {...}, "done": false}
      data: {"step": 2, "label": "Intelligence Bundle", "data": {...}, "done": false}
      data: {"step": 3, "label": "Battle Plan", "data": {...}, "done": true}

    On cache hit:
      data: {"cache_hit": true, "data": {...}, "done": true}
    """
    data = request.get_json(force=True, silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Request body is required"}), 400
    if not data.get("industry") or not data.get("product_description"):
        return jsonify({"error": "industry and product_description are required"}), 400

    def generate():
        # Cache check
        cache_key = _competitor_cache_key(data)
        cached = _competitor_cache_get(cache_key)
        if cached:
            cached["cache_hit"] = True
            yield _sse({"cache_hit": True, "data": cached, "done": True})
            return

        report = {
            "cache_hit": False,
            "startup": data.get("company_name", "Unknown"),
            "industry": data.get("industry", "N/A"),
            "agent_steps_completed": 0,
        }

        # Step 1
        scan = _step_market_scan(data)
        if not scan:
            yield _sse({"error": "Step 1 failed", "done": True})
            return
        report["market_summary"] = scan.get("market_summary")
        report["competitors_identified"] = scan.get("competitors", [])
        report["agent_steps_completed"] = 1
        yield _sse({"step": 1, "label": "Market Scan", "data": {
            "market_summary": scan.get("market_summary"),
            "competitors_identified": scan.get("competitors", []),
        }, "done": False})

        # Step 2
        bundle = _step_intelligence_bundle(data, scan.get("competitors", []))
        if not bundle:
            yield _sse({"error": "Step 2 failed", "partial": report, "done": True})
            return
        report["competitor_analysis"] = bundle.get("competitor_analysis", [])
        report["positioning"] = bundle.get("positioning", {})
        report["gap_intelligence"] = bundle.get("gap_intelligence", {})
        report["agent_steps_completed"] = 2
        yield _sse({"step": 2, "label": "Intelligence Bundle", "data": {
            "competitor_analysis": bundle.get("competitor_analysis", []),
            "positioning": bundle.get("positioning", {}),
            "gap_intelligence": bundle.get("gap_intelligence", {}),
        }, "done": False})

        # Step 3
        battle_plan = _step_battle_plan(data, bundle)
        if not battle_plan:
            yield _sse({"error": "Step 3 failed", "partial": report, "done": True})
            return
        report["battle_plan"] = battle_plan
        report["agent_steps_completed"] = 3
        yield _sse({"step": 3, "label": "Battle Plan", "data": {
            "battle_plan": battle_plan,
        }, "done": True})

        # Cache full result
        _competitor_cache_set(cache_key, report)

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# ─────────────────────────────────────────────────────────────
# Market Analysis — SSE Stream
# ─────────────────────────────────────────────────────────────

@bp.post("/stream/market/analyze")
def stream_market_analyze():
    """
    POST /api/stream/market/analyze

    Same input as /api/market/analyze.
    Streams each step as SSE events — client sees results progressively.

    Event format:
      data: {"step": 1, "label": "Market Landscape", "data": {...}, "done": false}
      data: {"step": 2, "label": "Intelligence Bundle", "data": {...}, "done": false}
      data: {"step": 3, "label": "Market Strategy", "data": {...}, "done": true}

    On cache hit:
      data: {"cache_hit": true, "data": {...}, "done": true}
    """
    data = request.get_json(force=True, silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Request body is required"}), 400
    if not data.get("industry") or not data.get("product_description"):
        return jsonify({"error": "industry and product_description are required"}), 400

    def generate():
        # Cache check
        cache_key = _market_cache_key(data)
        cached = _market_cache_get(cache_key)
        if cached:
            cached["cache_hit"] = True
            yield _sse({"cache_hit": True, "data": cached, "done": True})
            return

        report = {
            "cache_hit": False,
            "startup": data.get("company_name", "Unknown"),
            "industry": data.get("industry", "N/A"),
            "agent_steps_completed": 0,
        }

        # Step 1
        landscape = _step_market_landscape(data)
        if not landscape:
            yield _sse({"error": "Step 1 failed", "done": True})
            return
        report.update({
            "tam": landscape.get("tam"),
            "sam": landscape.get("sam"),
            "som": landscape.get("som"),
            "cagr": landscape.get("cagr"),
            "market_maturity": landscape.get("market_maturity"),
            "key_segments": landscape.get("key_segments", []),
            "top_trends": landscape.get("top_trends", []),
            "market_summary": landscape.get("market_summary"),
            "agent_steps_completed": 1,
        })
        yield _sse({"step": 1, "label": "Market Landscape", "data": {
            "tam": landscape.get("tam"),
            "sam": landscape.get("sam"),
            "som": landscape.get("som"),
            "cagr": landscape.get("cagr"),
            "market_maturity": landscape.get("market_maturity"),
            "key_segments": landscape.get("key_segments", []),
            "top_trends": landscape.get("top_trends", []),
            "market_summary": landscape.get("market_summary"),
        }, "done": False})

        # Step 2
        bundle = _market_intelligence_bundle(data, landscape)
        if not bundle:
            yield _sse({"error": "Step 2 failed", "partial": report, "done": True})
            return
        report.update({
            "entry_barriers": bundle.get("entry_barriers", []),
            "customer_profile": bundle.get("customer_profile", {}),
            "demand_signals": bundle.get("demand_signals", {}),
            "agent_steps_completed": 2,
        })
        yield _sse({"step": 2, "label": "Intelligence Bundle", "data": {
            "entry_barriers": bundle.get("entry_barriers", []),
            "customer_profile": bundle.get("customer_profile", {}),
            "demand_signals": bundle.get("demand_signals", {}),
        }, "done": False})

        # Step 3
        strategy = _step_market_strategy(data, landscape, bundle)
        if not strategy:
            yield _sse({"error": "Step 3 failed", "partial": report, "done": True})
            return
        report.update({
            "gtm_strategy": strategy.get("gtm_strategy", {}),
            "market_entry_plan": strategy.get("market_entry_plan", []),
            "expansion_roadmap": strategy.get("expansion_roadmap", []),
            "key_risks": strategy.get("key_risks", []),
            "success_metrics": strategy.get("success_metrics", []),
            "verdict": strategy.get("verdict"),
            "agent_steps_completed": 3,
        })
        yield _sse({"step": 3, "label": "Market Strategy", "data": {
            "gtm_strategy": strategy.get("gtm_strategy", {}),
            "market_entry_plan": strategy.get("market_entry_plan", []),
            "expansion_roadmap": strategy.get("expansion_roadmap", []),
            "key_risks": strategy.get("key_risks", []),
            "success_metrics": strategy.get("success_metrics", []),
            "verdict": strategy.get("verdict"),
        }, "done": True})

        # Cache full result
        _market_cache_set(cache_key, report)

    return Response(stream_with_context(generate()), mimetype="text/event-stream")
