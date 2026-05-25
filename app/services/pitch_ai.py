"""
Pitch Deck AI Content Generator Service

Uses Ollama Cloud API (online, no local install needed) to generate
professional investor pitch deck content for each slide section.
Requires: pip install ollama
API Key:  ollama.com/settings/keys -> set OLLAMA_API_KEY env var
"""

import os
import json
import re
from typing import Optional
from ollama import Client

# ── Ollama Cloud API config ───────────────────────────────────────────────────
client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
)
MODEL_NAME = "gpt-oss:20b-cloud"   # free model on Ollama Cloud

# ── Slide sections ────────────────────────────────────────────────────────────
SLIDE_SECTIONS = [
    'cover', 'problem', 'solution', 'market', 'product',
    'business_model', 'competition', 'team', 'traction', 'funding_needs'
]


# ── JSON cleaner ──────────────────────────────────────────────────────────────
def _clean_json(text: str) -> str:
    """
    Strip markdown fences and extract the first valid JSON object.
    Handles ```json blocks, preamble text, and trailing commentary.
    """
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else text


# ── Safe fallback ─────────────────────────────────────────────────────────────
def _fallback(section: str) -> dict:
    """Return a safe fallback slide when generation fails."""
    return {
        "title": section.replace('_', ' ').title(),
        "content": "Content generation failed. Please regenerate this slide.",
        "chart_data": None
    }


# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_prompt(section: str, context: dict, current_content: Optional[str]) -> str:
    """
    Builds a deeply descriptive, section-specific prompt for each pitch deck slide.
    Each section has its own detailed instructions, success criteria, and examples
    so the model produces consistent, investor-grade, structured JSON every time.
    """

    # ── Extract all context fields cleanly ──────────────────────────────────
    startup_name  = context.get('startupName',   'the startup')
    problem       = context.get('problem',        'not specified')
    solution      = context.get('solution',       'not specified')
    target_market = context.get('targetMarket',   'not specified')
    biz_model     = context.get('businessModel',  'not specified')
    team_info     = context.get('teamInfo',       'not specified')
    traction      = context.get('traction',       'early stage, no data yet')
    funding_needs = context.get('fundingNeeds',   'not specified')
    industry      = context.get('industry',       'technology')
    stage         = context.get('stage',          'early-stage')
    location      = context.get('location',       'not specified')
    current_year  = context.get('currentYear',    '2026')   # ← NEW

    # ── Shared system context injected into every prompt ─────────────────────
    system_context = f"""
You are a world-class startup pitch consultant with 20+ years experience
helping founders raise seed, Series A, and Series B rounds from top-tier VCs
including Sequoia, a16z, and Y Combinator.

You are building one slide of an investor pitch deck for:

  Startup Name   : {startup_name}
  Industry       : {industry}
  Stage          : {stage}
  Location       : {location}
  Current Year   : {current_year}
  Problem        : {problem}
  Solution       : {solution}
  Target Market  : {target_market}
  Business Model : {biz_model}
  Team Info      : {team_info}
  Traction       : {traction}
  Funding Needs  : {funding_needs}

ABSOLUTE OUTPUT RULES — violating any of these will cause system failure:
1. Return ONLY a single raw JSON object. Zero markdown. Zero explanation. Zero preamble.
2. Each bullet point must be on its own line inside the content string, separated by \\n.
3. Never use dashes (-), asterisks (*), or bullet symbols as prefixes. Plain text only.
4. Be specific and data-driven. Use real numbers, percentages, and market figures.
5. Never use vague buzzwords like "innovative", "cutting-edge", or "revolutionary".
6. Write like a top-tier McKinsey analyst — sharp, credible, direct, and concise.
7. Every sentence must earn its place. No filler. No padding.
8. All dates, quarters, and timeframes must be in {current_year} or later — never reference past dates.
"""

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 1 — COVER
    # ════════════════════════════════════════════════════════════════════════
    if section == 'cover':
        return system_context + f"""
SLIDE TYPE: Cover / Title Slide

YOUR MISSION:
Create one single, powerful tagline for {startup_name} that makes an investor
stop scrolling and want to read the next slide immediately.

WHAT MAKES A GREAT TAGLINE:
- It communicates the WHAT and the WHY in under 12 words
- It is bold, memorable, and investor-facing — not marketing copy
- It avoids all jargon and buzzwords
- It creates immediate clarity about who benefits and how
- Example of bad tagline: "Disrupting the future of enterprise AI solutions"
- Example of good tagline: "The operating system for startup funding decisions"

YOUR TASK:
Write one tagline for {startup_name} based on the context above.

Return ONLY this exact JSON:
{{"title": "{startup_name}", "content": "<one powerful tagline under 12 words>", "chart_data": null}}
"""

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 2 — PROBLEM
    # ════════════════════════════════════════════════════════════════════════
    if section == 'problem':
        return system_context + f"""
SLIDE TYPE: The Problem

YOUR MISSION:
Articulate the pain point {startup_name} solves so clearly and urgently that
investors immediately feel the weight of this problem and the cost of inaction.

WHAT MAKES A GREAT PROBLEM SLIDE:
- Opens with the human or business pain — make it visceral and relatable
- Escalates to scale — how many people or businesses suffer from this
- Includes at least one hard quantified stat (cost in $, time wasted, failure rate)
- Shows that existing solutions are inadequate — why the problem persists today
- Ends with a sense of urgency — why NOW is the right time to solve this

STRUCTURE — write exactly 4 bullet points:
  Bullet 1: The core human or business pain (specific and relatable)
  Bullet 2: The scale of the problem — how widespread and costly it is
  Bullet 3: Why existing solutions fail to solve it completely
  Bullet 4: The urgency — what market force or trend makes this critical right now

Each bullet: one sharp declarative sentence, maximum 18 words.
The reader must finish this slide thinking "this problem absolutely must be solved."

Return ONLY this exact JSON:
{{"title": "The Problem", "content": "<bullet1>\\n<bullet2>\\n<bullet3>\\n<bullet4>", "chart_data": null}}
"""

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 3 — SOLUTION
    # ════════════════════════════════════════════════════════════════════════
    if section == 'solution':
        return system_context + f"""
SLIDE TYPE: Our Solution

YOUR MISSION:
Present {startup_name}'s solution as the inevitable, elegant answer to the problem
just described. Show the mechanism — HOW it works — not just WHAT it is.

WHAT MAKES A GREAT SOLUTION SLIDE:
- Directly mirrors the problem bullets — each problem gets a solution
- Explains the core mechanism clearly (what actually happens under the hood)
- Shows a measurable, specific outcome for the user
- Highlights the key differentiator that makes this better than alternatives
- Demonstrates 10x improvement — faster, cheaper, more accurate, or more scalable

STRUCTURE — write exactly 4 bullet points:
  Bullet 1: Core mechanism — what {startup_name} does technically or operationally
  Bullet 2: Primary user benefit with a specific measurable outcome (%, $, time saved)
  Bullet 3: Key differentiator vs existing alternatives — what makes this unique
  Bullet 4: Why this approach is 10x better — the unfair advantage in the solution itself

Each bullet: one sharp sentence, maximum 18 words, no vague language.

Return ONLY this exact JSON:
{{"title": "Our Solution", "content": "<bullet1>\\n<bullet2>\\n<bullet3>\\n<bullet4>", "chart_data": null}}
"""

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 4 — MARKET
    # ════════════════════════════════════════════════════════════════════════
    if section == 'market':
        return system_context + f"""
SLIDE TYPE: Market Opportunity

YOUR MISSION:
Quantify the market opportunity to show investors the scale of the prize
that {startup_name} is going after. Make it feel both massive and believable.

WHAT MAKES A GREAT MARKET SLIDE:
- TAM: the entire global market for this problem category (realistic, cited mentally)
- SAM: the specific segment {startup_name} can realistically reach with its current model
- SOM: what {startup_name} can realistically capture in 3-5 years (conservative, credible)
- TAM must be larger than SAM, SAM larger than SOM — ratios must feel realistic
- The content sentences must explain WHY this market is growing NOW — macro tailwinds

MARKET SIZE RULES:
- TAM must be in range of $5B to $500B depending on industry
- SAM must be 10-30% of TAM
- SOM must be 5-15% of SAM
- All values in USD billions, as plain numbers (e.g. 120 means $120B)

CONTENT SENTENCES must cover:
- Current market size and annual growth rate (CAGR)
- The macro trend or regulatory shift driving growth
- Why {startup_name} is positioned to capture significant share

Return ONLY this exact JSON:
{{
  "title": "Market Opportunity",
  "content": "<2-3 sentences on market size, CAGR, and growth tailwinds>",
  "chart_data": {{
    "type": "pie",
    "title": "Market Size (USD Billions)",
    "labels": ["TAM", "SAM", "SOM"],
    "values": [<tam_number>, <sam_number>, <som_number>]
  }}
}}
"""

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 5 — PRODUCT
    # ════════════════════════════════════════════════════════════════════════
    if section == 'product':
        return system_context + f"""
SLIDE TYPE: Product

YOUR MISSION:
Describe {startup_name}'s product in a way that makes investors understand
both the depth of the technology and the simplicity of the user experience.

WHAT MAKES A GREAT PRODUCT SLIDE:
- Shows what the user actually does — the core interaction
- Highlights the key technical capability that enables the solution
- Demonstrates how the product gets smarter or better over time (data flywheel)
- Shows integration into the user's existing workflow — low friction adoption
- Ends with a clear output — what does the user have after using the product

STRUCTURE — write exactly 5 bullet points:
  Bullet 1: Core user experience — what does the user do first when they open the product
  Bullet 2: Key technical capability — the engine powering the product
  Bullet 3: Data or network advantage — how the product improves with usage or scale
  Bullet 4: Integration or ecosystem fit — how it plugs into existing tools or workflows
  Bullet 5: The output — what measurable result does the user achieve

Each bullet: feature or capability name followed by one crisp description, max 18 words.

Return ONLY this exact JSON:
{{"title": "Product", "content": "<bullet1>\\n<bullet2>\\n<bullet3>\\n<bullet4>\\n<bullet5>", "chart_data": null}}
"""

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 6 — BUSINESS MODEL
    # ════════════════════════════════════════════════════════════════════════
    if section == 'business_model':
        return system_context + f"""
SLIDE TYPE: Business Model

YOUR MISSION:
Show investors exactly how {startup_name} makes money — a clear, credible
path to revenue with strong unit economics and multiple growth levers.

WHAT MAKES A GREAT BUSINESS MODEL SLIDE:
- Primary revenue stream is crystal clear — pricing model, who pays, how often
- Secondary revenue stream shows diversification and expansion opportunity
- Unit economics give investors confidence — LTV:CAC ratio, gross margin, payback period
- Shows a natural expansion path — how revenue grows without proportional cost growth
- Revenue stream split chart must add up to 100%

STRUCTURE — write exactly 4 bullet points:
  Bullet 1: Primary revenue stream — pricing model (SaaS/transaction/licensing), price point, billing cadence
  Bullet 2: Secondary or upsell revenue stream — what additional value drives more revenue
  Bullet 3: Unit economics highlight — LTV, CAC, gross margin %, or months to payback
  Bullet 4: Revenue expansion lever — how revenue scales as the customer base grows

CHART — revenue stream percentage split must sum to exactly 100:
  Label each stream with a real business name (e.g. "SaaS Subscriptions", "API Usage", "Professional Services")

Return ONLY this exact JSON:
{{
  "title": "Business Model",
  "content": "<bullet1>\\n<bullet2>\\n<bullet3>\\n<bullet4>",
  "chart_data": {{
    "type": "pie",
    "title": "Revenue Stream Split",
    "labels": ["<Stream 1>", "<Stream 2>", "<Stream 3>"],
    "values": [<pct1>, <pct2>, <pct3>]
  }}
}}
"""

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 7 — COMPETITION
    # ════════════════════════════════════════════════════════════════════════
    if section == 'competition':
        return system_context + f"""
SLIDE TYPE: Competitive Landscape

YOUR MISSION:
Show investors that {startup_name} deeply understands its competitive landscape
and has a defensible, sustainable advantage that is hard for competitors to replicate.

WHAT MAKES A GREAT COMPETITION SLIDE:
- Names real, specific competitors — not vague categories
- Honestly acknowledges competitor strengths before pivoting to {startup_name}'s advantage
- Clearly articulates the PRIMARY unfair advantage (proprietary data, network effect, IP, speed, cost)
- Explains why existing solutions have a structural weakness — not just "we are better"
- Describes the moat — what makes {startup_name} increasingly hard to replicate over time

STRUCTURE — write exactly 4 bullet points:
  Bullet 1: Name 2-3 real key competitors and their single biggest weakness
  Bullet 2: {startup_name}'s primary unfair competitive advantage — be specific (data, network, IP, speed, cost)
  Bullet 3: Structural reason why existing solutions cannot fully solve the problem
  Bullet 4: The moat — what makes {startup_name} harder to copy as it grows (data flywheel, switching costs, network effects)

Each bullet: one specific, confident sentence. Maximum 18 words. No vague claims.

Return ONLY this exact JSON:
{{"title": "Competitive Landscape", "content": "<bullet1>\\n<bullet2>\\n<bullet3>\\n<bullet4>", "chart_data": null}}
"""

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 8 — TEAM
    # ════════════════════════════════════════════════════════════════════════
    if section == 'team':
        return system_context + f"""
SLIDE TYPE: Our Team

YOUR MISSION:
Convince investors that {startup_name}'s team is uniquely qualified to execute
this vision. Investors bet on teams first — the idea is secondary.

WHAT MAKES A GREAT TEAM SLIDE:
- Leads with the most impressive credential — prior exit, top-tier company, or domain breakthrough
- Shows domain expertise — why THIS team understands this problem better than anyone else
- Demonstrates technical depth — the CTO or lead engineer's most relevant achievement
- Includes advisors or backers if notable — name-drops matter to investors
- Shows the team has worked on this problem before — founder-market fit

STRUCTURE — write exactly 4 bullet points:
  Bullet 1: CEO/Founder — most impressive prior credential (exit, company, role, or research)
  Bullet 2: CTO or Technical Lead — key technical achievement most relevant to {startup_name}
  Bullet 3: Domain expertise proof — why THIS team has unique insight into this problem
  Bullet 4: Advisors, investors, or total combined years of relevant experience

Use {team_info} if provided. If not provided, generate credible realistic profiles
that fit a {stage} {industry} startup. Be specific — name roles and achievements.

Return ONLY this exact JSON:
{{"title": "Our Team", "content": "<bullet1>\\n<bullet2>\\n<bullet3>\\n<bullet4>", "chart_data": null}}
"""

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 9 — TRACTION
    # ════════════════════════════════════════════════════════════════════════
    if section == 'traction':
        return system_context + f"""
SLIDE TYPE: Traction & Growth

YOUR MISSION:
Show investors concrete proof that the market wants what {startup_name} is building.
Real traction de-risks the investment more than any pitch.

WHAT MAKES A GREAT TRACTION SLIDE:
- Uses the hardest, most impressive metrics available — revenue, paying customers, retention
- Shows a clear upward trajectory — investors look for slope, not just absolute numbers
- Includes a milestone that signals product-market fit (e.g. NPS > 50, net revenue retention > 110%)
- Monthly growth chart shows realistic but impressive momentum (20-40% MoM for early stage)
- Content sentences name specific milestones with dates or timeframes

CONTENT SENTENCES must cover (2-3 sentences):
  - Most impressive metric achieved to date (customers, revenue, GMV, or users)
  - A product-market fit signal (retention rate, NPS, renewal rate, or referral rate)
  - A recent milestone that validates the business (partnership, pilot, award, or waitlist size)

CHART — 6 months of growth data:
  - Values must show clear upward trend with realistic growth rates
  - For {stage} startup: start modest, grow 25-40% month over month
  - Use {traction} context if provided to make numbers credible

DATE RULE: All milestone dates and timeframes must use {current_year} or later — never past dates.

Return ONLY this exact JSON:
{{
  "title": "Traction & Growth",
  "content": "<2-3 sentences on key milestones and proof points>",
  "chart_data": {{
    "type": "bar",
    "title": "Monthly Growth",
    "labels": ["Month 1", "Month 2", "Month 3", "Month 4", "Month 5", "Month 6"],
    "values": [<v1>, <v2>, <v3>, <v4>, <v5>, <v6>]
  }}
}}
"""

    # ════════════════════════════════════════════════════════════════════════
    # SLIDE 10 — FUNDING ASK
    # ════════════════════════════════════════════════════════════════════════
    if section == 'funding_needs':
        return system_context + f"""
SLIDE TYPE: Funding Ask

YOUR MISSION:
Make a clear, confident funding ask that shows investors exactly how their
capital will be deployed and what value-creating milestones it will unlock.

WHAT MAKES A GREAT FUNDING SLIDE:
- The ask is specific — a clear dollar amount and round type (Seed/Pre-Seed/Series A)
- Use of funds is broken down by percentage and category — shows financial discipline
- Milestones are specific and time-bound — what will be achieved in 12-18 months
- The milestone unlocks the NEXT fundraise — shows a clear path forward
- Ends with the vision of what {startup_name} looks like 18 months from now

STRUCTURE — write exactly 4 bullet points:
  Bullet 1: Total raise amount, round type, and current committed amount if any
  Bullet 2: Use of funds breakdown — 3 categories with percentages (must sum to 100%)
  Bullet 3: Primary milestone this funding unlocks — specific, measurable, time-bound
  Bullet 4: What {startup_name} looks like 18 months from now — the next inflection point

Use {funding_needs} if provided. Otherwise generate realistic figures for {stage} {industry} startup.
Seed round: $500K-$3M. Series A: $3M-$15M. Be specific and confident.
All target dates and quarters must be in {current_year} or later — never reference past dates.

Return ONLY this exact JSON:
{{"title": "Funding Ask", "content": "<bullet1>\\n<bullet2>\\n<bullet3>\\n<bullet4>", "chart_data": null}}
"""

    # ════════════════════════════════════════════════════════════════════════
    # REGENERATION / UNKNOWN SECTION
    # ════════════════════════════════════════════════════════════════════════
    if current_content:
        return system_context + f"""
SLIDE TYPE: {section.replace('_', ' ').title()} — Improvement Request

EXISTING CONTENT TO IMPROVE:
{current_content}

YOUR MISSION:
Rewrite and significantly improve the above slide content. Make every sentence
sharper, more specific, and more compelling for investors. Apply these upgrades:
- Replace any vague language with specific data points or examples
- Strengthen weak verbs — use active, confident language
- Ensure every bullet point earns its place — cut any filler
- Add quantification where it is missing (%, $, time, scale)
- Make the narrative arc clearer — each bullet should build on the previous
- All dates and timeframes must be in {current_year} or later — never reference past dates

Return ONLY this exact JSON:
{{"title": "{section.replace('_', ' ').title()}", "content": "<improved bullets, one per line separated by \\n>", "chart_data": null}}
"""

    # Generic fallback for any other section
    return system_context + f"""
SLIDE TYPE: {section.replace('_', ' ').title()}

Create compelling, investor-grade content for this slide.
Be specific, data-driven, and concise.
One bullet point per line, separated by \\n.
Maximum 4 bullet points. Maximum 18 words per bullet.
All dates and timeframes must be in {current_year} or later.

Return ONLY this exact JSON:
{{"title": "{section.replace('_', ' ').title()}", "content": "<3-4 bullets, one per line separated by \\n>", "chart_data": null}}
"""


# ── Core section generator ────────────────────────────────────────────────────
def generate_pitch_deck_section(
    section: str,
    context: dict,
    current_content: Optional[str] = None
) -> dict:
    """
    Generate content for one pitch deck slide using Ollama Cloud API.

    Uses the official ollama Python library with the hosted Cloud endpoint —
    no localhost or local GPU required.

    Args:
        section:         Slide key (e.g. 'cover', 'problem', 'market')
        context:         Startup information dictionary from the frontend form
        current_content: Optional existing content to improve (for regeneration)

    Returns:
        Dict with keys: 'title' (str), 'content' (str), 'chart_data' (dict or None)
    """
    prompt = _build_prompt(section, context, current_content)

    for attempt in range(2):
        try:
            # ✅ Uses the ollama Client defined at module level — no requests, no bare vars
            response = client.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )

            raw_text = response['message']['content']
            cleaned  = _clean_json(raw_text)
            result   = json.loads(cleaned)

            # Validate required keys are present
            if 'title' not in result or 'content' not in result:
                raise ValueError(f"Missing required keys for section: {section}")

            # Normalize content — model sometimes returns list instead of string
            if isinstance(result['content'], list):
                result['content'] = '\n'.join(str(item) for item in result['content'])

            result.setdefault('chart_data', None)
            return result

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[pitch_ai] Attempt {attempt + 1} JSON parse failed for '{section}': {e}")
            if attempt == 0:
                prompt += "\n\nCRITICAL: Return ONLY the JSON object. Zero additional text."
                continue

        except Exception as e:
            print(f"[pitch_ai] Ollama Cloud error for '{section}': {e}")
            break

    return _fallback(section)


# ── Full deck generator ───────────────────────────────────────────────────────
def generate_full_deck(context: dict) -> list:
    """
    Generate all 10 slides. Each slide is independently isolated —
    one failure never stops the rest from generating.

    Args:
        context: Startup information dictionary from the frontend form

    Returns:
        List of 10 slide dicts, each with 'title', 'content', 'chart_data'
    """
    deck_slides = []

    for section in SLIDE_SECTIONS:
        try:
            slide_data = generate_pitch_deck_section(section, context)
            print(f"[pitch_ai] ✓ {section}")
        except Exception as e:
            print(f"[pitch_ai] ✗ Fatal error on '{section}': {e}")
            slide_data = _fallback(section)

        deck_slides.append(slide_data)

    return deck_slides