"""D2 (Seniority) — agent + tools when enabled, else deterministic from profiles."""

import json
import re
from typing import Optional

from src.config import D2_AGENT_ENABLED, EXTRACTION_MODEL
from src.scoring.llm_client import parse_json, has_llm

_D2_WEIGHTS = {"leadership": 0.25, "architecture": 0.25, "scale": 0.20, "ownership": 0.15, "years": 0.15}

_D2_TOOLS = [
    {"type": "function", "function": {"name": "check_years_requirement", "description": "Check years. Call with years_required, total_years. Returns score 1.0 if meets, 0.8 if 1yr short, 0.5 if 2yr short.", "parameters": {"type": "object", "properties": {"years_required": {"type": "integer"}, "total_years": {"type": "integer"}}, "required": ["years_required", "total_years"]}}},
    {"type": "function", "function": {"name": "assess_leadership", "description": "Score leadership evidence. Pass verbatim quote. Strong: led team, mentored.", "parameters": {"type": "object", "properties": {"evidence": {"type": "string"}}, "required": ["evidence"]}}},
    {"type": "function", "function": {"name": "assess_architecture", "description": "Score architecture evidence. Strong: architected, designed system.", "parameters": {"type": "object", "properties": {"evidence": {"type": "string"}}, "required": ["evidence"]}}},
    {"type": "function", "function": {"name": "assess_scale", "description": "Score scale evidence. Strong: N users, TPS, $XM.", "parameters": {"type": "object", "properties": {"evidence": {"type": "string"}}, "required": ["evidence"]}}},
    {"type": "function", "function": {"name": "assess_ownership", "description": "Score ownership evidence. Strong: owned end-to-end.", "parameters": {"type": "object", "properties": {"evidence": {"type": "string"}}, "required": ["evidence"]}}},
]

_D2_SYSTEM = (
    "You are a senior recruiter scoring candidate seniority (D2). "
    "You MUST call exactly 5 tools: check_years_requirement, assess_leadership, assess_architecture, assess_scale, assess_ownership. "
    "Use the tool results as the signal scores. D2 = 0.25*L + 0.25*A + 0.20*S + 0.15*O + 0.15*Y. "
    "Return JSON only: {\"d2_score\": 0.XX, \"signals\": {\"leadership\": 0.X, \"architecture\": 0.X, \"scale\": 0.X, \"ownership\": 0.X, \"years\": 0.X}, \"rationale\": \"one sentence\"}"
)


def _assess_leadership_impl(evidence: str) -> tuple:
    rl = (evidence or "").lower().strip()
    if not rl or len(rl) < 5:
        return 0.0, ""
    score, matched = 0.0, ""
    if re.search(r'led\s+(?:a\s+)?(?:team|development|initiatives)|manag(?:ed|ing)\s+(?:a\s+)?\d+|mentor(?:ed|ing)?', rl):
        score, matched = 0.7, "leadership phrase"
    m = re.search(r'(\d+)[\s-]*(engineer|person|member|people)\s*(team|group)', rl)
    if m:
        s = min(1.0, 0.6 + int(m.group(1)) / 10)
        if s > score:
            score, matched = s, m.group(0)
    return round(min(1.0, score), 2), matched


def _assess_architecture_impl(evidence: str) -> tuple:
    rl = (evidence or "").lower().strip()
    if not rl or len(rl) < 5:
        return 0.0, ""
    for pat in [r'architect(?:ed|ing|ure)', r'design(?:ed|ing)\s+(?:a\s+)?(?:system|platform|pipeline|architecture)',
                r'technical\s+(?:architecture|ownership|decisions)', r'end[\s-]to[\s-]end']:
        if re.search(pat, rl):
            return 0.9, pat[:20]
    return 0.0, ""


def _assess_scale_impl(evidence: str) -> tuple:
    rl = (evidence or "").lower().strip()
    if not rl or len(rl) < 5:
        return 0.0, ""
    for pat, w in [(r'(\d[\d,]+)\+?\s*(?:active\s+)?users', 0.85), (r'(\d[\d.]*)\s*[Mm]illion?\s*(?:users|DAU)?', 0.85),
                   (r'(\d[\d,]+)\+?\s*tps', 0.9), (r'\$[\d.]+[BMKbmk]', 0.9)]:
        m = re.search(pat, rl)
        if m:
            return round(w, 2), m.group(0)[:40]
    return 0.0, ""


def _assess_ownership_impl(evidence: str) -> tuple:
    rl = (evidence or "").lower().strip()
    if not rl or len(rl) < 5:
        return 0.0, ""
    for pat in [r'from\s+(?:ideation|scratch)\s+(?:through|to)\s+production', r'end[\s-]to[\s-]end\s+(?:ownership|responsibility)',
                r'single[\s-]threaded\s+owner', r'owned\s+(?:the|from)', r'drove\s+(?:migration|implementation)\s+to\s+completion']:
        if re.search(pat, rl):
            return 0.9, pat[:30]
    if re.search(r'production\s+deployment|owned|ownership', rl):
        return 0.6, "ownership phrase"
    return 0.0, ""


def _execute_d2_tool(name: str, args: dict) -> str:
    if name == "check_years_requirement":
        req = max(0, int(args.get("years_required", 0)))
        act = max(0, int(args.get("total_years", 0)))
        if req == 0:
            score, ok = min(1.0, act / 5) if act > 0 else 0.3, True
        else:
            gap = req - act
            score = 1.0 if gap <= 0 else 0.8 if gap <= 1 else 0.5 if gap <= 2 else 0.2
            ok = act >= req
        return f"score: {score} | ok: {ok}"
    if name == "assess_leadership":
        s, m = _assess_leadership_impl(args.get("evidence", ""))
        return f"score: {s} | matched: {m}" if m else f"score: {s}"
    if name == "assess_architecture":
        s, m = _assess_architecture_impl(args.get("evidence", ""))
        return f"score: {s} | matched: {m}" if m else f"score: {s}"
    if name == "assess_scale":
        s, m = _assess_scale_impl(args.get("evidence", ""))
        return f"score: {s} | matched: {m}" if m else f"score: {s}"
    if name == "assess_ownership":
        s, m = _assess_ownership_impl(args.get("evidence", ""))
        return f"score: {s} | matched: {m}" if m else f"score: {s}"
    return ""


def _score_seniority_signals(signal_text: str) -> dict:
    rl = (signal_text or "").lower()
    signals = {}
    if re.search(r'led\s+(?:a\s+)?(?:team|development)|manag(?:ed|ing)\s+(?:a\s+)?\d+|mentor', rl):
        lead = 0.7
    else:
        lead = 0.0
    m = re.search(r'(\d+)[\s-]*(engineer|person|member|people)\s*(team|group)', rl)
    if m:
        lead = max(lead, min(1.0, int(m.group(1)) / 6))
    signals["leadership"] = lead
    arch = 0.0
    for pat in [r'architect(?:ed|ing|ure)', r'design(?:ed|ing)\s+(?:a\s+)?(?:system|platform|pipeline)', r'end[\s-]to[\s-]end']:
        if re.search(pat, rl):
            arch = max(arch, 0.9)
    signals["architecture"] = min(1.0, arch)
    scale = 0.0
    for pat, w in [(r'(\d[\d,]+)\+?\s*(?:active\s+)?users', 0.8), (r'\d+[Mm]\s*(?:users|DAU)?', 0.85), (r'(\d[\d,]+)\+?\s*tps', 0.9), (r'\$[\d.]+[BMKbmk]', 0.9)]:
        if re.search(pat, rl):
            scale = max(scale, w)
    signals["scale"] = min(1.0, scale)
    own = 0.0
    for pat, w in [(r'end[\s-]to[\s-]end\s+(?:ownership|responsibility)', 0.9), (r'from\s+(?:ideation|scratch)\s+(?:through|to)\s+production', 0.9), (r'\bownership\b|owned\s+(?:the|from)', 0.6)]:
        if re.search(pat, rl):
            own = max(own, w)
    signals["ownership"] = min(1.0, own)
    return signals


def compute_d2_from_profiles(jd_profile: dict, resume_profile: dict) -> tuple:
    years_required = max(0, int(jd_profile.get("years_required") or 0))
    total_years = max(0, int(resume_profile.get("total_years") or 0))
    ss = resume_profile.get("seniority_signals") or {}
    signal_parts = [str(ss.get(k, "") or "") for k in ("leadership", "architecture", "scale", "ownership")]
    signal_text = " ".join(p for p in signal_parts if p)
    signals = _score_seniority_signals(signal_text)
    if years_required > 0 and total_years > 0:
        gap = years_required - total_years
        signals["years"] = 1.0 if gap <= 0 else 0.8 if gap <= 1 else 0.5 if gap <= 2 else 0.2
    elif years_required > 0:
        signals["years"] = 0.2
    else:
        signals["years"] = min(1.0, total_years / 5) if total_years > 0 else 0.3
    d2 = sum(signals[k] * _D2_WEIGHTS[k] for k in _D2_WEIGHTS)
    d2 = round(min(1.0, max(0.0, d2)), 4)
    jd_seniority = str(jd_profile.get("seniority") or "mid").lower().strip()
    seniority_order = ["junior", "mid", "senior", "staff", "lead", "executive"]
    jd_idx = seniority_order.index(jd_seniority) if jd_seniority in seniority_order else 2
    res_idx = 0 if total_years < 2 else 1 if total_years < 5 else 2 if total_years < 8 else 3
    if jd_idx - res_idx == 1:
        d2 *= 0.9
    elif jd_idx - res_idx >= 2:
        d2 *= 0.75
    d2 = round(min(1.0, max(0.0, d2)), 4)
    signals_detail = {"signals": [{"type": k, "score": signals[k], "weight": _D2_WEIGHTS[k]} for k in _D2_WEIGHTS], "years_required": years_required, "total_years": total_years, "years_ok": total_years >= years_required if years_required > 0 else True}
    return d2, signals_detail


def _call_d2_agent_openai(prompt: str, system: str) -> Optional[str]:
    from openai import OpenAI
    from src.config import OPENAI_API_KEY
    client = OpenAI(api_key=OPENAI_API_KEY)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    for _ in range(8):
        resp = client.chat.completions.create(
            model=EXTRACTION_MODEL, messages=msgs, max_tokens=600, temperature=0,
            tools=_D2_TOOLS, tool_choice="auto",
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content
        msgs.append({"role": "assistant", "content": msg.content or "", "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except Exception:
                args = {}
            result = _execute_d2_tool(tc.function.name, args)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return None


def _call_d2_agent(jd_profile: dict, resume_profile: dict) -> Optional[tuple]:
    prompt = f"""Score this candidate's seniority (D2). You MUST call all 5 tools.

JD_PROFILE:
{json.dumps(jd_profile, indent=2)}

RESUME_PROFILE:
{json.dumps(resume_profile, indent=2)}

REQUIRED: check_years_requirement, assess_leadership, assess_architecture, assess_scale, assess_ownership. After ALL 5, return JSON."""
    raw = _call_d2_agent_openai(prompt, _D2_SYSTEM)
    if not raw:
        return None
    result = parse_json(raw)
    if result and "d2_score" in result:
        d2 = max(0.0, min(1.0, float(result.get("d2_score", 0))))
        sig = result.get("signals", {})
        signals_detail = {"signals": [{"type": k, "score": sig.get(k, 0), "weight": _D2_WEIGHTS.get(k, 0.15)} for k in _D2_WEIGHTS], "years_required": jd_profile.get("years_required"), "total_years": resume_profile.get("total_years"), "years_ok": (resume_profile.get("total_years", 0) or 0) >= (jd_profile.get("years_required", 0) or 0), "rationale": result.get("rationale", "")}
        return round(d2, 4), signals_detail
    return None


def compute_d2(jd_profile: dict, resume_profile: dict) -> tuple:
    if D2_AGENT_ENABLED and has_llm():
        try:
            out = _call_d2_agent(jd_profile, resume_profile)
            if out is not None:
                return out
        except Exception:
            pass
    return compute_d2_from_profiles(jd_profile, resume_profile)
