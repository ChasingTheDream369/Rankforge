"""D3 (Domain) — ontology first (exact/adjacent), LLM tool fallback when not covered."""

import json
from typing import Optional

from src.config import D3_LLM_FALLBACK, EXTRACTION_MODEL
from src.scoring.extraction_schema import ADJACENT_DOMAINS, canonicalize_domain
from src.scoring.llm_client import call_openai, parse_json, has_llm

from matcherapp.apps.system_prompts.dimensions import D3_DOMAIN_FALLBACK_SYSTEM, D3_DOMAIN_FALLBACK_USER

D3_FALLBACK_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_domain_fit_assessment",
        "description": "Submit your domain fit assessment. You MUST call this with score and reason.",
        "parameters": {
            "type": "object",
            "properties": {
                "score": {"type": "number", "description": "0-1: 1.0=same industry, 0.6-0.8=adjacent, 0.3-0.5=overlap, 0-0.2=unrelated"},
                "reason": {"type": "string", "description": "One sentence justification"},
            },
            "required": ["score", "reason"],
        },
    },
}


def compute_d3_from_profiles(jd_profile: dict, resume_profile: dict) -> Optional[tuple]:
    """
    Deterministic D3 from profiles. Returns (score, reason) when in ontology (exact or adjacent), None when not.
    """
    jd_domain = canonicalize_domain(str(jd_profile.get("domain", "") or ""))
    resume_domains_raw = resume_profile.get("domains") or []
    resume_domains = [d for d in (canonicalize_domain(str(d)) for d in resume_domains_raw if d) if d]
    if not resume_domains:
        resume_domains = ["other"]

    if jd_domain in resume_domains:
        return (1.0, f"Exact match: {jd_domain}")

    for r in resume_domains:
        if (jd_domain, r) in ADJACENT_DOMAINS or (r, jd_domain) in ADJACENT_DOMAINS:
            return (0.6, f"Adjacent domains: {jd_domain} ↔ {r}")

    return None


def call_d3_llm_fallback(jd_profile: dict, resume_profile: dict) -> tuple:
    jd_domain = str(jd_profile.get("domain", "") or "other")
    resume_domains = resume_profile.get("domains") or ["other"]
    prompt = D3_DOMAIN_FALLBACK_USER.format(jd_domain=jd_domain, resume_domains=resume_domains)
    if not has_llm():
        return (0.2, "LLM fallback failed; default unrelated")
    try:
        return call_d3_fallback_openai(prompt)
    except Exception:
        return (0.2, "LLM fallback failed; default unrelated")


def call_d3_fallback_openai(prompt: str) -> tuple:
    from openai import OpenAI
    from src.config import OPENAI_API_KEY
    client = OpenAI(api_key=OPENAI_API_KEY)
    msgs = [{"role": "system", "content": D3_DOMAIN_FALLBACK_SYSTEM}, {"role": "user", "content": prompt}]
    for _ in range(3):
        resp = client.chat.completions.create(
            model=EXTRACTION_MODEL, messages=msgs, max_tokens=150, temperature=0,
            tools=[D3_FALLBACK_TOOL], tool_choice={"type": "function", "function": {"name": "submit_domain_fit_assessment"}},
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == "submit_domain_fit_assessment":
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                        s = max(0.0, min(1.0, float(args.get("score", 0.2))))
                        r = str(args.get("reason", ""))[:120]
                        return (round(s, 2), r or "No reason")
                    except Exception:
                        pass
        if msg.content:
            parsed = parse_json(msg.content)
            if parsed and "score" in parsed:
                s = max(0.0, min(1.0, float(parsed.get("score", 0.2))))
                return (round(s, 2), str(parsed.get("reason", ""))[:120])
    return (0.2, "Parse failed; default unrelated")


def compute_d3(jd_profile: dict, resume_profile: dict, use_llm_fallback: bool = True) -> tuple:
    out = compute_d3_from_profiles(jd_profile, resume_profile)
    if out is not None:
        return out
    if use_llm_fallback and has_llm():
        try:
            return call_d3_llm_fallback(jd_profile, resume_profile)
        except Exception:
            pass
    return (0.2, "Not in ontology; LLM fallback disabled or failed")
