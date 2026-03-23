"""Profile extraction — JD and resume. OpenAI with optional tool-calling."""

import json
from typing import Optional

from src.config import EXTRACTION_USE_TOOLS, EXTRACTION_MODEL
from src.scoring.llm_client import call_openai, call_extraction_llm, parse_json, has_llm

_EXTRACTION_TOOLS = [
    {"type": "function", "function": {"name": "canonicalize_skill", "description": "Get canonical form of a skill. Call for EVERY skill.", "parameters": {"type": "object", "properties": {"skill_name": {"type": "string"}}, "required": ["skill_name"]}}},
    {"type": "function", "function": {"name": "canonicalize_domain", "description": "Get canonical domain. Returns: fintech|enterprise_saas|ai_ml|healthcare|platform_devops|ecommerce|other", "parameters": {"type": "object", "properties": {"domain_name": {"type": "string"}}, "required": ["domain_name"]}}},
]


def _execute_extraction_tool(name: str, args: dict) -> str:
    from src.scoring.extraction_schema import canonicalize_skill, canonicalize_domain
    if name == "canonicalize_skill":
        return canonicalize_skill(args.get("skill_name", ""))
    if name == "canonicalize_domain":
        return canonicalize_domain(args.get("domain_name", ""))
    return ""


def _call_extraction_with_tools(prompt: str, system: str, max_tokens: int = 800) -> Optional[str]:
    if not has_llm():
        return None
    from openai import OpenAI
    from src.config import OPENAI_API_KEY, OPENAI_TEMPERATURE
    client = OpenAI(api_key=OPENAI_API_KEY)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    for _ in range(8):
        resp = client.chat.completions.create(
            model=EXTRACTION_MODEL, messages=msgs, max_tokens=max_tokens, temperature=OPENAI_TEMPERATURE,
            tools=_EXTRACTION_TOOLS, tool_choice="auto",
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
            result = _execute_extraction_tool(tc.function.name, args)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return None


_JD_SYSTEM = "You are a structured extraction assistant. Extract ONLY what is explicitly stated. Use canonicalize_skill for EVERY skill and canonicalize_domain for domain. Return JSON only."
_JD_PROMPT = """Extract a structured profile from this job description.

JOB DESCRIPTION:
{jd_text}

Return JSON: {{"required_skills": [{{"name": "...", "importance": "core" or "nice"}}], "years_required": 0, "domain": "fintech|enterprise_saas|ai_ml|healthcare|platform_devops|ecommerce|other", "seniority": "junior|mid|senior|staff|lead|executive", "hard_constraints": []}}
Max 10 skills. Use tools for skill names and domain."""

_RESUME_SYSTEM = "You are a structured extraction assistant. Extract ONLY what is explicitly written. Evidence must be VERBATIM. Use canonicalize_skill for EVERY skill and canonicalize_domain for domains. Return JSON only."
_RESUME_PROMPT = """Extract a structured profile from this resume.

RESUME:
{resume_text}

Return JSON: {{"skills": [{{"name": "...", "level": "BUILT_WITH"|"USED"|"LISTED", "evidence": "verbatim quote"}}], "total_years": 0, "domains": [], "seniority_signals": {{"leadership": "quote or null", "architecture": "quote or null", "scale": "quote or null", "ownership": "quote or null"}}, "highlights": []}}
Evidence MUST be verbatim. Use tools for skills and domains."""


def extract_jd_profile(jd_text: str) -> Optional[dict]:
    from src.scoring.extraction_schema import normalize_jd_profile
    prompt = _JD_PROMPT.format(jd_text=jd_text[:2500])
    if EXTRACTION_USE_TOOLS and has_llm():
        raw = _call_extraction_with_tools(prompt, _JD_SYSTEM, 600)
    else:
        raw = call_extraction_llm(prompt, 600, _JD_SYSTEM)
    result = parse_json(raw)
    if result and "required_skills" in result and "domain" in result:
        return normalize_jd_profile(result)
    return None


def extract_resume_profile(resume_text: str) -> Optional[dict]:
    from src.scoring.extraction_schema import normalize_resume_profile
    prompt = _RESUME_PROMPT.format(resume_text=resume_text[:4000])
    if EXTRACTION_USE_TOOLS and has_llm():
        raw = _call_extraction_with_tools(prompt, _RESUME_SYSTEM, 800)
    else:
        raw = call_extraction_llm(prompt, 800, _RESUME_SYSTEM)
    result = parse_json(raw)
    if result and "skills" in result and "total_years" in result:
        return normalize_resume_profile(result, source_text=resume_text[:3000], validate_evidence=True)
    return None
