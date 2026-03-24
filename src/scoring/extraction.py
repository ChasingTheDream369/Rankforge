"""Profile extraction — JD and resume. OpenAI with optional tool-calling."""

import json
from typing import Optional

from src.scoring.llm_client import call_extraction_llm, parse_json, has_llm

from matcherapp.apps.system_prompts.extraction import (
    JD_PROMPT,
    JD_SYSTEM,
    RESUME_PROMPT,
    RESUME_SYSTEM,
)

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
    from src.config import OPENAI_TEMPERATURE, EXTRACTION_MODEL
    from src.scoring.llm_client import get_openai_client
    client = get_openai_client()
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


def extract_jd_profile(jd_text: str) -> Optional[dict]:
    """Two attempts: transient parse/API glitches should not force raw-text scoring."""
    from src.scoring.extraction_schema import normalize_jd_profile
    for attempt in range(2):
        cap = 4000 if attempt else 2500
        tokens = 1600 if attempt else 1200
        prompt = JD_PROMPT.format(jd_text=jd_text[:cap])
        raw = call_extraction_llm(prompt, tokens, JD_SYSTEM)
        result = parse_json(raw)
        if result and "required_skills" in result and "domain" in result:
            return normalize_jd_profile(result)
    return None


def extract_resume_profile(resume_text: str) -> Optional[dict]:
    """Two attempts with more headroom on retry; relax evidence check on retry only."""
    from src.scoring.extraction_schema import normalize_resume_profile
    for attempt in range(2):
        cap = 8000 if attempt else 4000
        tokens = 2500 if attempt else 2000
        prompt = RESUME_PROMPT.format(resume_text=resume_text[:cap])
        raw = call_extraction_llm(prompt, tokens, RESUME_SYSTEM)
        result = parse_json(raw)
        if result and "skills" in result and "total_years" in result:
            src = resume_text[:6000 if attempt else 3000]
            return normalize_resume_profile(
                result, source_text=src, validate_evidence=(attempt == 0)
            )
    return None
