"""D1 / D2 / D3 LLM fallback and agent prompts."""

import json
from typing import Any, Dict

# --- D1 skill-fit fallback ---
D1_SKILL_FIT_SYSTEM = "Assess skill fit. You MUST call submit_skill_fit_assessment with your score."

D1_SKILL_FIT_USER = """JD requires skill: {skill_name}. Resume context: {context}
Assess fit 0.0-1.0. You MUST call submit_skill_fit_assessment(score) with your assessment."""

# --- D2 agent ---
D2_SYSTEM = (
    "You are a senior recruiter scoring candidate seniority (D2). "
    "You MUST call exactly 5 tools: check_years_requirement, assess_leadership, assess_architecture, assess_scale, assess_ownership. "
    "Use the tool results as the signal scores. D2 = 0.25*L + 0.25*A + 0.20*S + 0.15*O + 0.15*Y. "
    "Return JSON only: {\"d2_score\": 0.XX, \"signals\": {\"leadership\": 0.X, \"architecture\": 0.X, \"scale\": 0.X, \"ownership\": 0.X, \"years\": 0.X}, \"rationale\": \"one sentence\"}"
)

D2_AGENT_USER = """Score this candidate's seniority (D2). You MUST call all 5 tools.

JD_PROFILE:
{jd_json}

RESUME_PROFILE:
{resume_json}

REQUIRED: check_years_requirement, assess_leadership, assess_architecture, assess_scale, assess_ownership. After ALL 5, return JSON."""


def format_d2_agent_user_message(jd_profile: Dict[str, Any], resume_profile: Dict[str, Any]) -> str:
    return D2_AGENT_USER.format(
        jd_json=json.dumps(jd_profile, indent=2),
        resume_json=json.dumps(resume_profile, indent=2),
    )


# --- D3 domain fallback ---
D3_DOMAIN_FALLBACK_SYSTEM = (
    "Score domain fit. You MUST call submit_domain_fit_assessment with score and reason."
)

D3_DOMAIN_FALLBACK_USER = """Score domain fit (0.0–1.0): JD domain={jd_domain}, Resume domains={resume_domains}.
Rules: 1.0=same industry; 0.6–0.8=adjacent/transferable; 0.3–0.5=some overlap; 0.0–0.2=unrelated.
You MUST call submit_domain_fit_assessment(score, reason) with your assessment."""
