"""
Two-Stage LLM Scorer — orchestrator.

Delegates to stubs: extraction, d1, d2, d3, deterministic, llm_client.
OpenAI only.
"""

import json
import math
from typing import Optional

from src.config import (
    LLM_PROVIDER, D1_LLM_FALLBACK, D2_AGENT_ENABLED, D3_LLM_FALLBACK,
)
from src.contracts import SkillEvidence

from src.scoring.llm_client import call_scoring_llm, parse_json, has_llm
from src.scoring.extraction import extract_jd_profile, extract_resume_profile
from src.scoring.d1 import compute_d1_from_profiles
from src.scoring.d2 import compute_d2
from src.scoring.d3 import compute_d3
from src.scoring.d4 import compute_d4
from src.scoring.deterministic import score_deterministic

# Re-export for external consumers
__all__ = [
    "extract_jd_profile", "extract_resume_profile",
    "score_resume", "score_profiles", "verify_score", "score_deterministic",
    "compute_d1_from_profiles", "compute_d2", "compute_d3",
    "compute_ce_weight", "is_resume", "LLM_PROVIDER",
    "compute_base_score", "compute_skill_penalty", "compute_final_score",
    "compute_confidence", "classify_recommendation",
]

# Dimension weights
W_SKILLS = 0.40
W_SENIORITY = 0.35
W_DOMAIN = 0.15
W_CONSTRAINTS = 0.10
MAX_CE_WEIGHT = 0.25
CE_RAMP_N = 5

# Stage 2 prompts
_SCORING_SYSTEM = (
    "You are a senior technical recruiter scoring candidates against job requirements. "
    "Score based solely on evidence in the profiles. Temperature=0. No creativity. Return valid JSON."
)

_FEW_SHOT_EXAMPLES = """## FEW-SHOT EXAMPLES
### Example 1 — STRONG_MATCH
JD_PROFILE: {"required_skills": [{"name": "Go/Python", "importance": "core"}, {"name": "PostgreSQL", "importance": "core"}, {"name": "AWS", "importance": "core"}, {"name": "Kafka", "importance": "core"}, {"name": "payment systems", "importance": "core"}, {"name": "microservices", "importance": "core"}], "years_required": 5, "domain": "fintech/payments", "seniority": "senior"}
RESUME_PROFILE: {"skills": [{"name": "Go/Python", "level": "BUILT_WITH", "evidence": "payment microservices handling 8000 TPS"}, {"name": "PostgreSQL", "level": "BUILT_WITH", "evidence": "50TB+ data"}, {"name": "Kafka", "level": "BUILT_WITH", "evidence": "event-driven microservices"}, {"name": "AWS", "level": "BUILT_WITH", "evidence": "ECS migration"}, {"name": "ETL/Airflow", "level": "BUILT_WITH", "evidence": "transaction reconciliation"}], "total_years": 7, "domains": ["fintech"], "seniority_signals": {"architecture": "migrated monolith to microservices", "leadership": "mentored 3 junior engineers", "scale": "8000 TPS, 2M+ users"}}
OUTPUT: {"d1_skills": 0.92, "d2_seniority": 0.88, "d3_domain": 1.0, "d4_constraints": 0.95, "confidence": "HIGH", "recommendation": "STRONG_MATCH", "strengths": ["Built payment microservices at scale"], "gaps": [], "rationale": "Exceptional fit."}
### Example 2 — NO_MATCH
JD_PROFILE: {"required_skills": [{"name": "Go/Python", "importance": "core"}, {"name": "PostgreSQL", "importance": "core"}], "years_required": 5, "domain": "fintech", "seniority": "senior"}
RESUME_PROFILE: {"skills": [{"name": "Kubernetes", "level": "BUILT_WITH", "evidence": "platform engineering"}, {"name": "Go", "level": "LISTED", "evidence": "basic scripting"}], "total_years": 6, "domains": ["devops"], "seniority_signals": {"architecture": "Kubernetes design", "ownership": "infra operations"}}
OUTPUT: {"d1_skills": 0.25, "d2_seniority": 0.45, "d3_domain": 0.3, "d4_constraints": 0.55, "confidence": "HIGH", "recommendation": "NO_MATCH", "strengths": ["Strong infra"], "gaps": ["No backend product experience"], "rationale": "Platform engineer, not backend."}"""

_SCORING_CRITERIA = """
## SCORING CRITERIA (use for all 4 dimensions)

D1 (Skills, weight 0.40): Match required_skills to resume skills. BUILT_WITH=1.0, USED=0.7, LISTED=0.3, ABSENT=0.0. Core skills weighted higher. D1 = weighted average.

D2 (Seniority, weight 0.35): Leadership (team/mentoring), Architecture (system design), Scale (users/TPS/metrics), Ownership (end-to-end). Use seniority_signals + total_years. 0.0–1.0.

D3 (Domain, weight 0.15): Same domain=1.0, adjacent/transferable=0.5–0.8, unrelated=0.0–0.3. Check jd_profile.domain vs resume_profile.domains.

D4 (Constraints, weight 0.10): Fraction of hard_constraints met. For each item in jd_profile.hard_constraints, check resume (total_years, skills, highlights). met=1.0, partial=0.5, not met=0.0. d4_constraints = average. If hard_constraints empty, use 1.0.
"""

_SCORING_PROMPT = """{few_shot}
---
## NOW SCORE THIS CANDIDATE
JD_PROFILE: {jd_profile_json}
RESUME_PROFILE: {resume_profile_json}
{criteria}

Return JSON only: {{"d1_skills": 0.XX, "d2_seniority": 0.XX, "d3_domain": 0.XX, "d4_constraints": 0.XX, "confidence": "HIGH|MEDIUM|LOW", "recommendation": "STRONG_MATCH|GOOD_MATCH|PARTIAL_MATCH|WEAK_MATCH|NO_MATCH", "strengths": [], "gaps": [], "rationale": "..."}}"""

_VERIFY_PROMPT = """You scored with LOW confidence. Review and revise.
JD_PROFILE: {jd_profile_json}
RESUME_PROFILE: {resume_profile_json}
INITIAL: D1={d1:.2f}, D2={d2:.2f}, D3={d3:.2f}, D4={d4:.2f}, {recommendation}
{criteria}
Return updated JSON (same schema): {{"d1_skills": 0.XX, "d2_seniority": 0.XX, "d3_domain": 0.XX, "d4_constraints": 0.XX, "confidence": "...", "recommendation": "...", "strengths": [], "gaps": [], "rationale": "..."}}"""


def score_profiles(jd_profile: dict, resume_profile: dict) -> Optional[dict]:
    prompt = _SCORING_PROMPT.format(
        few_shot=_FEW_SHOT_EXAMPLES,
        jd_profile_json=json.dumps(jd_profile, separators=(',', ':')),
        resume_profile_json=json.dumps(resume_profile, separators=(',', ':')),
        criteria=_SCORING_CRITERIA,
    )
    raw = call_scoring_llm(prompt, max_tokens=700, system=_SCORING_SYSTEM)
    result = parse_json(raw)
    if result and all(k in result for k in ("d1_skills", "d2_seniority", "d3_domain", "d4_constraints")):
        return result
    return None


def verify_score(jd_profile: dict, resume_profile: dict, initial: dict) -> Optional[dict]:
    prompt = _VERIFY_PROMPT.format(
        jd_profile_json=json.dumps(jd_profile, separators=(',', ':')),
        resume_profile_json=json.dumps(resume_profile, separators=(',', ':')),
        d1=initial.get("d1_skills", 0), d2=initial.get("d2_seniority", 0),
        d3=initial.get("d3_domain", 0), d4=initial.get("d4_constraints", 0),
        recommendation=initial.get("recommendation", "?"),
        criteria=_SCORING_CRITERIA,
    )
    raw = call_scoring_llm(prompt, max_tokens=600, system=_SCORING_SYSTEM)
    result = parse_json(raw)
    if result and all(k in result for k in ("d1_skills", "d2_seniority", "d3_domain", "d4_constraints")):
        return result
    return None


def is_resume(text: str) -> bool:
    if not has_llm():
        return True
    from src.scoring.llm_client import call_extraction_llm
    snippet = text[:1200]
    prompt = "Is the following document a professional resume or CV? Answer with exactly one word: YES or NO.\n\nDOCUMENT:\n" + snippet
    try:
        raw = call_extraction_llm(prompt, max_tokens=5, system="Answer YES or NO only.")
        return raw is not None and raw.strip().upper().startswith("YES")
    except Exception:
        return True


def compute_ce_weight(n_candidates: int) -> float:
    if n_candidates <= 1:
        return 0.0
    return min(MAX_CE_WEIGHT, MAX_CE_WEIGHT * (n_candidates - 1) / CE_RAMP_N)


def score_resume(
    jd_text: str,
    resume_text: str,
    ce_logit: float = 0.0,
    n_candidates: int = 1,
    adversarial_penalty: float = 0.0,
    verbose: bool = False,
    jd_profile: Optional[dict] = None,
    resume_profile: Optional[dict] = None,
) -> dict:
    mode = "deterministic"
    result = None
    rsp = None

    if has_llm():
        jdp = jd_profile or extract_jd_profile(jd_text)
        rsp = resume_profile or extract_resume_profile(resume_text)

        if jdp and rsp:
            result = score_profiles(jdp, rsp)
            mode = "llm_two_stage"

            d1_det, skill_detail_d1 = compute_d1_from_profiles(jdp, rsp, use_llm_fallback=D1_LLM_FALLBACK)
            d2_det, signals_detail_d2 = compute_d2(jdp, rsp)
            d3_det, d3_reason = compute_d3(jdp, rsp, use_llm_fallback=D3_LLM_FALLBACK)
            d4_det, d4_checks = compute_d4(jdp, rsp)
            if result:
                result["d1_skills"] = d1_det
                result["d2_seniority"] = d2_det
                result["d3_domain"] = d3_det
                result["d4_constraints"] = d4_det
                result["_skill_detail_d1"] = skill_detail_d1
                result["_signals_detail_d2"] = signals_detail_d2
                result["_d3_reason"] = d3_reason
                result["_d4_checks"] = d4_checks

            if result and result.get("confidence") == "LOW":
                if verbose:
                    print("        [Agentic] LOW confidence — running verification pass")
                verified = verify_score(jdp, rsp, result)
                if verified:
                    result = verified
                    mode = "agentic_retry"

        if result is None:
            if verbose:
                print("        [Fallback] LLM scoring failed — using deterministic")
            fallback = score_deterministic(jd_text, resume_text)
            result = {
                "d1_skills": fallback["d1_skills"]["score"], "d2_seniority": fallback["d2_seniority"]["score"],
                "d3_domain": fallback["d3_domain"]["score"], "d4_constraints": fallback["d4_constraints"]["score"],
                "confidence": fallback["confidence"], "strengths": fallback["strengths"],
                "gaps": fallback["gaps"], "rationale": fallback["rationale"], "_raw_detail": fallback,
            }
            mode = "deterministic_fallback"
    else:
        fallback = score_deterministic(jd_text, resume_text)
        result = {
            "d1_skills": fallback["d1_skills"]["score"], "d2_seniority": fallback["d2_seniority"]["score"],
            "d3_domain": fallback["d3_domain"]["score"], "d4_constraints": fallback["d4_constraints"]["score"],
            "confidence": fallback["confidence"], "strengths": fallback["strengths"],
            "gaps": fallback["gaps"], "rationale": fallback["rationale"], "_raw_detail": fallback,
        }

    d1 = float(result.get("d1_skills", 0))
    d2 = float(result.get("d2_seniority", 0))
    d3 = float(result.get("d3_domain", 0))
    d4 = float(result.get("d4_constraints", 0))
    d1, d2, d3, d4 = (max(0.0, min(1.0, v)) for v in (d1, d2, d3, d4))

    dim = W_SKILLS * d1 + W_SENIORITY * d2 + W_DOMAIN * d3 + W_CONSTRAINTS * d4
    alpha = compute_ce_weight(n_candidates)
    ce_sig = 1.0 / (1.0 + math.exp(-ce_logit))
    raw = (1.0 - alpha) * dim + alpha * ce_sig
    final = round(max(0.0, min(1.0, raw * (1.0 - adversarial_penalty))), 4)

    if verbose:
        print(f"      D1={d1:.2f} D2={d2:.2f} D3={d3:.2f} D4={d4:.2f} | dim={dim:.3f} | final={final:.4f} [{mode}]")

    conf = result.get("confidence", "MEDIUM")
    if n_candidates > CE_RAMP_N and ce_sig > 0:
        div = abs(dim - ce_sig)
        if div > 0.50:
            conf = "LOW"
        elif div > 0.30:
            conf = {"HIGH": "MEDIUM", "MEDIUM": "LOW", "LOW": "LOW"}.get(conf, "LOW")

    rec = result.get("recommendation") or (
        "STRONG_MATCH" if final >= 0.70 and conf in ("HIGH", "MEDIUM") else
        "GOOD_MATCH" if final >= 0.55 else
        "PARTIAL_MATCH" if final >= 0.35 else
        "WEAK_MATCH" if final >= 0.20 else "NO_MATCH"
    )

    raw_detail = result.get("_raw_detail", {})
    skill_detail = raw_detail.get("d1_skills", {}) if raw_detail else {}
    if result.get("_skill_detail_d1"):
        skill_detail = result["_skill_detail_d1"]
    elif not skill_detail and rsp and rsp.get("skills"):
        skill_detail = {"skills_checked": rsp["skills"]}

    # Layer justification for UI (D2/D3/D4)
    seniority_detail = result.get("_signals_detail_d2") or {}
    if not seniority_detail and raw_detail and "d2_seniority" in raw_detail:
        d2_raw = raw_detail["d2_seniority"]
        seniority_detail = {"signals": d2_raw.get("signals", []), "rationale": str(d2_raw.get("reasoning", ""))[:200]}

    domain_detail = {}
    if result.get("_d3_reason") is not None:
        domain_detail = {"reason": str(result["_d3_reason"])}
    elif raw_detail and "d3_domain" in raw_detail:
        d3_raw = raw_detail["d3_domain"]
        domain_detail = {"reason": str(d3_raw.get("reasoning", ""))[:200]}

    constraint_detail = result.get("_d4_checks") or []
    if not constraint_detail and raw_detail and "d4_constraints" in raw_detail:
        constraint_detail = raw_detail["d4_constraints"].get("checks", [])

    return {
        "d1_skills": d1, "d2_seniority": d2, "d3_domain": d3, "d4_constraints": d4,
        "dim_composite": round(dim, 4), "ce_sigmoid": round(ce_sig, 4), "ce_weight": round(alpha, 4),
        "raw_score": round(raw, 4), "adversarial_penalty": adversarial_penalty,
        "final_score": final, "confidence": conf, "recommendation": rec, "mode": mode,
        "strengths": result.get("strengths", []), "gaps": result.get("gaps", []),
        "rationale": result.get("rationale", ""), "skill_detail": skill_detail,
        "seniority_detail": seniority_detail, "domain_detail": domain_detail, "constraint_detail": constraint_detail,
    }


# Legacy stubs (ablation.py, etc.)
def compute_base_score(ce_logit: float) -> float:
    return 1.0 / (1.0 + math.exp(-ce_logit))


def compute_skill_penalty(skill_detail: dict):
    CRITICAL_IMPORTANCE = 4
    SKILL_IMPORTANCE = {"python": 5, "java": 5, "javascript": 5, "typescript": 5, "aws": 5, "kubernetes": 5, "docker": 4, "kafka": 4, "postgresql": 4, "react": 4, "sql": 4}
    PENALTY_PER_CRITICAL = 0.85
    missing = skill_detail.get("missing", [])
    evidence = []
    mult = 1.0
    for skill in missing:
        importance = SKILL_IMPORTANCE.get(skill.lower(), 3)
        if importance >= CRITICAL_IMPORTANCE:
            mult *= PENALTY_PER_CRITICAL
            evidence.append(SkillEvidence(requirement=skill, status="MISSING_CRITICAL", evidence_text="", evidence_location="none", confidence="HIGH", strength=0.0))
        else:
            evidence.append(SkillEvidence(requirement=skill, status="MISSING", evidence_text="", evidence_location="none", confidence="HIGH", strength=0.0))
    return round(mult, 4), evidence


def compute_final_score(ce_logit, skill_detail, adversarial_penalty=0.0):
    base = compute_base_score(ce_logit)
    penalty_mult, evidence = compute_skill_penalty(skill_detail)
    final = round(base * penalty_mult * (1.0 - adversarial_penalty), 4)
    return final, base, penalty_mult, evidence


def compute_confidence(evidence, llm_used=False):
    return "MEDIUM"


def classify_recommendation(score, confidence):
    if score >= 0.70 and confidence in ("HIGH", "MEDIUM"):
        return "STRONG_MATCH"
    if score >= 0.55:
        return "GOOD_MATCH"
    if score >= 0.35:
        return "PARTIAL_MATCH"
    if score >= 0.20:
        return "WEAK_MATCH"
    return "NO_MATCH"
