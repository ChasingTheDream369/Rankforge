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
    CE_WEIGHT,
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
    "get_dimension_weights", "normalize_custom_dimension_weights", "resolve_dimension_weights",
]

# Base dimension weights (general hiring)
W_SKILLS = 0.40
W_SENIORITY = 0.35
W_DOMAIN = 0.15
W_CONSTRAINTS = 0.10

# Role-level weight overrides: junior emphasizes skills, senior/staff emphasizes experience
ROLE_WEIGHTS = {
    "junior": (0.50, 0.25, 0.15, 0.10),   # skills-heavy
    "mid": (0.40, 0.35, 0.15, 0.10),      # default
    "senior": (0.35, 0.45, 0.12, 0.08),  # seniority-heavy
    "staff": (0.30, 0.50, 0.12, 0.08),   # leadership/architecture
    "lead": (0.30, 0.50, 0.12, 0.08),
    "executive": (0.25, 0.55, 0.12, 0.08),
}


def get_dimension_weights(jd_profile: Optional[dict] = None) -> tuple:
    """Return (w_skills, w_seniority, w_domain, w_constraints) from JD seniority if available."""
    if not jd_profile:
        return (W_SKILLS, W_SENIORITY, W_DOMAIN, W_CONSTRAINTS)
    seniority = (jd_profile.get("seniority") or "mid").lower().strip()
    return ROLE_WEIGHTS.get(seniority, (W_SKILLS, W_SENIORITY, W_DOMAIN, W_CONSTRAINTS))


def normalize_custom_dimension_weights(
    d1: float, d2: float, d3: float, d4: float
) -> Optional[tuple]:
    """Normalize four non-negative importance values to sum 1.0. None if sum is zero."""
    a = max(0.0, float(d1))
    b = max(0.0, float(d2))
    c = max(0.0, float(d3))
    d = max(0.0, float(d4))
    s = a + b + c + d
    if s <= 1e-9:
        return None
    return (a / s, b / s, c / s, d / s)


def resolve_dimension_weights(
    jd_profile: Optional[dict],
    custom_dim_weights: Optional[tuple] = None,
) -> tuple:
    """
    If custom_dim_weights is a 4-tuple summing to ~1, use it (re-normalize if needed).
    Otherwise fall back to get_dimension_weights(jd_profile) — base constants / role presets.
    """
    if custom_dim_weights is not None and len(custom_dim_weights) == 4:
        t = normalize_custom_dimension_weights(*custom_dim_weights)
        if t is not None:
            return t
    return get_dimension_weights(jd_profile)

# Stage 2 prompts
_SCORING_SYSTEM = (
    "You are a senior technical recruiter scoring candidates against job requirements. "
    "Score based solely on evidence in the profiles. Temperature=0. No creativity. Return valid JSON."
)

_FEW_SHOT_FINTECH = """## FEW-SHOT EXAMPLES (FinTech / Payments)
### Example 1 — STRONG_MATCH
JD_PROFILE: {"required_skills": [{"name": "Go/Python", "importance": "core"}, {"name": "PostgreSQL", "importance": "core"}, {"name": "AWS", "importance": "core"}, {"name": "Kafka", "importance": "core"}, {"name": "payment systems", "importance": "core"}, {"name": "microservices", "importance": "core"}], "years_required": 5, "domain": "fintech/payments", "seniority": "senior"}
RESUME_PROFILE: {"skills": [{"name": "Go/Python", "level": "BUILT_WITH", "evidence": "payment microservices handling 8000 TPS"}, {"name": "PostgreSQL", "level": "BUILT_WITH", "evidence": "50TB+ data"}, {"name": "Kafka", "level": "BUILT_WITH", "evidence": "event-driven microservices"}, {"name": "AWS", "level": "BUILT_WITH", "evidence": "ECS migration"}, {"name": "ETL/Airflow", "level": "BUILT_WITH", "evidence": "transaction reconciliation"}], "total_years": 7, "domains": ["fintech"], "seniority_signals": {"architecture": "migrated monolith to microservices", "leadership": "mentored 3 junior engineers", "scale": "8000 TPS, 2M+ users"}}
OUTPUT: {"d1_skills": 0.92, "d2_seniority": 0.88, "d3_domain": 1.0, "d4_constraints": 0.95, "confidence": "HIGH", "recommendation": "STRONG_MATCH", "strengths": ["Built payment microservices at scale"], "gaps": [], "rationale": "Exceptional fit."}
### Example 2 — NO_MATCH
JD_PROFILE: {"required_skills": [{"name": "Go/Python", "importance": "core"}, {"name": "PostgreSQL", "importance": "core"}], "years_required": 5, "domain": "fintech", "seniority": "senior"}
RESUME_PROFILE: {"skills": [{"name": "Kubernetes", "level": "BUILT_WITH", "evidence": "platform engineering"}, {"name": "Go", "level": "LISTED", "evidence": "basic scripting"}], "total_years": 6, "domains": ["devops"], "seniority_signals": {"architecture": "Kubernetes design", "ownership": "infra operations"}}
OUTPUT: {"d1_skills": 0.25, "d2_seniority": 0.45, "d3_domain": 0.3, "d4_constraints": 0.55, "confidence": "HIGH", "recommendation": "NO_MATCH", "strengths": ["Strong infra"], "gaps": ["No backend product experience"], "rationale": "Platform engineer, not backend."}"""

_FEW_SHOT_AI_ML = """## FEW-SHOT EXAMPLES (AI / ML)
### Example 1 — STRONG_MATCH
JD_PROFILE: {"required_skills": [{"name": "Python", "importance": "core"}, {"name": "PyTorch/TensorFlow", "importance": "core"}, {"name": "LLMs", "importance": "core"}, {"name": "RAG pipelines", "importance": "core"}, {"name": "vector databases", "importance": "nice"}], "years_required": 3, "domain": "ai_ml", "seniority": "senior"}
RESUME_PROFILE: {"skills": [{"name": "Python", "level": "BUILT_WITH", "evidence": "production ML pipelines"}, {"name": "PyTorch", "level": "BUILT_WITH", "evidence": "fine-tuned BERT for NER"}, {"name": "LangChain", "level": "BUILT_WITH", "evidence": "RAG for customer support"}, {"name": "Pinecone", "level": "USED", "evidence": "vector store for retrieval"}], "total_years": 5, "domains": ["ai_ml"], "seniority_signals": {"architecture": "designed RAG architecture", "scale": "10M+ queries/month"}}
OUTPUT: {"d1_skills": 0.88, "d2_seniority": 0.82, "d3_domain": 1.0, "d4_constraints": 0.9, "confidence": "HIGH", "recommendation": "STRONG_MATCH", "strengths": ["Strong LLM/RAG experience"], "gaps": [], "rationale": "Direct domain and skill match."}
### Example 2 — PARTIAL_MATCH
JD_PROFILE: {"required_skills": [{"name": "Python", "importance": "core"}, {"name": "scikit-learn", "importance": "core"}, {"name": "SQL", "importance": "core"}], "years_required": 2, "domain": "ai_ml", "seniority": "mid"}
RESUME_PROFILE: {"skills": [{"name": "Python", "level": "BUILT_WITH", "evidence": "data analysis scripts"}, {"name": "R", "level": "USED", "evidence": "statistical models"}, {"name": "SQL", "level": "USED", "evidence": "ETL queries"}], "total_years": 3, "domains": ["other"], "seniority_signals": {"ownership": "owned data pipeline"}}
OUTPUT: {"d1_skills": 0.55, "d2_seniority": 0.5, "d3_domain": 0.4, "d4_constraints": 0.7, "confidence": "MEDIUM", "recommendation": "PARTIAL_MATCH", "strengths": ["Python, SQL"], "gaps": ["No scikit-learn; adjacent domain"], "rationale": "Transferable skills but ML library gap."}"""


def _get_few_shot_examples(jd_profile: Optional[dict] = None) -> str:
    """Select few-shot examples by JD domain; avoid priming AI/ML JDs with only fintech context."""
    if not jd_profile:
        return _FEW_SHOT_FINTECH
    domain = str(jd_profile.get("domain", "")).lower()
    if "ai" in domain or "ml" in domain or "machine" in domain or "llm" in domain:
        return _FEW_SHOT_AI_ML
    return _FEW_SHOT_FINTECH

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
        few_shot=_get_few_shot_examples(jd_profile),
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


def compute_ce_weight(n_candidates: int = 1) -> float:
    """Fixed CE weight. All candidates get normalized CE score; blend is always (1-α)*dim + α*sigmoid(ce_logit)."""
    return CE_WEIGHT


def score_resume(
    jd_text: str,
    resume_text: str,
    ce_logit: float = 0.0,
    n_candidates: int = 1,
    adversarial_penalty: float = 0.0,
    verbose: bool = False,
    jd_profile: Optional[dict] = None,
    resume_profile: Optional[dict] = None,
    custom_dim_weights: Optional[tuple] = None,
) -> dict:
    mode = "deterministic"
    result = None
    rsp = None
    jdp = jd_profile  # may be extracted below; used for dimension weights and skill_detail

    if has_llm():
        jdp = jdp or extract_jd_profile(jd_text)
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

    w_skills, w_seniority, w_domain, w_constraints = resolve_dimension_weights(jdp, custom_dim_weights)
    _custom_ok = (
        custom_dim_weights is not None
        and len(custom_dim_weights) == 4
        and normalize_custom_dimension_weights(*custom_dim_weights) is not None
    )
    dim = w_skills * d1 + w_seniority * d2 + w_domain * d3 + w_constraints * d4
    alpha = compute_ce_weight(n_candidates)
    ce_sig = 1.0 / (1.0 + math.exp(-ce_logit))
    raw = (1.0 - alpha) * dim + alpha * ce_sig
    final = round(max(0.0, min(1.0, raw * (1.0 - adversarial_penalty))), 4)

    if verbose:
        print(f"      D1={d1:.2f} D2={d2:.2f} D3={d3:.2f} D4={d4:.2f} | dim={dim:.3f} | final={final:.4f} [{mode}]")

    conf = result.get("confidence", "MEDIUM")
    if ce_sig > 0:
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
        "dim_weights": {
            "d1_skills": round(w_skills, 4),
            "d2_seniority": round(w_seniority, 4),
            "d3_domain": round(w_domain, 4),
            "d4_constraints": round(w_constraints, 4),
            "source": "custom" if _custom_ok else "default",
        },
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
    """Derive HIGH/MEDIUM/LOW from skill evidence. LLM adds uncertainty → downgrade by one level when borderline."""
    n_missing_critical = 0
    n_missing = 0
    n_matched = 0

    if isinstance(evidence, list):
        for e in evidence:
            if hasattr(e, "status"):
                s = getattr(e, "status", "")
                if s == "MISSING_CRITICAL":
                    n_missing_critical += 1
                elif s == "MISSING":
                    n_missing += 1
                elif s in ("MATCHED", "ADJACENT", "GROUP"):
                    n_matched += 1
    elif isinstance(evidence, dict):
        n_missing_critical = len(evidence.get("missing_core", []))
        n_missing = len(evidence.get("missing", []))
        n_matched = len(evidence.get("matched", [])) + len(evidence.get("adjacent_matched", [])) + len(evidence.get("group_matched", []))
        if not n_matched and not n_missing and not n_missing_critical:
            skills_checked = evidence.get("skills_checked", evidence.get("d1_breakdown", []))
            for s in skills_checked:
                mt = s.get("match_type") or ""
                if "absent" in str(mt).lower():
                    n_missing += 1
                elif s.get("score", 0) > 0:
                    n_matched += 1
    total = n_matched + n_missing + n_missing_critical
    if total == 0:
        return "MEDIUM"

    if n_missing_critical > 0:
        base = "LOW"
    elif n_missing > n_matched:
        base = "LOW"
    elif n_matched >= total * 0.7:
        base = "HIGH"
    elif n_matched >= total * 0.4:
        base = "MEDIUM"
    else:
        base = "LOW"

    if llm_used and base == "HIGH" and n_matched < total * 0.85:
        return "MEDIUM"  # LLM uncertainty
    if llm_used and base == "MEDIUM" and n_missing > total * 0.3:
        return "LOW"
    return base


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
