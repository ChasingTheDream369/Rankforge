"""
Two-Stage LLM Scorer — orchestrator.

Delegates to stubs: extraction, d1, d2, d3, deterministic, llm_client.
OpenAI only.
"""

import json
import math
import re
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

from matcherapp.apps.system_prompts.scoring import (
    SCORING_CRITERIA,
    SCORING_PROMPT,
    SCORING_SYSTEM,
    VERIFY_PROMPT,
    few_shot_for_jd_profile,
    RESUME_GATE_SYSTEM,
    RESUME_GATE_USER_LEAD,
    RESUME_GATE_USER_MID,
)

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


def score_profiles(jd_profile: dict, resume_profile: dict) -> Optional[dict]:
    prompt = SCORING_PROMPT.format(
        few_shot=few_shot_for_jd_profile(jd_profile),
        jd_profile_json=json.dumps(jd_profile, separators=(',', ':')),
        resume_profile_json=json.dumps(resume_profile, separators=(',', ':')),
        criteria=SCORING_CRITERIA,
    )
    raw = call_scoring_llm(prompt, max_tokens=700, system=SCORING_SYSTEM)
    result = parse_json(raw)
    if result and all(k in result for k in ("d1_skills", "d2_seniority", "d3_domain", "d4_constraints")):
        return result
    return None


def verify_score(jd_profile: dict, resume_profile: dict, initial: dict) -> Optional[dict]:
    prompt = VERIFY_PROMPT.format(
        jd_profile_json=json.dumps(jd_profile, separators=(',', ':')),
        resume_profile_json=json.dumps(resume_profile, separators=(',', ':')),
        d1=initial.get("d1_skills", 0), d2=initial.get("d2_seniority", 0),
        d3=initial.get("d3_domain", 0), d4=initial.get("d4_constraints", 0),
        recommendation=initial.get("recommendation", "?"),
        criteria=SCORING_CRITERIA,
    )
    raw = call_scoring_llm(prompt, max_tokens=600, system=SCORING_SYSTEM)
    result = parse_json(raw)
    if result and all(k in result for k in ("d1_skills", "d2_seniority", "d3_domain", "d4_constraints")):
        return result
    return None


def is_resume(text: str) -> bool:
    if not has_llm():
        return True

    # Fast pattern-based pre-check — requires markers from TWO distinct categories
    # to avoid false-positives on non-resume docs (e.g. earnings reports with "Summary").
    resume_contact_pattern = re.compile(
        r'(?i)\b(email|phone|linkedin|github|portfolio|contact)\b'
        r'|[\w.+-]+@[\w-]+\.[a-z]{2,}'  # bare email address
    )
    resume_section_pattern = re.compile(
        r'(?i)\b(experience|education|employment|work history|skills|certifications?|'
        r'bachelor|master|university|college|projects?)\b'
    )
    if resume_contact_pattern.search(text) and len(resume_section_pattern.findall(text)) >= 2:
        return True

    from src.scoring.llm_client import call_extraction_llm

    # Check a larger leading snippet — compressed PDFs need more chars to be representative.
    try:
        snippet = text[:3000]
        raw = call_extraction_llm(
            RESUME_GATE_USER_LEAD + snippet, max_tokens=5, system=RESUME_GATE_SYSTEM
        )
        if raw is not None and raw.strip().upper().startswith("YES"):
            return True

        # Fallback: check the middle of the document in case the header is sparse.
        if len(text) > 3000:
            mid = len(text) // 2
            snippet2 = text[mid: mid + 2000]
            raw2 = call_extraction_llm(
                RESUME_GATE_USER_MID + snippet2,
                max_tokens=5, system=RESUME_GATE_SYSTEM,
            )
            if raw2 is not None and raw2.strip().upper().startswith("YES"):
                return True

        return False
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
        # Extract JD profile — only if not already provided (extracted once per run upstream).
        # Keep it isolated so a resume-extraction failure can't wipe a valid JD profile.
        if not jdp:
            try:
                jdp = extract_jd_profile(jd_text)
            except Exception as _jd_err:
                if verbose:
                    print(f"        [LLM] JD extraction failed ({_jd_err.__class__.__name__})")
                jdp = None

        # Extract resume profile independently.
        rsp = resume_profile
        if not rsp:
            try:
                rsp = extract_resume_profile(resume_text)
            except Exception as _rsp_err:
                if verbose:
                    print(f"        [LLM] Resume extraction failed ({_rsp_err.__class__.__name__}) — will use deterministic")

        if jdp and rsp:
            # Compute profile-based dimension scores FIRST — these are the most
            # valuable outputs and don't depend on the scoring LLM at all.
            d1_det = d2_det = d3_det = d4_det = None
            skill_detail_d1 = signals_detail_d2 = d3_reason = d4_checks = None
            try:
                d1_det, skill_detail_d1 = compute_d1_from_profiles(jdp, rsp, use_llm_fallback=D1_LLM_FALLBACK)
                d2_det, signals_detail_d2 = compute_d2(jdp, rsp)
                d3_det, d3_reason = compute_d3(jdp, rsp, use_llm_fallback=D3_LLM_FALLBACK)
                d4_det, d4_checks = compute_d4(jdp, rsp)
            except Exception as _dim_err:
                if verbose:
                    print(f"        [DIM] Dimension scoring failed ({_dim_err.__class__.__name__})")

            # Now call the scoring LLM for the narrative (strengths, gaps, rationale).
            # If this fails or returns None, the dimension scores above are still used.
            try:
                result = score_profiles(jdp, rsp)
                mode = "llm_two_stage"
            except Exception as _score_err:
                if verbose:
                    print(f"        [LLM] Scoring LLM failed ({_score_err.__class__.__name__})")
                result = None

            dims_ok = all(v is not None for v in (d1_det, d2_det, d3_det, d4_det))

            if result and dims_ok:
                # Best case: LLM narrative + profile-based dimensions
                result["d1_skills"] = d1_det
                result["d2_seniority"] = d2_det
                result["d3_domain"] = d3_det
                result["d4_constraints"] = d4_det
                result["_skill_detail_d1"] = skill_detail_d1
                result["_signals_detail_d2"] = signals_detail_d2
                result["_d3_reason"] = d3_reason
                result["_d4_checks"] = d4_checks

                if result.get("confidence") == "LOW":
                    if verbose:
                        print("        [Agentic] LOW confidence — running verification pass")
                    verified = verify_score(jdp, rsp, result)
                    if verified:
                        result = verified
                        mode = "agentic_retry"

            elif dims_ok:
                # Scoring LLM failed/returned bad JSON, but dimension scores are good.
                # Use profile-based dimensions with empty narrative instead of falling
                # all the way back to raw-text deterministic scoring.
                if verbose:
                    print("        [LLM] Scoring LLM returned no result — using profile-based dimensions")
                result = {
                    "d1_skills": d1_det, "d2_seniority": d2_det,
                    "d3_domain": d3_det, "d4_constraints": d4_det,
                    "_skill_detail_d1": skill_detail_d1,
                    "_signals_detail_d2": signals_detail_d2,
                    "_d3_reason": d3_reason, "_d4_checks": d4_checks,
                    "confidence": "MEDIUM", "strengths": [], "gaps": [],
                    "rationale": "Scored from extracted profiles (scoring LLM unavailable).",
                }
                mode = "llm_two_stage"

        # Raw-text deterministic only when OpenAI is configured but BOTH profile
        # extractions failed after retries (rare: repeated API/parse failures).
        if result is None:
            if verbose:
                print("        [Fallback] No profiles after LLM retries — raw-text deterministic")
            fallback = score_deterministic(jd_text, resume_text)
            result = {
                "d1_skills": fallback["d1_skills"]["score"], "d2_seniority": fallback["d2_seniority"]["score"],
                "d3_domain": fallback["d3_domain"]["score"], "d4_constraints": fallback["d4_constraints"]["score"],
                "confidence": fallback["confidence"], "strengths": fallback["strengths"],
                "gaps": fallback["gaps"], "rationale": fallback["rationale"], "_raw_detail": fallback,
            }
            mode = "deterministic_fallback"
    else:
        # No API key — regex-only path (intended deterministic use).
        fallback = score_deterministic(jd_text, resume_text)
        result = {
            "d1_skills": fallback["d1_skills"]["score"], "d2_seniority": fallback["d2_seniority"]["score"],
            "d3_domain": fallback["d3_domain"]["score"], "d4_constraints": fallback["d4_constraints"]["score"],
            "confidence": fallback["confidence"], "strengths": fallback["strengths"],
            "gaps": fallback["gaps"], "rationale": fallback["rationale"], "_raw_detail": fallback,
        }
        mode = "deterministic_no_key"

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
