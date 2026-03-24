"""Stage-2 scoring prompts and resume-document gate."""

from typing import Optional

SCORING_SYSTEM = (
    "You are a senior technical recruiter scoring candidates against job requirements. "
    "Score based solely on evidence in the profiles. Temperature=0. No creativity. Return valid JSON."
)

FEW_SHOT_FINTECH = """## FEW-SHOT EXAMPLES (FinTech / Payments)
### Example 1 — STRONG_MATCH
JD_PROFILE: {"required_skills": [{"name": "Go/Python", "importance": "core"}, {"name": "PostgreSQL", "importance": "core"}, {"name": "AWS", "importance": "core"}, {"name": "Kafka", "importance": "core"}, {"name": "payment systems", "importance": "core"}, {"name": "microservices", "importance": "core"}], "years_required": 5, "domain": "fintech/payments", "seniority": "senior"}
RESUME_PROFILE: {"skills": [{"name": "Go/Python", "level": "BUILT_WITH", "evidence": "payment microservices handling 8000 TPS"}, {"name": "PostgreSQL", "level": "BUILT_WITH", "evidence": "50TB+ data"}, {"name": "Kafka", "level": "BUILT_WITH", "evidence": "event-driven microservices"}, {"name": "AWS", "level": "BUILT_WITH", "evidence": "ECS migration"}, {"name": "ETL/Airflow", "level": "BUILT_WITH", "evidence": "transaction reconciliation"}], "total_years": 7, "domains": ["fintech"], "seniority_signals": {"architecture": "migrated monolith to microservices", "leadership": "mentored 3 junior engineers", "scale": "8000 TPS, 2M+ users"}}
OUTPUT: {"d1_skills": 0.92, "d2_seniority": 0.88, "d3_domain": 1.0, "d4_constraints": 0.95, "confidence": "HIGH", "recommendation": "STRONG_MATCH", "strengths": ["Built payment microservices at scale"], "gaps": [], "rationale": "Exceptional fit."}
### Example 2 — NO_MATCH
JD_PROFILE: {"required_skills": [{"name": "Go/Python", "importance": "core"}, {"name": "PostgreSQL", "importance": "core"}], "years_required": 5, "domain": "fintech", "seniority": "senior"}
RESUME_PROFILE: {"skills": [{"name": "Kubernetes", "level": "BUILT_WITH", "evidence": "platform engineering"}, {"name": "Go", "level": "LISTED", "evidence": "basic scripting"}], "total_years": 6, "domains": ["devops"], "seniority_signals": {"architecture": "Kubernetes design", "ownership": "infra operations"}}
OUTPUT: {"d1_skills": 0.25, "d2_seniority": 0.45, "d3_domain": 0.3, "d4_constraints": 0.55, "confidence": "HIGH", "recommendation": "NO_MATCH", "strengths": ["Strong infra"], "gaps": ["No backend product experience"], "rationale": "Platform engineer, not backend."}"""

FEW_SHOT_AI_ML = """## FEW-SHOT EXAMPLES (AI / ML)
### Example 1 — STRONG_MATCH
JD_PROFILE: {"required_skills": [{"name": "Python", "importance": "core"}, {"name": "PyTorch/TensorFlow", "importance": "core"}, {"name": "LLMs", "importance": "core"}, {"name": "RAG pipelines", "importance": "core"}, {"name": "vector databases", "importance": "nice"}], "years_required": 3, "domain": "ai_ml", "seniority": "senior"}
RESUME_PROFILE: {"skills": [{"name": "Python", "level": "BUILT_WITH", "evidence": "production ML pipelines"}, {"name": "PyTorch", "level": "BUILT_WITH", "evidence": "fine-tuned BERT for NER"}, {"name": "LangChain", "level": "BUILT_WITH", "evidence": "RAG for customer support"}, {"name": "Pinecone", "level": "USED", "evidence": "vector store for retrieval"}], "total_years": 5, "domains": ["ai_ml"], "seniority_signals": {"architecture": "designed RAG architecture", "scale": "10M+ queries/month"}}
OUTPUT: {"d1_skills": 0.88, "d2_seniority": 0.82, "d3_domain": 1.0, "d4_constraints": 0.9, "confidence": "HIGH", "recommendation": "STRONG_MATCH", "strengths": ["Strong LLM/RAG experience"], "gaps": [], "rationale": "Direct domain and skill match."}
### Example 2 — PARTIAL_MATCH
JD_PROFILE: {"required_skills": [{"name": "Python", "importance": "core"}, {"name": "scikit-learn", "importance": "core"}, {"name": "SQL", "importance": "core"}], "years_required": 2, "domain": "ai_ml", "seniority": "mid"}
RESUME_PROFILE: {"skills": [{"name": "Python", "level": "BUILT_WITH", "evidence": "data analysis scripts"}, {"name": "R", "level": "USED", "evidence": "statistical models"}, {"name": "SQL", "level": "USED", "evidence": "ETL queries"}], "total_years": 3, "domains": ["other"], "seniority_signals": {"ownership": "owned data pipeline"}}
OUTPUT: {"d1_skills": 0.55, "d2_seniority": 0.5, "d3_domain": 0.4, "d4_constraints": 0.7, "confidence": "MEDIUM", "recommendation": "PARTIAL_MATCH", "strengths": ["Python, SQL"], "gaps": ["No scikit-learn; adjacent domain"], "rationale": "Transferable skills but ML library gap."}"""


def few_shot_for_jd_profile(jd_profile: Optional[dict] = None) -> str:
    """Select few-shot block by JD domain."""
    if not jd_profile:
        return FEW_SHOT_FINTECH
    domain = str(jd_profile.get("domain", "")).lower()
    if "ai" in domain or "ml" in domain or "machine" in domain or "llm" in domain:
        return FEW_SHOT_AI_ML
    return FEW_SHOT_FINTECH


SCORING_CRITERIA = """
## SCORING CRITERIA (use for all 4 dimensions)

D1 (Skills, weight 0.40): Match required_skills to resume skills. BUILT_WITH=1.0, USED=0.7, LISTED=0.3, ABSENT=0.0. Core skills weighted higher. D1 = weighted average.

D2 (Seniority, weight 0.35): Leadership (team/mentoring), Architecture (system design), Scale (users/TPS/metrics), Ownership (end-to-end). Use seniority_signals + total_years. 0.0–1.0.

D3 (Domain, weight 0.15): Same domain=1.0, adjacent/transferable=0.5–0.8, unrelated=0.0–0.3. Check jd_profile.domain vs resume_profile.domains.

D4 (Constraints, weight 0.10): Fraction of hard_constraints met. For each item in jd_profile.hard_constraints, check resume (total_years, skills, highlights). met=1.0, partial=0.5, not met=0.0. d4_constraints = average. If hard_constraints empty, use 1.0.
"""

SCORING_PROMPT = """{few_shot}
---
## NOW SCORE THIS CANDIDATE
JD_PROFILE: {jd_profile_json}
RESUME_PROFILE: {resume_profile_json}
{criteria}

Return JSON only: {{"d1_skills": 0.XX, "d2_seniority": 0.XX, "d3_domain": 0.XX, "d4_constraints": 0.XX, "confidence": "HIGH|MEDIUM|LOW", "recommendation": "STRONG_MATCH|GOOD_MATCH|PARTIAL_MATCH|WEAK_MATCH|NO_MATCH", "strengths": [], "gaps": [], "rationale": "..."}}"""

VERIFY_PROMPT = """You scored with LOW confidence. Review and revise.
JD_PROFILE: {jd_profile_json}
RESUME_PROFILE: {resume_profile_json}
INITIAL: D1={d1:.2f}, D2={d2:.2f}, D3={d3:.2f}, D4={d4:.2f}, {recommendation}
{criteria}
Return updated JSON (same schema): {{"d1_skills": 0.XX, "d2_seniority": 0.XX, "d3_domain": 0.XX, "d4_constraints": 0.XX, "confidence": "...", "recommendation": "...", "strengths": [], "gaps": [], "rationale": "..."}}"""

# --- Resume gate (is_resume) ---
RESUME_GATE_SYSTEM = "Answer YES or NO only."
RESUME_GATE_USER_LEAD = (
    "Does the following text appear to be a professional resume or CV, or does it contain "
    "typical resume content such as work experience, education, skills, or contact information? "
    "Answer with exactly one word: YES or NO.\n\nDOCUMENT:\n"
)
RESUME_GATE_USER_MID = (
    "Does the following text contain professional resume content "
    "(experience, education, skills)? Answer YES or NO only.\n\nTEXT:\n"
)
