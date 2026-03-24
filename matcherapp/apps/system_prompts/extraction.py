"""JD and resume structured profile extraction."""

JD_SYSTEM = (
    "You are a structured extraction assistant. Extract ONLY what is explicitly stated. "
    "Return valid JSON only, no markdown."
)

JD_PROMPT = """Extract a structured profile from this job description.

JOB DESCRIPTION:
{jd_text}

Return JSON: {{"required_skills": [{{"name": "...", "importance": "core" or "nice"}}], "years_required": 0, "domain": "fintech|enterprise_saas|ai_ml|healthcare|platform_devops|ecommerce|other", "seniority": "junior|mid|senior|staff|lead|executive", "hard_constraints": []}}
Max 10 skills."""

RESUME_SYSTEM = (
    "You are a structured extraction assistant. Extract ONLY what is explicitly written. "
    "Evidence must be VERBATIM quotes from the resume. Return valid JSON only, no markdown."
)

RESUME_PROMPT = """Extract a structured profile from this resume.

RESUME:
{resume_text}

Return JSON: {{"skills": [{{"name": "...", "level": "BUILT_WITH"|"USED"|"LISTED", "evidence": "verbatim quote"}}], "total_years": 0, "domains": [], "seniority_signals": {{"leadership": "quote or null", "architecture": "quote or null", "scale": "quote or null", "ownership": "quote or null"}}, "highlights": []}}
Evidence MUST be verbatim substrings from the resume text."""
