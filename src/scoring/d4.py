"""
D4 (Constraints) — fraction of JD hard_constraints met by resume.

IN:  jd_profile.hard_constraints (list), resume_profile (skills, total_years, highlights)
OUT: d4_constraints in [0, 1]

HOW: For each constraint, assess_constraint() returns 1.0|0.5|0.0. D4 = average.
     Fallback: check_constraints() on raw text when no profiles.
"""

import re


def assess_constraint(constraint: str, evidence: str) -> float:
    """
    Deterministic: score one constraint against evidence. Returns 1.0 (met), 0.5 (partial), 0.0 (not met).
    Evidence = concatenation of resume profile fields (total_years, skills, highlights).
    """
    c = (constraint or "").lower()
    e = (evidence or "").lower()
    if not c:
        return 1.0

    # Years: "5+ years", "3+ yrs experience", etc.
    ym = re.search(r'(\d+)\+?\s*years?', c)
    if ym:
        req = int(ym.group(1))
        yr = re.findall(r'(20\d{2})\s*[-–—]\s*(20\d{2}|present|current|now)', e)
        if yr:
            spans = [(int(s), 2026 if x in ('present', 'current', 'now') else int(x)) for s, x in yr]
            actual = max(x for _, x in spans) - min(s for s, _ in spans)
            if actual >= req:
                return 1.0
            if actual >= req - 1:
                return 0.5
        # Check for "N years" in evidence
        m = re.search(r'(\d+)\s*years?', e)
        if m and int(m.group(1)) >= req:
            return 1.0
        return 0.0

    # Languages
    LANGS = ["python", "go", "java", "typescript", "javascript", "rust", "scala"]
    for lang in LANGS:
        if lang in c:
            return 1.0 if lang in e else 0.0

    # Clouds
    for cloud in ["aws", "gcp", "azure"]:
        if cloud in c:
            terms = {"aws": ["aws", "ec2", "s3", "lambda", "ecs"], "gcp": ["gcp", "bigquery", "gke"], "azure": ["azure", "aks"]}
            return 1.0 if any(t in e for t in terms.get(cloud, [cloud])) else 0.0

    # DBs
    DBS = ["postgresql", "postgres", "mysql", "redis", "mongodb", "dynamodb"]
    for db in DBS:
        if db in c:
            return 1.0 if db in e or "postgres" in e and "postgresql" in c else 0.0

    # Keywords: kafka, microservices, docker, kubernetes
    for kw in ["kafka", "microservices", "docker", "kubernetes", "k8s"]:
        if kw in c:
            variants = {"kafka": ["kafka", "kinesis"], "microservices": ["microservice", "micro-service"], "docker": ["docker", "container"], "kubernetes": ["kubernetes", "k8s"], "k8s": ["k8s", "kubernetes"]}
            return 1.0 if any(v in e for v in variants.get(kw, [kw])) else 0.0

    # Generic: constraint phrase appears in evidence
    words = [w for w in re.split(r'\W+', c) if len(w) > 2]
    if words and sum(1 for w in words if w in e) >= max(1, len(words) // 2):
        return 1.0
    if words and any(w in e for w in words):
        return 0.5
    return 0.0


def build_evidence(resume_profile: dict) -> str:
    """Concatenate resume fields for constraint matching."""
    parts = []
    if resume_profile.get("total_years"):
        parts.append(f"{resume_profile['total_years']} years")
    for s in (resume_profile.get("skills") or [])[:15]:
        parts.append(str(s.get("name", "")))
        parts.append(str(s.get("evidence", "")))
    parts.extend(resume_profile.get("highlights") or [])
    for k, v in (resume_profile.get("seniority_signals") or {}).items():
        if v:
            parts.append(str(v))
    return " ".join(parts).lower()


def compute_d4_from_profiles(jd_profile: dict, resume_profile: dict) -> tuple:
    """
    D4 from profiles. For each hard_constraint, assess vs resume evidence. Return (score, checks).
    """
    constraints = jd_profile.get("hard_constraints") or []
    if not constraints:
        return (1.0, [])

    evidence = build_evidence(resume_profile)
    checks = []
    for c in constraints[:15]:
        s = assess_constraint(str(c), evidence)
        checks.append({"constraint": str(c)[:80], "score": s})
    avg = sum(x["score"] for x in checks) / len(checks)
    return (round(avg, 4), checks)


def compute_d4(jd_profile: dict, resume_profile: dict) -> tuple:
    """
    D4 from profiles. Caller uses check_constraints on raw text when no profiles (score_deterministic).
    Returns (d4_score, checks).
    """
    return compute_d4_from_profiles(jd_profile, resume_profile)
