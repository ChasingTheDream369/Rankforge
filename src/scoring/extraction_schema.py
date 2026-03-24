"""
Extraction Schema & Normalization — standardize JD and resume profiles for scoring.

- Fixed domain vocabulary (prevents "fintech" vs "FinTech" vs "financial services")
- Canonical skill mapping (Python/django/flask → python, golang → go, postgres → postgresql)
- Evidence validation (must be verbatim substring; null if not found)
- Reduces hallucination by constraining LLM output and post-normalizing

Field usage:
- highlights: Extracted from resume as 1-line summaries. Passed in RESUME_PROFILE to scoring LLM
  for context (D1–D4). Not used in deterministic scorer. Stored in skills_cache.
- hard_constraints: Extracted from JD. Used in D4 (Constraints) by scoring LLM — "fraction met".
  Deterministic scorer uses regex on raw JD text (years, langs, clouds), not jd_profile.hard_constraints.
"""

import re
from typing import Dict, List, Optional, Any

# Fixed domain vocabulary — closed set; all freeform mapped to one of these
# Canonical values: fintech | enterprise_saas | ai_ml | healthcare | platform_devops | ecommerce | other
DOMAIN_CANONICAL = {
    "fintech": "fintech",
    "payments": "fintech",
    "banking": "fintech",
    "financial": "fintech",
    "financial services": "fintech",
    "fintech/payments": "fintech",
    "fintech payments": "fintech",
    "enterprise_saas": "enterprise_saas",
    "enterprise": "enterprise_saas",
    "saas": "enterprise_saas",
    "b2b": "enterprise_saas",
    "ai_ml": "ai_ml",
    "ai": "ai_ml",
    "machine learning": "ai_ml",
    "ml": "ai_ml",
    "nlp": "ai_ml",
    "healthcare": "healthcare",
    "medical": "healthcare",
    "platform_devops": "platform_devops",
    "platform": "platform_devops",
    "devops": "platform_devops",
    "infrastructure": "platform_devops",
    "ecommerce": "ecommerce",
    "marketplace": "ecommerce",
    "retail": "ecommerce",
    # Role/function terms LLMs often return — domain = industry, so map to other
    "backend development": "other",
    "backend": "other",
    "frontend development": "other",
    "frontend": "other",
    "fullstack": "other",
    "full stack": "other",
    "software engineering": "other",
    "product development": "other",
    "engineering": "other",
}
# Canonical domain set — use for validation / prompts
DOMAIN_VALUES = frozenset({"fintech", "enterprise_saas", "ai_ml", "healthcare", "platform_devops", "ecommerce", "other"})

# Adjacent domains for D3 scoring (exact=1.0, adjacent=0.6)
ADJACENT_DOMAINS = {
    ("enterprise_saas", "fintech"), ("ai_ml", "enterprise_saas"),
    ("fintech", "platform_devops"), ("ai_ml", "fintech"),
}

# Skill aliases — map common variants to canonical (before ontology lookup)
# Extends ontology coverage for LLM-extracted variants
SKILL_ALIASES = {
    "node.js": "node",
    "nodejs": "node",
    "rest api": "rest_api",
    "rest apis": "rest_api",
    "restful": "rest_api",
    "restful api": "rest_api",
    "sql": "sql",  # keep generic; ontology maps sql->postgresql via adjacent
    "nosql": "mongodb",
    "event-driven": "event_driven",
    "event driven": "event_driven",
    "microservice": "microservices",
    "micro-services": "microservices",
    "distributed systems": "distributed_systems",
    "ci cd": "ci_cd",
    "cicd": "ci_cd",
    "continuous integration": "ci_cd",
    "machine learning": "machine_learning",
    "deep learning": "deep_learning",
    "data pipeline": "data_pipeline",
    "data pipelines": "data_pipeline",
    "scikit-learn": "scikit_learn",
    "scikit learn": "scikit_learn",
}

# Seniority — normalize variants
SENIORITY_CANONICAL = {
    "junior": "junior", "jr": "junior", "entry": "junior",
    "mid": "mid", "mid-level": "mid", "middle": "mid",
    "senior": "senior", "sr": "senior",
    "staff": "staff", "principal": "staff",
    "lead": "lead", "tech lead": "lead",
    "executive": "executive", "director": "executive", "vp": "executive",
}


def domain_string_to_canonical(raw: str) -> str:
    """Map freeform domain string to canonical. Returns 'other' if unknown."""
    if not raw or not isinstance(raw, str):
        return "other"
    key = raw.lower().strip()
    return DOMAIN_CANONICAL.get(key, "other")


def seniority_string_to_canonical(raw: str) -> str:
    """Map freeform seniority to canonical."""
    if not raw or not isinstance(raw, str):
        return "mid"
    key = raw.lower().strip()
    return SENIORITY_CANONICAL.get(key, "mid")


def skill_string_to_canonical(name: str, skill_canonical_map: Optional[Dict[str, str]] = None) -> str:
    """
    Map skill name to canonical form. Uses:
    1. SKILL_ALIASES for common variants
    2. Ontology map (surface form -> canonical)
    3. Compound split (Go/Python -> first matched part)
    4. Substring/longest match in ontology
    5. Fallback: lowercased trimmed (no inventing)
    """
    if not name or not isinstance(name, str):
        return ""
    n = name.lower().strip()
    if not n or len(n) < 2:
        return ""
    cmap = skill_canonical_map or {}

    # 1. Explicit alias
    if n in SKILL_ALIASES:
        return SKILL_ALIASES[n]

    # 2. Exact match in ontology map
    if n in cmap:
        return cmap[n]

    # 3. Compound "x/y" or "x, y" — canonicalize each part, return first match
    for sep in ("/", ",", " and ", " or "):
        if sep in n:
            parts = [p.strip() for p in n.split(sep) if p.strip()]
            for p in parts:
                if p in SKILL_ALIASES:
                    return SKILL_ALIASES[p]
                if p in cmap:
                    return cmap[p]
            # Use first part canonicalized if any matches
            if parts:
                first = skill_string_to_canonical(parts[0], cmap)
                if first:
                    return first
            return parts[0] if parts else n

    # 4. Longest ontology key that matches (substring)
    for key in sorted(cmap.keys(), key=len, reverse=True):
        if key in n or n in key:
            return cmap.get(key, key)
    return n  # return as-is if no match


def normalize_jd_profile(raw: Optional[dict], skill_canonical_map: Optional[Dict[str, str]] = None) -> Optional[dict]:
    """
    Post-process JD profile: canonical domains, skills, seniority.
    Ensures scoring receives consistent format.
    """
    if not raw or not isinstance(raw, dict):
        return raw
    scm = skill_canonical_map or get_skill_canonical_map()

    out = {}
    # required_skills
    skills_raw = raw.get("required_skills")
    if isinstance(skills_raw, list):
        out["required_skills"] = []
        for s in skills_raw[:15]:  # cap at 15
            if isinstance(s, dict) and "name" in s:
                name = skill_string_to_canonical(str(s.get("name", "")), scm)
                if name:
                    imp = str(s.get("importance", "nice")).lower()
                    out["required_skills"].append({
                        "name": name,
                        "importance": "core" if imp == "core" else "nice"
                    })
            elif isinstance(s, str) and s.strip():
                out["required_skills"].append({
                    "name": skill_string_to_canonical(s.strip(), scm),
                    "importance": "nice"
                })
    else:
        out["required_skills"] = []

    # years_required — integer 0-50
    yr = raw.get("years_required")
    try:
        out["years_required"] = max(0, min(50, int(yr))) if yr is not None else 0
    except (TypeError, ValueError):
        out["years_required"] = 0

    # domain
    out["domain"] = domain_string_to_canonical(raw.get("domain", ""))

    # seniority
    out["seniority"] = seniority_string_to_canonical(raw.get("seniority", ""))

    # hard_constraints — keep as list of strings, truncate
    hc = raw.get("hard_constraints")
    if isinstance(hc, list):
        out["hard_constraints"] = [str(x).strip()[:200] for x in hc if x][:10]
    else:
        out["hard_constraints"] = []

    return out


def evidence_is_verbatim(evidence: str, source_text: str) -> bool:
    """Check if evidence is a substring of source (allowing minor normalization)."""
    if not evidence or not source_text:
        return False
    ev = evidence.strip().lower()
    src = source_text.lower()
    # Normalize whitespace for comparison
    ev_norm = re.sub(r'\s+', ' ', ev)
    src_norm = re.sub(r'\s+', ' ', src)
    return ev_norm in src_norm or ev[:50] in src_norm


def normalize_resume_profile(
    raw: Optional[dict],
    source_text: str = "",
    skill_canonical_map: Optional[Dict[str, str]] = None,
    validate_evidence: bool = True,
) -> Optional[dict]:
    """
    Post-process resume profile: canonical skills, domains, validate evidence.
    If validate_evidence=True and evidence is not verbatim in source, set to null.
    """
    if not raw or not isinstance(raw, dict):
        return raw
    scm = skill_canonical_map or get_skill_canonical_map()
    source_lower = (source_text or "").lower()

    out = {}
    # skills
    skills_raw = raw.get("skills")
    if isinstance(skills_raw, list):
        out["skills"] = []
        for s in skills_raw[:15]:
            if isinstance(s, dict) and "name" in s:
                name = skill_string_to_canonical(str(s.get("name", "")), scm)
                if not name:
                    continue
                level = str(s.get("level", "LISTED")).upper()
                if level not in ("BUILT_WITH", "USED", "LISTED"):
                    level = "LISTED"
                evidence = str(s.get("evidence", "")).strip()[:300]
                if validate_evidence and evidence and source_text:
                    if not evidence_is_verbatim(evidence, source_text):
                        evidence = ""
                        level = "LISTED"  # downgrade — can't verify BUILT_WITH/USED
                out["skills"].append({"name": name, "level": level, "evidence": evidence})
            elif isinstance(s, str) and s.strip():
                out["skills"].append({
                    "name": skill_string_to_canonical(s.strip(), scm),
                    "level": "LISTED",
                    "evidence": ""
                })
    else:
        out["skills"] = []

    # total_years
    ty = raw.get("total_years")
    try:
        out["total_years"] = max(0, min(50, int(ty))) if ty is not None else 0
    except (TypeError, ValueError):
        out["total_years"] = 0

    # domains — normalize each
    domains_raw = raw.get("domains")
    if isinstance(domains_raw, list):
        seen = set()
        out["domains"] = []
        for d in domains_raw[:5]:
            c = domain_string_to_canonical(str(d)) if d else "other"
            if c != "other" and c not in seen:
                seen.add(c)
                out["domains"].append(c)
        if not out["domains"]:
            out["domains"] = ["other"]
    else:
        out["domains"] = ["other"]

    # seniority_signals — keep quotes but sanitize
    ss = raw.get("seniority_signals")
    if isinstance(ss, dict):
        out["seniority_signals"] = {}
        for k in ("leadership", "architecture", "scale", "ownership"):
            v = ss.get(k)
            if v and isinstance(v, str):
                v = v.strip()[:400]
                if validate_evidence and v and source_text and not evidence_is_verbatim(v, source_text):
                    v = ""
                out["seniority_signals"][k] = v if v else None
            else:
                out["seniority_signals"][k] = None
    else:
        out["seniority_signals"] = {"leadership": None, "architecture": None, "scale": None, "ownership": None}

    # highlights — keep as-is, truncate
    hl = raw.get("highlights")
    if isinstance(hl, list):
        out["highlights"] = [str(x).strip()[:200] for x in hl if x][:5]
    else:
        out["highlights"] = []

    return out


def get_ontology_skill_keys() -> set:
    """Return set of ontology surface forms. Kept for backward compat."""
    return set(build_ontology_skill_canonical_map().keys())


def build_ontology_skill_canonical_map() -> Dict[str, str]:
    """
    Build map: surface_form -> canonical for all ontology keys + adjacents.
    Used by skill_string_to_canonical for standardized skill matching.
    """
    try:
        from src.ingestion.ontology import SKILL_ONTOLOGY
    except Exception:
        return {}
    m: Dict[str, str] = {}
    for surface, meta in SKILL_ONTOLOGY.items():
        canon = meta.get("canonical", surface)
        m[surface.lower().strip()] = canon
        for adj in meta.get("adjacent", []):
            a = adj.lower().strip()
            if a and a not in m:
                m[a] = canon
    return m


def get_skill_canonical_map() -> Dict[str, str]:
    """Ontology-derived canonical map. SKILL_ALIASES override ontology when both present."""
    m = build_ontology_skill_canonical_map()
    m.update(SKILL_ALIASES)  # aliases override (e.g. to keep "sql" generic)
    return m


# --- Agent tools (for tool-augmented extraction) ---

def canonicalize_skill(skill_name: str) -> str:
    """
    Agent tool: map raw skill name to canonical form.
    Call this for every skill before including in required_skills or skills.
    Returns standardized form (e.g. "Golang" -> "go", "PostgreSQL" -> "postgresql").
    """
    result = skill_string_to_canonical(skill_name, get_skill_canonical_map())
    return result or skill_name.lower().strip() if skill_name else ""


def canonicalize_domain(domain_name: str) -> str:
    """
    Agent tool: map raw domain string to canonical domain.
    Call this for domain/domains before including in output.
    Returns one of: fintech, enterprise_saas, ai_ml, healthcare, platform_devops, ecommerce, other.
    """
    return domain_string_to_canonical(domain_name)
