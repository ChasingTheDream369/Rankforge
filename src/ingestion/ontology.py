"""
Skill Ontology & Extraction — three-tier matching with canonical forms.

CURRENT USAGE (pipeline):
  - SKILL_ONTOLOGY: used in sanitizer for keyword-stuffing detection only
  - compute_skill_overlap, extract_skills_ontology, get_adjacent_skills: NOT used
  - explainability.py expects skill_detail.adjacent_matched/group_matched but these
    are never populated (scorer uses LLM/ALIAS_MAP, not ontology)
  → Skills relevancy and adjacency graph: implemented but NOT wired into pipeline

Extraction chain (in order of preference):
  1. esco-skill-extractor — maps raw text to 13,000+ ESCO taxonomy skills
     via cosine similarity against pre-embedded ESCO vectors
  2. LLM structured extraction — OpenAI/Anthropic, temp=0, JSON output
  3. Curated 60+ entry ontology — canonical/group/adjacent lookup table
  4. Regex fallback — pattern-based extraction, fully offline

Three-tier overlap: exact (1.0) → adjacent (0.6) → group (0.3)

Production upgrade path:
  - Fine-tune bi-encoder on recruitment corpora (SkillBERT approach)
  - Resume2Vec-style domain embedding for 15%+ nDCG improvement
  - Replace curated ontology with full ESCO knowledge graph (13K skills)
"""

import re
import json
from typing import Set, Dict, Optional, List

from src.config import LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE
# === ESCO Skill Extractor (primary path when available) ===

ESCO_AVAILABLE = False
_esco_extractor = None


def get_esco_extractor():
    """Lazy-load ESCO extractor (downloads model on first use)."""
    global ESCO_AVAILABLE, _esco_extractor
    if _esco_extractor is not None:
        return _esco_extractor
    try:
        from esco_skill_extractor import SkillExtractor
        _esco_extractor = SkillExtractor()
        ESCO_AVAILABLE = True
        print("  ESCO skill extractor loaded (13K+ skills)")
        return _esco_extractor
    except ImportError:
        ESCO_AVAILABLE = False
        return None
    except Exception as e:
        print(f"  ESCO extractor failed to load: {e}")
        ESCO_AVAILABLE = False
        return None


def extract_skills_esco(text: str) -> Optional[List[str]]:
    """
    Extract skills using ESCO taxonomy via cosine similarity.
    Returns list of ESCO skill URIs or None if unavailable.
    """
    extractor = get_esco_extractor()
    if extractor is None:
        return None
    try:
        skills = extractor.get_skills([text])
        if skills and skills[0]:
            return skills[0]  # list of ESCO URIs for first text
    except Exception as e:
        print(f"  ESCO extraction failed: {e}")
    return None


def extract_occupations_esco(text: str) -> Optional[List[str]]:
    """
    Extract occupations using ESCO taxonomy via cosine similarity.
    Maps resume/JD text to standardized occupation IDs (e.g., "backend developer",
    "data engineer"). Used for role alignment scoring in transferability dimension.
    """
    extractor = get_esco_extractor()
    if extractor is None:
        return None
    try:
        occupations = extractor.get_occupations([text])
        if occupations and occupations[0]:
            return occupations[0]
    except Exception as e:
        print(f"  ESCO occupation extraction failed: {e}")
    return None

# === Skill Ontology (ESCO/O*NET-inspired) ===
# Each entry has: canonical form, group, adjacent skills, parent concept(s), importance (1-5 O*NET scale)
# Parent concepts enable graph expansion before retrieval
# Importance weights enable weighted scoring (missing a critical skill penalizes more)


SKILL_ONTOLOGY = {
    # Programming Languages — parent: "software development"
    "python":       {"canonical": "python",       "group": "programming",       "adjacent": ["django", "flask", "fastapi"], "parents": ["software development"], "importance": 5},
    "go":           {"canonical": "go",           "group": "programming",       "adjacent": ["golang"], "parents": ["software development"], "importance": 4},
    "golang":       {"canonical": "go",           "group": "programming",       "adjacent": ["go"], "parents": ["software development"], "importance": 4},
    "java":         {"canonical": "java",         "group": "programming",       "adjacent": ["spring boot", "spring"], "parents": ["software development"], "importance": 5},
    "typescript":   {"canonical": "typescript",   "group": "programming",       "adjacent": ["javascript", "react"], "parents": ["software development", "web development"], "importance": 4},
    "javascript":   {"canonical": "javascript",   "group": "programming",       "adjacent": ["typescript", "react", "node"], "parents": ["software development", "web development"], "importance": 4},
    "rust":         {"canonical": "rust",         "group": "programming",       "adjacent": ["go", "c++"], "parents": ["software development", "systems programming"], "importance": 3},
    "c++":          {"canonical": "c++",          "group": "programming",       "adjacent": ["rust", "c"], "parents": ["software development", "systems programming"], "importance": 3},
    "c#":           {"canonical": "c#",           "group": "programming",       "adjacent": [".net"], "parents": ["software development"], "importance": 3},
    "ruby":         {"canonical": "ruby",         "group": "programming",       "adjacent": ["rails"], "parents": ["software development", "web development"], "importance": 3},
    "scala":        {"canonical": "scala",        "group": "programming",       "adjacent": ["java", "spark"], "parents": ["software development", "data engineering"], "importance": 3},
    "kotlin":       {"canonical": "kotlin",       "group": "programming",       "adjacent": ["java"], "parents": ["software development", "mobile development"], "importance": 3},
    # Frameworks — parent: "web development" / "software development"
    "django":       {"canonical": "django",       "group": "web_framework",     "adjacent": ["python", "flask", "fastapi"], "parents": ["web development", "python"], "importance": 4},
    "flask":        {"canonical": "flask",        "group": "web_framework",     "adjacent": ["python", "django", "fastapi"], "parents": ["web development", "python"], "importance": 3},
    "fastapi":      {"canonical": "fastapi",      "group": "web_framework",     "adjacent": ["python", "flask"], "parents": ["web development", "python"], "importance": 4},
    "spring boot":  {"canonical": "spring_boot",  "group": "web_framework",     "adjacent": ["java", "spring"], "parents": ["web development", "java"], "importance": 4},
    "spring":       {"canonical": "spring",       "group": "web_framework",     "adjacent": ["java", "spring boot"], "parents": ["web development", "java"], "importance": 4},
    "react":        {"canonical": "react",        "group": "frontend",          "adjacent": ["javascript", "typescript", "angular", "vue"], "parents": ["web development", "frontend development"], "importance": 4},
    "angular":      {"canonical": "angular",      "group": "frontend",          "adjacent": ["react", "vue", "typescript"], "parents": ["web development", "frontend development"], "importance": 3},
    "vue":          {"canonical": "vue",          "group": "frontend",          "adjacent": ["react", "angular"], "parents": ["web development", "frontend development"], "importance": 3},
    "node":         {"canonical": "node",         "group": "web_framework",     "adjacent": ["javascript", "express"], "parents": ["web development", "backend development"], "importance": 4},
    "express":      {"canonical": "express",      "group": "web_framework",     "adjacent": ["node", "javascript"], "parents": ["web development"], "importance": 3},
    # Databases — parent: "data management"
    "postgresql":   {"canonical": "postgresql",   "group": "database",          "adjacent": ["postgres", "sql", "mysql"], "parents": ["data management", "relational databases"], "importance": 5},
    "postgres":     {"canonical": "postgresql",   "group": "database",          "adjacent": ["postgresql", "sql"], "parents": ["data management", "relational databases"], "importance": 5},
    "mysql":        {"canonical": "mysql",        "group": "database",          "adjacent": ["postgresql", "sql"], "parents": ["data management", "relational databases"], "importance": 4},
    "mongodb":      {"canonical": "mongodb",      "group": "nosql",            "adjacent": ["dynamodb"], "parents": ["data management", "nosql databases"], "importance": 3},
    "dynamodb":     {"canonical": "dynamodb",     "group": "nosql",            "adjacent": ["mongodb"], "parents": ["data management", "nosql databases", "aws"], "importance": 3},
    "redis":        {"canonical": "redis",        "group": "cache",            "adjacent": ["memcached"], "parents": ["data management", "caching systems"], "importance": 4},
    "elasticsearch":{"canonical": "elasticsearch","group": "search_engine",     "adjacent": ["opensearch"], "parents": ["data management", "search systems"], "importance": 3},
    # Message Queues — parent: "distributed systems"
    "kafka":        {"canonical": "kafka",        "group": "message_queue",     "adjacent": ["rabbitmq", "sqs", "event-driven"], "parents": ["distributed systems", "event-driven architecture"], "importance": 5},
    "rabbitmq":     {"canonical": "rabbitmq",     "group": "message_queue",     "adjacent": ["kafka", "celery"], "parents": ["distributed systems", "message queuing"], "importance": 4},
    "sqs":          {"canonical": "sqs",          "group": "message_queue",     "adjacent": ["kafka", "sns"], "parents": ["distributed systems", "aws"], "importance": 3},
    "celery":       {"canonical": "celery",       "group": "task_queue",        "adjacent": ["rabbitmq", "redis"], "parents": ["distributed systems", "task processing"], "importance": 3},
    # Infrastructure — parent: "cloud infrastructure"
    "docker":       {"canonical": "docker",       "group": "containerization",  "adjacent": ["kubernetes", "containers"], "parents": ["cloud infrastructure", "containerization"], "importance": 5},
    "kubernetes":   {"canonical": "kubernetes",   "group": "orchestration",     "adjacent": ["docker", "k8s", "eks", "gke"], "parents": ["cloud infrastructure", "container orchestration"], "importance": 5},
    "k8s":          {"canonical": "kubernetes",   "group": "orchestration",     "adjacent": ["docker", "kubernetes"], "parents": ["cloud infrastructure", "container orchestration"], "importance": 5},
    "terraform":    {"canonical": "terraform",    "group": "iac",              "adjacent": ["pulumi", "cloudformation", "ansible"], "parents": ["cloud infrastructure", "infrastructure as code"], "importance": 4},
    "ansible":      {"canonical": "ansible",      "group": "iac",              "adjacent": ["terraform"], "parents": ["cloud infrastructure", "configuration management"], "importance": 3},
    # Cloud — parent: "cloud computing"
    "aws":          {"canonical": "aws",          "group": "cloud",            "adjacent": ["gcp", "azure", "eks", "s3", "lambda"], "parents": ["cloud computing"], "importance": 5},
    "gcp":          {"canonical": "gcp",          "group": "cloud",            "adjacent": ["aws", "azure", "gke"], "parents": ["cloud computing"], "importance": 4},
    "azure":        {"canonical": "azure",        "group": "cloud",            "adjacent": ["aws", "gcp"], "parents": ["cloud computing"], "importance": 4},
    "eks":          {"canonical": "eks",          "group": "managed_k8s",       "adjacent": ["kubernetes", "aws"], "parents": ["cloud computing", "aws", "container orchestration"], "importance": 4},
    "ecs":          {"canonical": "ecs",          "group": "container_service", "adjacent": ["docker", "aws"], "parents": ["cloud computing", "aws"], "importance": 3},
    "lambda":       {"canonical": "lambda",       "group": "serverless",        "adjacent": ["aws"], "parents": ["cloud computing", "serverless computing"], "importance": 3},
    "s3":           {"canonical": "s3",           "group": "object_storage",    "adjacent": ["aws"], "parents": ["cloud computing", "data storage"], "importance": 3},
    # APIs — parent: "api design"
    "rest":         {"canonical": "rest_api",     "group": "api_protocol",      "adjacent": ["restful", "rest apis"], "parents": ["api design", "web services"], "importance": 4},
    "rest apis":    {"canonical": "rest_api",     "group": "api_protocol",      "adjacent": ["rest"], "parents": ["api design", "web services"], "importance": 4},
    "graphql":      {"canonical": "graphql",      "group": "api_protocol",      "adjacent": ["rest"], "parents": ["api design", "web services"], "importance": 3},
    "grpc":         {"canonical": "grpc",         "group": "api_protocol",      "adjacent": ["protobuf"], "parents": ["api design", "microservices"], "importance": 3},
    # Data/ETL — parent: "data engineering"
    "etl":          {"canonical": "etl",          "group": "data_pipeline",     "adjacent": ["airflow", "spark", "data pipeline", "data ingestion"], "parents": ["data engineering"], "importance": 5},
    "data pipeline":{"canonical": "data_pipeline","group": "data_pipeline",     "adjacent": ["etl", "airflow"], "parents": ["data engineering"], "importance": 5},
    "data ingestion":{"canonical": "data_pipeline","group": "data_pipeline",    "adjacent": ["etl", "data pipeline"], "parents": ["data engineering"], "importance": 5},
    "airflow":      {"canonical": "airflow",      "group": "data_pipeline",     "adjacent": ["etl", "spark"], "parents": ["data engineering", "workflow orchestration"], "importance": 4},
    "spark":        {"canonical": "spark",        "group": "data_pipeline",     "adjacent": ["etl", "airflow", "hadoop"], "parents": ["data engineering", "big data"], "importance": 4},
    # CI/CD & Observability — parent: "devops"
    "ci/cd":        {"canonical": "ci_cd",        "group": "devops",           "adjacent": ["jenkins", "github actions"], "parents": ["devops", "software delivery"], "importance": 4},
    "jenkins":      {"canonical": "jenkins",      "group": "ci_cd_tool",        "adjacent": ["ci/cd", "github actions"], "parents": ["devops", "ci/cd"], "importance": 3},
    "github actions":{"canonical": "github_actions","group": "ci_cd_tool",      "adjacent": ["ci/cd", "jenkins"], "parents": ["devops", "ci/cd"], "importance": 3},
    "argocd":       {"canonical": "argocd",       "group": "ci_cd_tool",        "adjacent": ["ci/cd", "kubernetes"], "parents": ["devops", "gitops"], "importance": 3},
    "prometheus":   {"canonical": "prometheus",   "group": "observability",     "adjacent": ["grafana", "datadog"], "parents": ["devops", "monitoring"], "importance": 3},
    "grafana":      {"canonical": "grafana",      "group": "observability",     "adjacent": ["prometheus"], "parents": ["devops", "monitoring"], "importance": 3},
    "datadog":      {"canonical": "datadog",      "group": "observability",     "adjacent": ["prometheus", "grafana"], "parents": ["devops", "monitoring"], "importance": 3},
    # Architecture Concepts — parent: "software architecture"
    "microservices":{"canonical": "microservices","group": "architecture",      "adjacent": ["distributed systems"], "parents": ["software architecture"], "importance": 4},
    "distributed systems":{"canonical": "distributed_systems","group": "architecture","adjacent": ["microservices"], "parents": ["software architecture"], "importance": 4},
    "event-driven": {"canonical": "event_driven", "group": "architecture",      "adjacent": ["kafka", "message queue"], "parents": ["software architecture", "distributed systems"], "importance": 3},
    # ML — parent: "artificial intelligence"
    "pytorch":      {"canonical": "pytorch",      "group": "ml_framework",      "adjacent": ["tensorflow", "deep learning"], "parents": ["artificial intelligence", "deep learning"], "importance": 4},
    "tensorflow":   {"canonical": "tensorflow",   "group": "ml_framework",      "adjacent": ["pytorch"], "parents": ["artificial intelligence", "deep learning"], "importance": 4},
    "scikit-learn": {"canonical": "scikit_learn",  "group": "ml_framework",     "adjacent": ["pandas"], "parents": ["artificial intelligence", "machine learning"], "importance": 3},
    "machine learning":{"canonical": "machine_learning","group": "ml",          "adjacent": ["deep learning", "pytorch", "tensorflow"], "parents": ["artificial intelligence"], "importance": 4},
    "deep learning":{"canonical": "deep_learning","group": "ml",               "adjacent": ["pytorch", "tensorflow"], "parents": ["artificial intelligence", "machine learning"], "importance": 4},
    "nlp":          {"canonical": "nlp",          "group": "ml",               "adjacent": ["machine learning"], "parents": ["artificial intelligence", "machine learning"], "importance": 4},
}


def extract_skills_ontology(text: str) -> Set[str]:
    """Extract canonical skills from text using ontology lookup."""
    text_lower = text.lower()
    found = set()
    sorted_skills = sorted(SKILL_ONTOLOGY.keys(), key=len, reverse=True)
    for surface_form in sorted_skills:
        pattern = r'\b' + re.escape(surface_form) + r'\b'
        if re.search(pattern, text_lower):
            found.add(SKILL_ONTOLOGY[surface_form]["canonical"])
    return found


def expand_with_hierarchy(skills: Set[str]) -> Set[str]:
    """
    Graph expansion: for each extracted skill, walk up the ontology hierarchy
    and append parent concepts. This enriches the profile before retrieval.

    Example: {"pytorch"} → {"pytorch", "artificial intelligence", "deep learning"}

    Used to expand resume text before bi-encoder embedding, so the semantic
    representation captures the broader competency area even if the resume
    only mentions a specific tool.
    """
    expanded = set(skills)
    for skill in skills:
        for surface, meta in SKILL_ONTOLOGY.items():
            if meta["canonical"] == skill:
                for parent in meta.get("parents", []):
                    expanded.add(parent)
                break
    return expanded


def get_skill_importance(canonical_skill: str) -> float:
    """
    Get O*NET-style importance weight (1-5) for a canonical skill.
    Returns normalized weight (0.2 to 1.0) for use in scoring.

    Importance 5 (critical)    → weight 1.0
    Importance 4 (important)   → weight 0.8
    Importance 3 (moderate)    → weight 0.6
    Importance 2 (minor)       → weight 0.4
    Importance 1 (nice-to-have)→ weight 0.2

    Production upgrade: load importance scores from O*NET database CSV
    per occupation, rather than static ontology values.
    """
    for surface, meta in SKILL_ONTOLOGY.items():
        if meta["canonical"] == canonical_skill:
            return meta.get("importance", 3) / 5.0
    return 0.6  # default: moderate importance


def compute_skill_overlap(jd_skills: Set[str], resume_skills: Set[str]) -> dict:
    """
    Three-tier skill overlap: exact (1.0) → adjacent (0.6) → group (0.3).
    Returns detailed match info for explainability.
    """
    if not jd_skills:
        return {"exact_ratio": 0.0, "adjacent_ratio": 0.0, "group_ratio": 0.0,
                "combined": 0.0, "matched": [], "adjacent_matched": [],
                "group_matched": [], "missing": []}

    exact = jd_skills & resume_skills
    missing = jd_skills - resume_skills

    adjacent_matches = set()
    for skill in list(missing):
        for surface, meta in SKILL_ONTOLOGY.items():
            if meta["canonical"] == skill:
                adj_canonicals = set()
                for adj in meta.get("adjacent", []):
                    if adj in SKILL_ONTOLOGY:
                        adj_canonicals.add(SKILL_ONTOLOGY[adj]["canonical"])
                if adj_canonicals & resume_skills:
                    adjacent_matches.add(skill)
                break

    group_matches = set()
    for skill in (missing - adjacent_matches):
        skill_group = None
        for surface, meta in SKILL_ONTOLOGY.items():
            if meta["canonical"] == skill:
                skill_group = meta["group"]
                break
        if skill_group:
            for rs in resume_skills:
                for surface, meta in SKILL_ONTOLOGY.items():
                    if meta["canonical"] == rs and meta["group"] == skill_group:
                        group_matches.add(skill)
                        break

    n = len(jd_skills)
    exact_r = len(exact) / n
    adj_r = len(adjacent_matches) / n
    grp_r = len(group_matches) / n
    combined = min(exact_r + 0.6 * adj_r + 0.3 * grp_r, 1.0)

    truly_missing = missing - adjacent_matches - group_matches

    return {
        "exact_ratio": round(exact_r, 4),
        "adjacent_ratio": round(adj_r, 4),
        "group_ratio": round(grp_r, 4),
        "combined": round(combined, 4),
        "matched": sorted(exact),
        "adjacent_matched": sorted(adjacent_matches),
        "group_matched": sorted(group_matches),
        "missing": sorted(truly_missing),
    }
# === LLM-based structured extraction (richer than ontology alone) ===


EXTRACTION_PROMPT = """Extract structured information from this resume/job description text.
Return ONLY valid JSON — no markdown fences, no preamble:
{
  "technical_skills": ["list of specific technical skills, tools, languages, frameworks"],
  "soft_skills": ["list of soft skills"],
  "certifications": ["list of certifications like AWS SAA, CISSP, PMP"],
  "experience_years": <integer of total years, or 0>,
  "education": ["list of degrees"],
  "domain": "primary industry domain",
  "seniority": "junior|mid|senior|lead|executive"
}
Map synonyms to canonical forms (e.g., "data ingestion workflows" → "ETL pipelines").
TEXT:
"""


def parse_llm_json(raw: str) -> Optional[dict]:
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def extract_skills_llm(text: str) -> Optional[dict]:
    """LLM-based structured extraction — OpenAI only."""
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE,
                messages=[
                    {"role": "system", "content": "Extract structured data. Return ONLY valid JSON."},
                    {"role": "user", "content": EXTRACTION_PROMPT + text[:4000]}
                ],
                response_format={"type": "json_object"}
            )
            return parse_llm_json(resp.choices[0].message.content)
        except Exception as e:
            print(f"  OpenAI extraction failed: {e}")

    return None


def extract_skills_regex(text: str) -> dict:
    """Fallback regex extraction when no LLM is available."""
    text_lower = text.lower()
    tech = set()
    for pattern in [
        r'\b(python|java|javascript|typescript|go|rust|c\+\+|ruby|scala|kotlin)\b',
        r'\b(react|angular|vue|django|flask|fastapi|spring|node\.?js)\b',
        r'\b(aws|gcp|azure|docker|kubernetes|k8s|terraform)\b',
        r'\b(postgresql|mysql|mongodb|redis|kafka|rabbitmq|elasticsearch)\b',
        r'\b(etl|data\s*pipeline|data\s*ingestion|airflow|spark)\b',
        r'\b(graphql|grpc|microservice|distributed\s*system)\b',
    ]:
        tech.update(re.findall(pattern, text_lower))

    years = re.findall(r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)', text_lower)
    certs = re.findall(r'\b(aws\s*(?:saa|sap|dva)|cissp|pmp|cka|ckad)\b', text_lower)

    return {
        "technical_skills": sorted(tech),
        "soft_skills": [],
        "certifications": sorted(set(certs)),
        "experience_years": max([int(y) for y in years], default=0),
        "education": [],
        "domain": "unknown",
        "seniority": "unknown",
    }


def extract_skills_structured(text: str) -> dict:
    """
    Extract structured skills + occupations.
    ESCO provides standardized taxonomy mapping (13K+ skills, 3K occupations).
    LLM provides structured profile (years, domain, seniority).
    Both are combined when available.
    """
    # Try ESCO taxonomy mapping
    esco_skills = extract_skills_esco(text)
    esco_occupations = extract_occupations_esco(text)

    # Try LLM for structured profile
    result = extract_skills_llm(text)
    if result is None:
        result = extract_skills_regex(text)

    # Merge ESCO data into result if available
    if esco_skills:
        result["esco_skill_uris"] = esco_skills
    if esco_occupations:
        result["esco_occupation_uris"] = esco_occupations

    return result
