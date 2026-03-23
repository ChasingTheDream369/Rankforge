"""Deterministic scorer — regex/keyword fallback when no LLM. Uses ADJACENT_DOMAINS from extraction_schema."""

import re

from src.scoring.extraction_schema import ADJACENT_DOMAINS

BUILD_VERBS = {
    "built", "architected", "designed", "engineered", "developed",
    "implemented", "created", "deployed", "launched", "shipped",
    "automated", "migrated", "scaled", "optimized", "led",
    "managed", "orchestrated", "delivered", "integrated",
}

USE_VERBS = {
    "used", "utilized", "worked with", "contributed",
    "maintained", "supported", "configured", "applied", "leveraged",
}

ALIAS_MAP = {
    "rag": ["retrieval augmented generation", "rag pipeline", "llamaindex", "langchain", "faiss"],
    "llm": ["large language model", "gpt", "openai"],
    "python": ["django", "flask", "fastapi"],
    "typescript": ["javascript", "react", "node", "angular"],
    "docker": ["container", "containerized", "docker compose"],
    "kubernetes": ["k8s", "orchestration"],
    "postgresql": ["postgres", "sql"],
    "kafka": ["event-driven", "message queue", "rabbitmq"],
    "ci/cd": ["github actions", "jenkins", "continuous integration"],
    "microservices": ["micro-service", "service-oriented"],
    "aws": ["ec2", "s3", "lambda", "ecs", "dynamodb", "sqs"],
}

DOMAIN_KEYWORDS = {
    "fintech": ["fintech", "payment", "banking", "financial", "trading", "lending"],
    "enterprise_saas": ["enterprise", "saas", "b2b", "multi-tenant"],
    "ai_ml": ["ai", "machine learning", "llm", "gpt", "genai", "nlp", "rag"],
    "healthcare": ["healthcare", "medical", "clinical", "patient", "hipaa"],
    "platform_devops": ["platform", "devops", "kubernetes", "ci/cd", "infrastructure"],
    "ecommerce": ["ecommerce", "marketplace", "retail"],
}


def _expand_terms(skill_text: str):
    terms = set()
    skill_lower = skill_text.lower().strip()
    terms.add(skill_lower)
    for w in re.split(r'\W+', skill_lower):
        if len(w) > 2:
            terms.add(w)
    expanded = set()
    for t in terms:
        for key, aliases in ALIAS_MAP.items():
            if t in key or key in t or any(a in t or t in a for a in aliases):
                expanded.add(key)
                expanded.update(aliases)
    terms.update(expanded)
    return {t for t in terms if len(t) > 2}


def find_skill_evidence(skill_name: str, resume_text: str):
    terms = _expand_terms(skill_name)
    sentences = re.split(r'[.\n•]+', resume_text)
    matches = []
    for sent in sentences:
        sl = sent.lower().strip()
        if not sl or len(sl) < 10:
            continue
        if any(t in sl for t in terms):
            matches.append(sent.strip())
    if not matches:
        return "ABSENT", 0.0, ""
    for sent in matches:
        if any(v in sent.lower() for v in BUILD_VERBS):
            return "BUILT_WITH", 1.0, sent[:200]
    for sent in matches:
        if any(v in sent.lower() for v in USE_VERBS):
            return "USED", 0.7, sent[:200]
    return "LISTED", 0.3, matches[0][:200]


def assess_seniority(resume_text: str, jd_years: int = 0):
    rl = resume_text.lower()
    signals = {}
    lead = 0.0
    if re.search(r'led\s+(?:a\s+)?(?:team|development)|manag(?:ed|ing)\s+(?:a\s+)?\d+|mentor', rl):
        lead = 0.7
    m = re.search(r'(\d+)[\s-]*(engineer|person|member|people)\s*(team|group)', rl)
    if m:
        lead = max(lead, min(1.0, int(m.group(1)) / 6))
    signals["leadership"] = lead

    arch = 0.0
    for pat in [r'architect(?:ed|ing|ure)', r'design(?:ed|ing)\s+(?:a\s+)?(?:system|platform|pipeline)',
                r'technical\s+(?:architecture|decisions)', r'end[\s-]to[\s-]end']:
        if re.search(pat, rl):
            arch = max(arch, 0.9)
    signals["architecture"] = min(1.0, arch)

    scale = 0.0
    for pat, w in [(r'(\d[\d,]+)\+?\s*(?:active\s+)?users', 0.8),
                   (r'\d+[Mm]\s*(?:users|DAU)?', 0.85), (r'(\d[\d.]*)\s*[Mm]illion', 0.85),
                   (r'(\d[\d,]+)\+?\s*tps', 0.9), (r'\$[\d.]+[BMKbmk]', 0.9)]:
        if re.search(pat, rl):
            scale = max(scale, w)
    signals["scale"] = min(1.0, scale)

    own = 0.0
    for pat, w in [(r'end[\s-]to[\s-]end\s+(?:ownership|responsibility)', 0.9),
                   (r'from\s+(?:ideation|scratch)\s+(?:through|to)\s+production', 0.9),
                   (r'production\s+deployment', 0.8), (r'\bownership\b|owned\s+(?:the|from)', 0.6)]:
        if re.search(pat, rl):
            own = max(own, w)
    signals["ownership"] = min(1.0, own)

    year_ranges = re.findall(
        r'(20\d{2})\s*[-–—]+\s*(20\d{2}|present|current|now)', resume_text.lower()
    )
    actual = 0
    if year_ranges:
        spans = [(int(s), 2026 if e in ('present', 'current', 'now') else int(e))
                 for s, e in year_ranges]
        actual = max(e for _, e in spans) - min(s for s, _ in spans)
    if jd_years > 0 and actual > 0:
        gap = jd_years - actual
        signals["years"] = 1.0 if gap <= 0 else 0.8 if gap <= 1 else 0.5 if gap <= 2 else 0.2
    else:
        signals["years"] = min(1.0, actual / 5) if actual > 0 else 0.3

    d2 = (0.25 * signals["leadership"] + 0.25 * signals["architecture"] +
          0.20 * signals["scale"] + 0.15 * signals["ownership"] + 0.15 * signals["years"])
    return round(d2, 4), signals


def assess_domain_fit(jd_text: str, resume_text: str):
    def _doms(text):
        tl = text.lower()
        r = []
        for d, kws in DOMAIN_KEYWORDS.items():
            h = sum(1 for kw in kws if kw in tl)
            if h > 0:
                r.append((d, round(min(1.0, h / max(2, len(kws) * 0.3)), 2)))
        r.sort(key=lambda x: x[1], reverse=True)
        return r
    jd_d = _doms(jd_text)
    res_d = _doms(resume_text)
    if not jd_d:
        return 0.5, "No JD domain found"
    if not res_d:
        return 0.3, "No resume domain found"
    jt = set(d for d, _ in jd_d[:2])
    rt = set(d for d, _ in res_d[:2])
    overlap = jt & rt
    if overlap:
        return round(min(1.0, 0.5 + 0.2 * len(overlap)), 4), f"Overlap: {', '.join(overlap)}"
    for j in jt:
        for r in rt:
            if (j, r) in ADJACENT_DOMAINS or (r, j) in ADJACENT_DOMAINS:
                return 0.55, "Adjacent domains"
    return 0.25, f"Different domains: JD={jt} vs Resume={rt}"


def check_constraints(jd_text: str, resume_text: str):
    jl, rl = jd_text.lower(), resume_text.lower()
    checks = []

    ym = re.search(r'(\d+)\+?\s*years?\s*(?:of\s+)?(?:production\s+)?(?:software\s+)?(?:engineering\s+)?experience', jl)
    if ym:
        req = int(ym.group(1))
        yr = re.findall(r'(20\d{2})\s*[-–—]+\s*(20\d{2}|present|current|now)', rl)
        actual = 0
        if yr:
            spans = [(int(s), 2026 if e in ('present', 'current', 'now') else int(e)) for s, e in yr]
            actual = max(e for _, e in spans) - min(s for s, _ in spans)
        checks.append({"req": f"{req}+ yrs exp", "met": actual >= req, "partial": actual >= req - 1})

    LANGS = ["python", "go", "java", "typescript", "javascript", "rust", "scala"]
    jd_langs = [lang for lang in LANGS if lang in jl]
    if jd_langs:
        has_any = any(lang in rl for lang in jd_langs)
        has_all = all(lang in rl for lang in jd_langs)
        checks.append({"req": f"Lang ({'/'.join(jd_langs)})", "met": has_all, "partial": has_any})

    CLOUDS = [("aws", ["ec2", "s3", "lambda", "ecs"]), ("gcp", ["bigquery", "gke"]), ("azure", ["azure", "aks"])]
    for cloud, services in CLOUDS:
        if cloud in jl:
            in_resume = cloud in rl or any(svc in rl for svc in services)
            checks.append({"req": cloud.upper(), "met": in_resume, "partial": in_resume})
            break

    DBS = ["postgresql", "postgres", "mysql", "redis", "mongodb", "cassandra", "dynamodb"]
    jd_dbs = [db for db in DBS if db in jl]
    if jd_dbs:
        has_any = any(db in rl for db in jd_dbs)
        checks.append({"req": f"DB ({', '.join(jd_dbs[:3])})", "met": has_any, "partial": has_any})

    TECH_PATTERNS = [
        ("kafka", ["kafka", "kinesis", "event-driven", "message queue"]),
        ("microservices", ["microservice", "micro-service", "service mesh"]),
        ("docker", ["docker", "container", "kubernetes", "k8s"]),
    ]
    for kw, variants in TECH_PATTERNS:
        if kw in jl or any(v in jl for v in variants):
            found = kw in rl or any(v in rl for v in variants)
            checks.append({"req": kw, "met": found, "partial": found})

    if not checks:
        return 1.0, []
    score = sum(1.0 if c["met"] else (0.5 if c["partial"] else 0.0) for c in checks) / len(checks)
    return round(score, 4), checks


def score_deterministic(jd_text: str, resume_text: str) -> dict:
    reqs = []
    for line in jd_text.split('\n'):
        if line.strip().startswith(('*', '-', '•')):
            req = re.sub(r'^[*\-•]\s*', '', line.strip())
            if len(req) > 10:
                reqs.append(req)
    if not reqs:
        reqs = ["relevant experience"]
    reqs = reqs[:10]

    skills = []
    for req in reqs:
        level, score, evidence = find_skill_evidence(req, resume_text)
        skills.append({"skill": req[:80], "level": level, "score": score, "evidence": evidence[:150]})
    d1 = round(sum(s["score"] for s in skills) / max(len(skills), 1), 4)

    ym = re.search(r'(\d+)\+?\s*years?', jd_text.lower())
    d2, sen_sig = assess_seniority(resume_text, int(ym.group(1)) if ym else 0)
    d3, dom_reason = assess_domain_fit(jd_text, resume_text)
    d4, con_checks = check_constraints(jd_text, resume_text)

    built = sum(1 for s in skills if s["level"] == "BUILT_WITH")
    absent = sum(1 for s in skills if s["level"] == "ABSENT")
    total = max(len(skills), 1)
    conf = ("HIGH" if built / total >= 0.5 and absent / total <= 0.2
            else "LOW" if absent / total >= 0.5 else "MEDIUM")

    return {
        "d1_skills": {"score": d1, "skills_checked": skills},
        "d2_seniority": {"score": d2, "signals": [{"type": k, "score": v} for k, v in sen_sig.items()]},
        "d3_domain": {"score": d3, "reasoning": dom_reason},
        "d4_constraints": {"score": d4, "checks": con_checks},
        "confidence": conf,
        "strengths": [f"{s['skill'][:40]}: {s['level']}" for s in skills if s["level"] in ("BUILT_WITH", "USED")][:5],
        "gaps": [f"{s['skill'][:40]}: {s['level']}" for s in skills if s["level"] in ("ABSENT", "LISTED")][:4],
        "rationale": f"Skills: {d1:.0%}, Seniority: {d2:.0%}, Domain: {d3:.0%}, Constraints: {d4:.0%}",
    }
