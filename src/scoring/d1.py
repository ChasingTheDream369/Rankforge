"""D1 (Skills) — ontology first (exact/adjacent/group), LLM tool fallback when absent."""

import json
from typing import Dict, Optional

from src.config import D1_LLM_FALLBACK, EXTRACTION_MODEL
from src.scoring.llm_client import parse_json, has_llm

_D1_EXACT = {"BUILT_WITH": 1.0, "USED": 0.7, "LISTED": 0.3}
_D1_ADJACENT = {"BUILT_WITH": 0.6, "USED": 0.5, "LISTED": 0.2}
_D1_GROUP = {"BUILT_WITH": 0.3, "USED": 0.25, "LISTED": 0.1}

_D1_FALLBACK_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_skill_fit_assessment",
        "description": "Submit your assessment. You MUST call this with the score after evaluating the resume against the JD skill.",
        "parameters": {
            "type": "object",
            "properties": {"score": {"type": "number", "description": "0-1: 1.0=strong evidence, 0.5=partial, 0=no relevance"}},
            "required": ["score"],
        },
    },
}


def _get_ontology_adjacent_canonicals(canonical_skill: str) -> set:
    try:
        from src.ingestion.ontology import SKILL_ONTOLOGY
    except Exception:
        return set()
    adj = set()
    for surface, meta in SKILL_ONTOLOGY.items():
        if meta.get("canonical") == canonical_skill:
            for a in meta.get("adjacent", []):
                if a in SKILL_ONTOLOGY:
                    c = SKILL_ONTOLOGY[a]["canonical"]
                    if c != canonical_skill:
                        adj.add(c)
            break
    return adj


def _get_ontology_group(canonical_skill: str) -> Optional[str]:
    try:
        from src.ingestion.ontology import SKILL_ONTOLOGY
    except Exception:
        return None
    for surface, meta in SKILL_ONTOLOGY.items():
        if meta.get("canonical") == canonical_skill:
            return meta.get("group")
    return None


def _get_ontology_skills_in_group(group_name: str) -> set:
    try:
        from src.ingestion.ontology import SKILL_ONTOLOGY
    except Exception:
        return set()
    return {m["canonical"] for m in SKILL_ONTOLOGY.values() if m.get("group") == group_name}


def _call_d1_skill_fit_tool(skill_name: str, resume_profile: dict) -> float:
    if not has_llm():
        return 0.0
    skills_summary = ", ".join(
        ((s.get("name", "") or "") + ": " + ((s.get("evidence", "") or "")[:60]))
        for s in (resume_profile.get("skills") or [])[:12]
    )
    highlights = " | ".join((resume_profile.get("highlights") or [])[:5])
    context = f"Skills: {skills_summary[:400]}. Highlights: {highlights[:300]}"
    prompt = f"""JD requires skill: {skill_name}. Resume context: {context}
Assess fit 0.0-1.0. You MUST call submit_skill_fit_assessment(score) with your assessment."""
    try:
        return _call_d1_fallback_openai(prompt)
    except Exception:
        return 0.0


def _call_d1_fallback_openai(prompt: str) -> float:
    from openai import OpenAI
    from src.config import OPENAI_API_KEY
    client = OpenAI(api_key=OPENAI_API_KEY)
    msgs = [{"role": "system", "content": "Assess skill fit. You MUST call submit_skill_fit_assessment with your score."}, {"role": "user", "content": prompt}]
    for _ in range(3):
        resp = client.chat.completions.create(
            model=EXTRACTION_MODEL, messages=msgs, max_tokens=100, temperature=0,
            tools=[_D1_FALLBACK_TOOL], tool_choice={"type": "function", "function": {"name": "submit_skill_fit_assessment"}},
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == "submit_skill_fit_assessment":
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                        s = float(args.get("score", 0))
                        return max(0.0, min(1.0, s))
                    except Exception:
                        pass
        if msg.content:
            parsed = parse_json(msg.content)
            if parsed and "score" in parsed:
                return max(0.0, min(1.0, float(parsed.get("score", 0))))
    return 0.0


def compute_d1_from_profiles(jd_profile: dict, resume_profile: dict, use_llm_fallback: bool = False) -> tuple:
    jd_skills = jd_profile.get("required_skills") or []
    resume_skills = resume_profile.get("skills") or []

    resume_lookup: Dict[str, tuple] = {}
    for s in resume_skills:
        name = (s.get("name") or "").lower().strip()
        if name:
            level = str(s.get("level") or "LISTED").upper()
            if level not in ("BUILT_WITH", "USED", "LISTED"):
                level = "LISTED"
            evidence = str(s.get("evidence", "")).strip()[:200]
            resume_lookup[name] = (level, evidence)

    resume_names = set(resume_lookup.keys())
    checked = []
    weighted_sum = 0.0
    weight_total = 0.0
    core_matched = 0
    core_total = 0

    for js in jd_skills[:15]:
        name = (js.get("name") or "").lower().strip()
        if not name:
            continue
        imp = str(js.get("importance") or "nice").lower()
        is_core = imp == "core"
        w = 2.0 if is_core else 1.0
        weight_total += w
        if is_core:
            core_total += 1

        if name in resume_lookup:
            level, evidence = resume_lookup[name]
            score = _D1_EXACT.get(level, 0.3)
            weighted_sum += score * w
            if is_core and score > 0:
                core_matched += 1
            checked.append({"jd_skill": name, "resume_match": name, "match_type": "exact", "level": level, "score": score, "evidence": evidence, "importance": "core" if is_core else "nice"})
            continue

        adj = _get_ontology_adjacent_canonicals(name)
        adj_match = adj & resume_names
        if adj_match:
            match_name = next(iter(adj_match))
            level, evidence = resume_lookup[match_name]
            score = _D1_ADJACENT.get(level, 0.2)
            weighted_sum += score * w
            if is_core and score > 0:
                core_matched += 1
            checked.append({"jd_skill": name, "resume_match": match_name, "match_type": "adjacent", "level": level, "score": score, "evidence": evidence, "importance": "core" if is_core else "nice"})
            continue

        jd_group = _get_ontology_group(name)
        if jd_group:
            group_skills = _get_ontology_skills_in_group(jd_group)
            group_match = (group_skills & resume_names) - {name}
            if group_match:
                match_name = next(iter(group_match))
                level, evidence = resume_lookup[match_name]
                score = _D1_GROUP.get(level, 0.1)
                weighted_sum += score * w
                if is_core and score > 0:
                    core_matched += 1
                checked.append({"jd_skill": name, "resume_match": match_name, "match_type": "group", "level": level, "score": score, "evidence": evidence, "importance": "core" if is_core else "nice"})
                continue

        score = 0.0
        match_type = "absent"
        if use_llm_fallback and has_llm():
            try:
                score = _call_d1_skill_fit_tool(name, resume_profile)
                if score > 0:
                    match_type = "llm_fallback"
                    weighted_sum += score * w
                    if is_core and score > 0:
                        core_matched += 1
            except Exception:
                pass
        checked.append({"jd_skill": name, "resume_match": None, "match_type": match_type, "level": None, "score": score, "evidence": "", "importance": "core" if is_core else "nice"})

    d1 = round(weighted_sum / weight_total, 4) if weight_total > 0 else 0.0
    core_coverage = round(core_matched / core_total, 4) if core_total > 0 else 0.0
    missing_core = [c["jd_skill"] for c in checked if c.get("importance") == "core" and c.get("score", 0) == 0]
    matched = [c["jd_skill"] for c in checked if c.get("match_type") == "exact"]
    adjacent_matched = [c["jd_skill"] for c in checked if c.get("match_type") == "adjacent"]
    group_matched = [c["jd_skill"] for c in checked if c.get("match_type") == "group"]
    missing = [c["jd_skill"] for c in checked if c.get("match_type") == "absent"]

    skill_detail = {
        "skills_checked": checked, "d1_breakdown": checked, "core_coverage": core_coverage,
        "missing_core": missing_core, "matched": matched, "adjacent_matched": adjacent_matched,
        "group_matched": group_matched, "missing": missing,
    }
    return d1, skill_detail
