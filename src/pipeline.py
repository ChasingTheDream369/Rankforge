"""
Pipeline Orchestrator — single entry point for the full matching system.

Flow:
  1. Sanitize resumes against adversarial attacks
  2. Stage 1 LLM — extract ResumeProfiles (gpt-4o-mini, one call per resume, cached)
  3. Stage 1 LLM — extract JDProfile (gpt-4o-mini, always fresh)
  4. Build retrieval index + hybrid search — BM25 + bi-encoder + RRF + CE
  5. Stage 2 LLM — score each candidate (gpt-4o, few-shot, temp=0, one call per resume)
  6. Agentic retry for any LOW-confidence results (bounded 1 extra call)

Cache: resume profiles stored in IndexStore — skipped entirely on second run with same resumes.
"""

import json
import os
from typing import Dict, List, Optional

from src.config import (
    JD_DIR, RESUME_DIR, GOLDEN_DATASET_PATH, LLM_PROVIDER,
)
from src.contracts import MatchResult, ThreatReport
from src.ingestion.extractor import extract_directory, extract_text
from src.ingestion.sanitizer import sanitize
from src.retrieval.engine import RetrievalEngine
from src.retrieval.index_store import IndexStore
from src.scoring.explainability import generate_rationale
from src.scoring.scorer import extract_jd_profile, extract_resume_profile


def run_pipeline(
    jd_text: str,
    resumes: Dict[str, dict],
    verbose: bool = True,
) -> List[MatchResult]:
    """
    Full pipeline: sanitize → extract profiles → retrieve → score → rank.

    Resume profiles and embeddings are cached in IndexStore.
    On second run with a different JD, both load from cache —
    no re-embedding, no re-extraction. Only the JD profile is re-processed.
    """

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  RESUME-JD MATCHING ENGINE v2 (LLM: {LLM_PROVIDER})")
        print(f"{'=' * 70}")

    # --- Check index + profile cache ---
    store = IndexStore()
    sanitized_texts = {rid: r["text"] for rid, r in resumes.items()}
    cache_hit = store.is_valid(sanitized_texts)

    if cache_hit:
        store.load()
        if verbose:
            print(f"\n  [CACHE HIT] Loaded {len(store.doc_ids)} resumes from index cache")
            print(f"  Skipping sanitization, embedding, and profile extraction for resumes")

        sanitized = dict(zip(store.doc_ids, store.doc_texts))
        threat_reports = {rid: ThreatReport(resume_id=rid) for rid in store.doc_ids}
        resume_profiles = store.skills_cache  # {rid: profile_dict}

    else:
        if verbose:
            print(f"\n  [1/5] Sanitizing {len(resumes)} resumes...")
        sanitized = {}
        threat_reports = {}
        for rid, rdata in resumes.items():
            cleaned, report = sanitize(rdata["text"], jd_text, rid)
            sanitized[rid] = cleaned
            threat_reports[rid] = report
            if not report.is_clean and verbose:
                print(f"    ⚠ {rid}: {report.threat_level} ({len(report.flags)} flags)")

        # Stage 1a: extract resume profiles (gpt-4o-mini, one call per resume)
        resume_profiles = {}
        if LLM_PROVIDER == "openai":
            if verbose:
                print(f"\n  [2/5] Extracting {len(sanitized)} resume profiles (gpt-4o-mini)...")
            for rid, text in sanitized.items():
                profile = extract_resume_profile(text)
                if profile:
                    resume_profiles[rid] = profile
                    if verbose:
                        yrs = profile.get("total_years", "?")
                        doms = profile.get("domains", [])
                        nsk = len(profile.get("skills", []))
                        print(f"    {rid}: {yrs}yr | {doms} | {nsk} skills")
                elif verbose:
                    print(f"    {rid}: extraction failed — deterministic fallback")
        else:
            if verbose:
                print(f"\n  [2/5] No API key — skipping LLM profile extraction")

        # Build and save index + profiles for future runs
        if verbose:
            print(f"\n  [3/5] Building retrieval index and caching to disk...")
        store.build(sanitized)
        store.skills_cache = resume_profiles
        store.save()

    # --- Stage 1b: JD profile — always fresh ---
    jd_profile = None
    if LLM_PROVIDER == "openai":
        if verbose:
            print(f"\n  [3/5] Extracting JD profile (gpt-4o-mini)...")
        jd_profile = extract_jd_profile(jd_text)
        if jd_profile and verbose:
            print(f"    Domain: {jd_profile.get('domain','?')} | "
                  f"Seniority: {jd_profile.get('seniority','?')} | "
                  f"{len(jd_profile.get('required_skills',[]))} skills | "
                  f"{jd_profile.get('years_required',0)}+ yrs required")

    # --- Retrieval ---
    if verbose:
        print(f"\n  [4/5] Building retrieval index...")

    engine = RetrievalEngine()
    engine.index(sanitized)
    retrieval_results = engine.search(jd_text)
    retrieval_order = [doc_id for doc_id, _ in retrieval_results]
    retrieval_scores = {doc_id: score for doc_id, score in retrieval_results}

    if verbose:
        print(f"    Retrieved and ranked {len(retrieval_results)} candidates")

    # --- Stage 2: score each candidate (gpt-4o, few-shot, temp=0) ---
    from src.scoring.scorer import score_resume as agent_score
    from src.contracts import MatchRationale

    candidates = [rid for rid in retrieval_order if rid in sanitized]
    n_cands = len(candidates)

    if verbose:
        print(f"\n  [5/5] Scoring {n_cands} candidates (gpt-4o, few-shot, temp=0)...")

    results = []
    for rid in candidates:
        resume_text = sanitized[rid]
        threat = threat_reports.get(rid, ThreatReport(resume_id=rid))
        ce_logit = engine.get_cross_encoder_logit(rid)
        penalty = threat.total_penalty if threat.total_penalty > 0 else 0.0

        scored = agent_score(
            jd_text=jd_text,
            resume_text=resume_text,
            ce_logit=ce_logit,
            n_candidates=n_cands,
            adversarial_penalty=penalty,
            verbose=verbose,
            jd_profile=jd_profile,
            resume_profile=resume_profiles.get(rid),
        )

        rationale = MatchRationale(
            summary=scored.get("rationale", ""),
            recommendation=scored["recommendation"],
            strengths=scored.get("strengths", []),
            gaps=scored.get("gaps", []),
            flags=threat.flags if not threat.is_clean else [],
        )

        stage_scores = engine.get_stage_scores(jd_text, rid)
        stage_scores["rrf"] = round(retrieval_scores.get(rid, 0), 4)
        stage_scores["ce_logit"] = round(ce_logit, 4)
        stage_scores["ce_sigmoid"] = scored["ce_sigmoid"]
        stage_scores["d1_skills"] = scored["d1_skills"]
        stage_scores["d2_seniority"] = scored["d2_seniority"]
        stage_scores["d3_domain"] = scored["d3_domain"]
        stage_scores["d4_constraints"] = scored["d4_constraints"]
        stage_scores["dim_composite"] = scored["dim_composite"]
        stage_scores["ce_weight"] = scored["ce_weight"]
        stage_scores["scoring_mode"] = scored["mode"]

        results.append(MatchResult(
            resume_id=rid,
            name=resumes[rid].get("name", rid),
            final_score=scored["final_score"],
            confidence=scored["confidence"],
            rationale=rationale,
            threat_report=threat,
            skill_detail=scored.get("skill_detail", {}),
            stage_scores=stage_scores,
            label=resumes[rid].get("label", -1.0),
        ))

    results.sort(key=lambda r: r.final_score, reverse=True)
    for rank, r in enumerate(results, 1):
        r.rank = rank

    if verbose:
        print(f"\n  [Done] {len(results)} candidates ranked.")

    return results


def load_sample_data(jd_dir: str = JD_DIR, resume_dir: str = RESUME_DIR) -> tuple:
    """Load JD and resumes from data directories."""
    jd_files = {}
    if os.path.isdir(jd_dir):
        for f in sorted(os.listdir(jd_dir)):
            if f.endswith('.txt'):
                with open(os.path.join(jd_dir, f), 'r') as fh:
                    jd_files[f.replace('.txt', '')] = fh.read()

    resume_files = {}
    if os.path.isdir(resume_dir):
        for f in sorted(os.listdir(resume_dir)):
            if f.endswith('.txt'):
                rid = f.replace('.txt', '')
                with open(os.path.join(resume_dir, f), 'r') as fh:
                    name = rid.replace('_', ' ').title()
                    resume_files[rid] = {"text": fh.read(), "name": name}

    if os.path.exists(GOLDEN_DATASET_PATH):
        with open(GOLDEN_DATASET_PATH, 'r') as f:
            content = f.read().strip()
        # Support both single JSON object and JSONL (multiple objects)
        golden = {}
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(content):
            content_slice = content[pos:].lstrip()
            if not content_slice:
                break
            # Skip comma separators between objects (non-standard but common)
            if content_slice.startswith(','):
                pos += len(content[pos:]) - len(content_slice) + 1
                continue
            skip = len(content[pos:]) - len(content_slice)
            try:
                obj, end = decoder.raw_decode(content_slice)
            except Exception:
                break
            golden.update(obj)
            pos += skip + end
        for jd_id, labels in golden.items():
            for rid, label in labels.items():
                if rid in resume_files:
                    resume_files[rid]["label"] = label

    return jd_files, resume_files
