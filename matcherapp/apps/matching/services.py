import hashlib
import sys
from typing import Optional, Tuple

from django.conf import settings
from django.utils import timezone

_project_root = str(settings.BASE_DIR)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)



def _extract_text(resume):
    """Extract and persist raw text from an uploaded resume file."""
    from src.ingestion.extractor import extract_text

    text = resume.raw_text
    if text and len(text) >= 100:
        return text

    if not resume.file:
        return ""

    try:
        text = extract_text(resume.file.path)
    except Exception:
        with open(resume.file.path, 'r', errors='ignore') as fh:
            text = fh.read()

    resume.raw_text = text
    resume.save(update_fields=['raw_text'])
    return text


def custom_dim_weights_tuple(match_run) -> Optional[Tuple[float, float, float, float]]:
    """If this run used JD-specific dimension weights, return (w1..w4); else None."""
    cfg = getattr(match_run, "scoring_config", None) or {}
    if not cfg.get("custom_dims"):
        return None
    w = cfg.get("weights")
    if isinstance(w, (list, tuple)) and len(w) == 4:
        try:
            return tuple(max(0.0, float(x)) for x in w)
        except (TypeError, ValueError):
            return None
    return None


def _score_rejected(reason: str) -> dict:
    """Return a zero-score sentinel for resumes that fail pre-flight checks."""
    return {
        'final_score': 0.0,
        'confidence': 'LOW',
        'recommendation': 'NO_MATCH',
        'd1_skills': 0.0,
        'd2_seniority': 0.0,
        'd3_domain': 0.0,
        'd4_constraints': 0.0,
        'dim_composite': 0.0,
        'ce_sigmoid': 0.5,
        'ce_weight': 0.0,
        'mode': 'rejected',
        'strengths': [],
        'gaps': [reason],
        'rationale': reason,
        'skill_detail': {},
    }


def _score_single_resume(
    jd_text, cleaned, ce_logit, threat, n_candidates,
    jd_profile, custom_weights,
):
    """Score one resume (thread-safe, no DB writes). Returns (scored, threat)."""
    from src.scoring.scorer import score_resume, is_resume

    if not is_resume(cleaned):
        threat.threat_level = "HIGH"
        threat.total_penalty = 0.9
        threat.flags.append("REJECTED: LLM non-resume gate triggered")
        return _score_rejected("Document does not appear to be a resume — rejected before scoring"), threat

    scored = score_resume(
        jd_text=jd_text,
        resume_text=cleaned,
        ce_logit=ce_logit,
        n_candidates=n_candidates,
        adversarial_penalty=threat.total_penalty,
        verbose=False,
        jd_profile=jd_profile,
        custom_dim_weights=custom_weights,
    )
    return scored, threat


def process_match_run(match_run_id):
    from django.db import close_old_connections

    from src.ingestion.sanitizer import sanitize
    from src.scoring.scorer import is_resume
    from src.scoring.llm_client import has_llm

    from matcherapp.models import MatchRun, MatchResult

    import os
    close_old_connections()

    run = MatchRun.objects.get(id=match_run_id)
    run.status = 'processing'
    cfg = run.scoring_config or {}
    cfg['_worker_pid'] = os.getpid()
    run.scoring_config = cfg
    run.save(update_fields=['status', 'scoring_config'])

    jd_text = run.job.description
    resumes = list(run.resumes.all())
    n = len(resumes)
    seen_hashes = {}
    custom_weights = custom_dim_weights_tuple(run)

    # ── Phase 1: Extract + sanitize; dedup ────────────────────────────────
    sanitized = {}
    threat_reports = {}
    dup_resume_ids = set()

    for resume in resumes:
        text = _extract_text(resume)
        if not text:
            text = f"Resume: {resume.name}"
        content_hash = hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()
        if content_hash in seen_hashes:
            dup_resume_ids.add(resume.id)
            continue
        seen_hashes[content_hash] = resume.name
        cleaned, threat = sanitize(text, jd_text, str(resume.id))
        sanitized[str(resume.id)] = cleaned
        threat_reports[resume.id] = threat

    # ── Phase 2: Hybrid retrieval ─────────────────────────────────────────
    ce_logits = {}
    stage_scores_by_rid = {}
    try:
        from src.retrieval.engine import RetrievalEngine
        engine = RetrievalEngine()
        if sanitized:
            engine.index(sanitized)
            engine.search(jd_text)
            for rid in sanitized:
                ce_logits[rid] = engine.get_cross_encoder_logit(rid)
                stage_scores_by_rid[rid] = engine.get_stage_scores(jd_text, rid)
                stage_scores_by_rid[rid]["ce_logit"] = ce_logits[rid]
    except Exception as e:
        print(f"  [RETRIEVAL] Hybrid retrieval unavailable ({e}) — using ce_logit=0")
        for rid in sanitized:
            ce_logits[rid] = 0.0
            stage_scores_by_rid[rid] = {"bm25": 0, "dense": 0, "ce_logit": 0}

    # ── Phase 2.5: Extract JD profile once (reused for every resume) ──────
    jd_profile = None
    if has_llm():
        try:
            from src.scoring.extraction import extract_jd_profile
            jd_profile = extract_jd_profile(jd_text)
        except Exception as e:
            print(f"  [EXTRACTION] JD profile extraction failed ({e}) — scorer will retry per-resume")

    try:
        processed_count = 0

        # ── Phase 3a: Write dedup results immediately ─────────────────────
        for resume in resumes:
            if resume.id not in dup_resume_ids:
                continue
            try:
                content_hash = hashlib.sha256(
                    (resume.raw_text or resume.name).encode('utf-8', errors='replace')
                ).hexdigest()
                dup_of = seen_hashes.get(content_hash, "another resume")
                scored = _score_rejected(f"Duplicate of {dup_of} — identical content submitted")
                MatchResult.objects.create(
                    match_run=run, resume=resume,
                    final_score=scored['final_score'], confidence=scored['confidence'],
                    recommendation=scored['recommendation'],
                    d1_skills=0.0, d2_seniority=0.0, d3_domain=0.0, d4_constraints=0.0,
                    dim_composite=0.0, ce_sigmoid=0.5, ce_weight_used=0.0,
                    scoring_mode='dedup', strengths=[], gaps=scored['gaps'],
                    rationale=scored['rationale'], skill_detail={}, stage_scores=scored,
                    threat_level='NONE', adversarial_penalty=0.0, threat_flags=[],
                )
            except Exception as e:
                print(f"Error saving dedup for {resume.name}: {e}")
            processed_count += 1
            run.processed = processed_count
            run.save(update_fields=['processed'])

        # ── Phase 3b: Score sequentially, save + update progress after each ─
        for resume in resumes:
            if resume.id in dup_resume_ids:
                continue
            rid = str(resume.id)
            cleaned = sanitized.get(rid, resume.raw_text or resume.name)
            threat = threat_reports.get(resume.id)
            if threat is None:
                from src.contracts import ThreatReport
                threat = ThreatReport(resume_id=rid)
            ce_logit = ce_logits.get(rid, 0.0)

            try:
                scored, threat = _score_single_resume(
                    jd_text, cleaned, ce_logit, threat, n,
                    jd_profile, custom_weights,
                )
                base_stage = stage_scores_by_rid.get(rid, {})
                for k, v in base_stage.items():
                    scored[k] = v
                MatchResult.objects.create(
                    match_run=run,
                    resume=resume,
                    final_score=scored['final_score'],
                    confidence=scored['confidence'],
                    recommendation=scored['recommendation'],
                    d1_skills=scored['d1_skills'],
                    d2_seniority=scored['d2_seniority'],
                    d3_domain=scored['d3_domain'],
                    d4_constraints=scored['d4_constraints'],
                    dim_composite=scored['dim_composite'],
                    ce_sigmoid=scored['ce_sigmoid'],
                    ce_weight_used=scored['ce_weight'],
                    scoring_mode=scored['mode'],
                    strengths=scored.get('strengths', []),
                    gaps=scored.get('gaps', []),
                    rationale=scored.get('rationale', ''),
                    skill_detail=scored.get('skill_detail', {}),
                    seniority_detail=scored.get('seniority_detail', {}),
                    domain_detail=scored.get('domain_detail', {}),
                    constraint_detail=scored.get('constraint_detail', []),
                    stage_scores=scored,
                    threat_level=threat.threat_level,
                    adversarial_penalty=threat.total_penalty,
                    threat_flags=threat.flags if hasattr(threat, 'flags') else [],
                )
            except Exception as e:
                print(f"Error scoring {resume.name}: {e}")
                import traceback
                traceback.print_exc()

            processed_count += 1
            run.processed = processed_count
            run.save(update_fields=['processed'])

        # ── Assign ranks ──────────────────────────────────────────────────
        results = list(run.results.order_by('-final_score'))
        for rank, result in enumerate(results, 1):
            result.rank = rank
            result.save(update_fields=['rank'])

        run.status = 'complete'
        run.completed_at = timezone.now()
        run.save(update_fields=['status', 'completed_at'])

    except Exception as e:
        run.status = 'failed'
        run.save(update_fields=['status'])
        raise
