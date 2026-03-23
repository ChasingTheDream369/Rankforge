import hashlib
import sys
from django.conf import settings
from django.utils import timezone

# Ensure project root is on sys.path so `src.*` imports work
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


def process_match_run(match_run_id):
    from src.ingestion.sanitizer import sanitize
    from src.scoring.scorer import score_resume, is_resume

    from matcherapp.models import MatchRun, MatchResult

    run = MatchRun.objects.get(id=match_run_id)
    run.status = 'processing'
    run.save(update_fields=['status'])

    jd_text = run.job.description
    resumes = list(run.resumes.all())
    n = len(resumes)
    seen_hashes = {}  # content_hash → resume name, for dedup within this run

    try:
        for i, resume in enumerate(resumes):
            try:
                # ── 1. Extract text ──────────────────────────────────────────
                text = _extract_text(resume)
                if not text:
                    text = f"Resume: {resume.name}"

                # ── 1b. Dedup — skip exact-content duplicates within this run ─
                content_hash = hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()
                if content_hash in seen_hashes:
                    print(f"  [DEDUP] {resume.name} is identical to {seen_hashes[content_hash]} — skipping")
                    scored = _score_rejected(
                        f"Duplicate of {seen_hashes[content_hash]} — identical content submitted"
                    )
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
                    run.processed = i + 1
                    run.save(update_fields=['processed'])
                    continue
                seen_hashes[content_hash] = resume.name

                # ── 2. Adversarial sanitization ──────────────────────────────
                cleaned, threat = sanitize(text, jd_text, str(resume.id))

                # ── 3. LLM non-resume gate (fast gpt-4o-mini, ~$0.0001) ──────
                if not is_resume(cleaned):
                    print(f"  [REJECTED] {resume.name} — LLM gate: not a resume")
                    scored = _score_rejected("Document does not appear to be a resume — rejected before scoring")
                    threat.threat_level = "HIGH"
                    threat.total_penalty = 0.9
                    threat.flags.append("REJECTED: LLM non-resume gate triggered")
                else:
                    # ── 4. Full 4D scoring ───────────────────────────────────
                    scored = score_resume(
                        jd_text=jd_text,
                        resume_text=cleaned,
                        ce_logit=0.0,
                        n_candidates=n,
                        adversarial_penalty=threat.total_penalty,
                        verbose=False,
                    )

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
                print(f"Error processing {resume.name}: {e}")
                import traceback
                traceback.print_exc()

            run.processed = i + 1
            run.save(update_fields=['processed'])

        # ── Assign ranks by final_score ──────────────────────────────────────
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
