import csv
import os
import sys
import json
import subprocess
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from matcherapp.decorators import login_required
from matcherapp.models import Job, MatchRun, Resume, MatchResult


def _launch_run_worker(run_id: int):
    """Start process_run in a detached subprocess that survives server reloads."""
    subprocess.Popen(
        [sys.executable, 'manage.py', 'process_run', str(run_id)],
        cwd=str(settings.BASE_DIR),
        start_new_session=True,
        stdout=open(os.path.join(settings.BASE_DIR, 'logs', f'run_{run_id}.log'), 'a')
               if os.path.isdir(os.path.join(settings.BASE_DIR, 'logs'))
               else subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


WEIGHT_PROFILE_PRESETS = {
    "junior":    (50, 25, 15, 10),
    "mid":       (40, 35, 15, 10),
    "senior":    (35, 45, 12, 8),
    "staff":     (30, 50, 12, 8),
    "executive": (25, 55, 12, 8),
}


def _parse_scoring_config_from_request(request) -> dict:
    """
    Build MatchRun.scoring_config from POST.
    Supports three modes:
      - "auto" (default): empty dict → scorer uses base constants + JD seniority presets.
      - Named profile (junior/mid/senior/staff/executive): preset weights, optionally tweaked.
      - "custom": user percentages for D1–D4.
    All non-auto modes normalize D1–D4 to sum 1.0.
    """
    profile = (request.POST.get("weight_profile") or "auto").strip().lower()

    if profile == "auto":
        if request.POST.get("custom_dim_weights") not in ("1", "true", "on", "yes"):
            return {}
        profile = "custom"

    if profile in WEIGHT_PROFILE_PRESETS:
        pd1, pd2, pd3, pd4 = WEIGHT_PROFILE_PRESETS[profile]
    else:
        pd1, pd2, pd3, pd4 = 0, 0, 0, 0

    try:
        d1 = float(request.POST.get("d1_pct", pd1) or pd1)
        d2 = float(request.POST.get("d2_pct", pd2) or pd2)
        d3 = float(request.POST.get("d3_pct", pd3) or pd3)
        d4 = float(request.POST.get("d4_pct", pd4) or pd4)
    except (TypeError, ValueError):
        if profile in WEIGHT_PROFILE_PRESETS:
            d1, d2, d3, d4 = WEIGHT_PROFILE_PRESETS[profile]
        else:
            return {}

    s = d1 + d2 + d3 + d4
    if s <= 0:
        return {}
    return {
        "custom_dims": True,
        "profile": profile,
        "weights": [d1 / s, d2 / s, d3 / s, d4 / s],
        "weights_raw_pct": [d1, d2, d3, d4],
    }


@login_required
@require_http_methods(['POST'])
def start_run(request):
    try:
        jd_text = request.POST.get('jd_text', '').strip()
        jd_title = request.POST.get('jd_title', 'Untitled Job').strip()
        scoring_mode = request.POST.get('scoring_mode', 'auto')

        if not jd_text:
            jd_file = request.FILES.get('jd_file')
            if jd_file:
                jd_text = jd_file.read().decode('utf-8', errors='ignore')
            else:
                return JsonResponse({'message': 'Job description is required.'}, status=400)

        if len(jd_text) > 20_000:
            return JsonResponse({'message': 'Job description too long (max 20,000 characters).'}, status=400)

        uploaded = request.FILES.getlist('resumes')
        if not uploaded:
            return JsonResponse({'message': 'At least one resume is required.'}, status=400)

        # Expand any ZIPs into individual files (in-memory → temp dir)
        import zipfile, tempfile, shutil
        from django.core.files.uploadedfile import InMemoryUploadedFile
        import io

        from src.config import MAX_RESUMES_PER_RUN
        resume_files = []  # list of (name_stem, file_object_or_path, is_temp)
        temp_dirs = []

        for uf in uploaded:
            ext = os.path.splitext(uf.name)[1].lower()
            if ext == '.zip':
                tmpdir = tempfile.mkdtemp()
                temp_dirs.append(tmpdir)
                try:
                    with zipfile.ZipFile(uf) as zf:
                        zf.extractall(tmpdir)
                    for root, _, files in os.walk(tmpdir):
                        if len(resume_files) >= MAX_RESUMES_PER_RUN:
                            break
                        for fname in sorted(files):
                            if fname.startswith('.') or fname.startswith('__'):
                                continue
                            fext = os.path.splitext(fname)[1].lower()
                            if fext in {'.zip', '.exe', '.dmg'}:
                                continue
                            fpath = os.path.join(root, fname)
                            stem = os.path.splitext(fname)[0].replace('_', ' ').replace('-', ' ').title()
                            resume_files.append((stem, fpath, True))
                except Exception as e:
                    for d in temp_dirs:
                        shutil.rmtree(d, ignore_errors=True)
                    return JsonResponse({'message': f'Could not read ZIP: {e}'}, status=400)
            else:
                if len(resume_files) < MAX_RESUMES_PER_RUN:
                    stem = os.path.splitext(uf.name)[0].replace('_', ' ').replace('-', ' ').title()
                    resume_files.append((stem, uf, False))

        if not resume_files:
            return JsonResponse({'message': 'No valid files found in upload.'}, status=400)
        if len(resume_files) > MAX_RESUMES_PER_RUN:
            for d in temp_dirs:
                shutil.rmtree(d, ignore_errors=True)
            return JsonResponse({'message': f'Maximum {MAX_RESUMES_PER_RUN} resumes per run.'}, status=400)

        job = Job.objects.create(title=jd_title, description=jd_text)
        scoring_cfg = _parse_scoring_config_from_request(request)
        run = MatchRun.objects.create(
            job=job,
            scoring_mode=scoring_mode,
            total_resumes=len(resume_files),
            status='pending',
            scoring_config=scoring_cfg,
        )

        for stem, file_obj, is_path in resume_files:
            if is_path:
                with open(file_obj, 'rb') as fh:
                    content = fh.read()
                fname = os.path.basename(file_obj)
                mem_file = InMemoryUploadedFile(
                    io.BytesIO(content), 'file', fname,
                    'application/octet-stream', len(content), None,
                )
                Resume.objects.create(match_run=run, name=stem, file=mem_file)
            else:
                Resume.objects.create(match_run=run, name=stem, file=file_obj)

        for d in temp_dirs:
            shutil.rmtree(d, ignore_errors=True)

        _launch_run_worker(run.id)

        return JsonResponse({
            'run_id': run.id,
            'redirect_url': f'/matching/run/{run.id}/',
            'message': f'Processing {len(resume_files)} resumes...',
            'showSnackbar': True,
        })
    except Exception as e:
        return JsonResponse({'message': str(e)}, status=500)


def _run_results_json(run):
    """Serialize MatchResults for the run-detail poll API (partial rows while processing, final ranks when complete)."""
    qs = run.results.select_related('resume').order_by('-final_score', 'id')
    results_data = []
    for idx, r in enumerate(qs, start=1):
        display_rank = r.rank if run.status == 'complete' and r.rank > 0 else idx
        results_data.append({
            'id': r.id,
            'resume_id': r.resume.id,
            'rank': display_rank,
            'name': r.resume.name,
            'final_score': round(r.final_score, 4),
            'confidence': r.confidence,
            'recommendation': r.recommendation,
            'd1_skills': round(r.d1_skills, 2),
            'd2_seniority': round(r.d2_seniority, 2),
            'd3_domain': round(r.d3_domain, 2),
            'd4_constraints': round(r.d4_constraints, 2),
            'threat_level': r.threat_level,
            'score_color': r.score_color,
            'detail_url': f'/matching/candidate/{r.id}/',
        })
    return results_data


@login_required
@require_http_methods(['GET'])
def run_status(request, run_id):
    try:
        run = MatchRun.objects.get(id=run_id)

        if run.status in ('processing', 'pending'):
            worker_pid = (run.scoring_config or {}).get('_worker_pid')
            worker_alive = False
            if worker_pid:
                try:
                    os.kill(worker_pid, 0)
                    worker_alive = True
                except OSError:
                    pass
            if not worker_alive and run.status == 'processing':
                run.results.all().delete()
                run.status = 'pending'
                run.processed = 0
                run.save(update_fields=['status', 'processed'])
                _launch_run_worker(run.id)

        # Expose rows during pending/processing so the UI is not blank for long LLM runs
        results_data = _run_results_json(run) if run.results.exists() else []
        return JsonResponse({
            'status': run.status,
            'processed': run.processed,
            'total': run.total_resumes,
            'progress_pct': run.progress_pct,
            'results': results_data,
        })
    except MatchRun.DoesNotExist:
        return JsonResponse({'message': 'Run not found.'}, status=404)


@login_required
@require_http_methods(['POST'])
def rescore_run(request, run_id):
    try:
        run = MatchRun.objects.get(id=run_id)
        run.results.all().delete()
        run.status = 'pending'
        run.processed = 0
        run.save(update_fields=['status', 'processed'])

        _launch_run_worker(run.id)

        return JsonResponse({'message': 'Re-scoring started.', 'run_id': run.id})
    except MatchRun.DoesNotExist:
        return JsonResponse({'message': 'Run not found.'}, status=404)
    except Exception as e:
        return JsonResponse({'message': str(e)}, status=500)


@login_required
@require_http_methods(['GET'])
def resume_preview(request, resume_id):
    try:
        from matcherapp.models import Resume
        resume = Resume.objects.get(id=resume_id)
        text = resume.raw_text
        # Re-extract if stored text is missing or too short (likely a failed extraction)
        if (not text or len(text) < 100) and resume.file:
            try:
                from src.ingestion.extractor import extract_text
                text = extract_text(resume.file.path)
            except Exception:
                try:
                    with open(resume.file.path, 'r', errors='ignore') as fh:
                        text = fh.read()
                except Exception:
                    text = ''
        file_url = resume.file.url if resume.file else None
        return JsonResponse({'name': resume.name, 'text': text or '(no content)', 'file_url': file_url})
    except Exception:
        return JsonResponse({'message': 'Resume not found.'}, status=404)


@login_required
@require_http_methods(['POST'])
def rescore_single(request, result_id):
    try:
        result = MatchResult.objects.select_related('resume', 'match_run__job').get(id=result_id)
        run = result.match_run
        resume = result.resume

        # Re-extract text if needed
        text = resume.raw_text
        if (not text or len(text) < 100) and resume.file:
            try:
                from src.ingestion.extractor import extract_text
                text = extract_text(resume.file.path)
                resume.raw_text = text
                resume.save(update_fields=['raw_text'])
            except Exception:
                pass
        if not text:
            text = resume.name

        import sys
        from django.conf import settings
        _root = str(settings.BASE_DIR)
        if _root not in sys.path:
            sys.path.insert(0, _root)

        from src.ingestion.sanitizer import sanitize
        from src.scoring.scorer import score_resume

        n = run.results.count()
        cleaned, threat = sanitize(text, run.job.description, str(resume.id))
        ce_logit = 0.0
        try:
            from src.retrieval.engine import RetrievalEngine
            all_texts = {}
            for r in run.resumes.all():
                rt = r.raw_text or (r.name if not r.file else "")
                if len(rt) < 100 and r.file:
                    try:
                        from src.ingestion.extractor import extract_text
                        rt = extract_text(r.file.path)
                    except Exception:
                        pass
                if rt:
                    all_texts[str(r.id)] = rt
            if all_texts:
                eng = RetrievalEngine()
                eng.index(all_texts)
                eng.search(run.job.description)
                ce_logit = eng.get_cross_encoder_logit(str(resume.id))
        except Exception:
            pass
        from matcherapp.apps.matching.services import custom_dim_weights_tuple

        scored = score_resume(
            jd_text=run.job.description,
            resume_text=cleaned,
            ce_logit=ce_logit,
            n_candidates=n,
            adversarial_penalty=threat.total_penalty,
            verbose=False,
            custom_dim_weights=custom_dim_weights_tuple(run),
        )

        result.final_score = scored['final_score']
        result.confidence = scored['confidence']
        result.recommendation = scored['recommendation']
        result.d1_skills = scored['d1_skills']
        result.d2_seniority = scored['d2_seniority']
        result.d3_domain = scored['d3_domain']
        result.d4_constraints = scored['d4_constraints']
        result.dim_composite = scored['dim_composite']
        result.ce_sigmoid = scored['ce_sigmoid']
        result.ce_weight_used = scored['ce_weight']
        result.scoring_mode = scored['mode']
        result.strengths = scored.get('strengths', [])
        result.gaps = scored.get('gaps', [])
        result.rationale = scored.get('rationale', '')
        result.skill_detail = scored.get('skill_detail', {})
        result.seniority_detail = scored.get('seniority_detail', {})
        result.domain_detail = scored.get('domain_detail', {})
        result.constraint_detail = scored.get('constraint_detail', [])
        result.stage_scores = scored
        result.threat_level = threat.threat_level
        result.adversarial_penalty = threat.total_penalty
        result.threat_flags = threat.flags if hasattr(threat, 'flags') else []
        result.save()

        # Re-rank all results in this run
        all_results = list(run.results.order_by('-final_score'))
        for rank, r in enumerate(all_results, 1):
            r.rank = rank
            r.save(update_fields=['rank'])

        return JsonResponse({'message': 'Re-scored.', 'new_score': result.final_score})
    except MatchResult.DoesNotExist:
        return JsonResponse({'message': 'Result not found.'}, status=404)
    except Exception as e:
        return JsonResponse({'message': str(e)}, status=500)


@login_required
@require_http_methods(['GET'])
def candidate_api(request, result_id):
    try:
        r = MatchResult.objects.select_related('resume', 'match_run__job').get(id=result_id)
        return JsonResponse({
            'id': r.id,
            'name': r.resume.name,
            'final_score': r.final_score,
            'confidence': r.confidence,
            'recommendation': r.recommendation,
            'd1_skills': r.d1_skills,
            'd2_seniority': r.d2_seniority,
            'd3_domain': r.d3_domain,
            'd4_constraints': r.d4_constraints,
            'dim_composite': r.dim_composite,
            'rationale': r.rationale,
            'strengths': r.strengths,
            'gaps': r.gaps,
            'skill_detail': r.skill_detail,
            'seniority_detail': r.seniority_detail,
            'domain_detail': r.domain_detail,
            'constraint_detail': r.constraint_detail,
            'stage_scores': r.stage_scores,
            'threat_level': r.threat_level,
            'threat_flags': r.threat_flags,
            'adversarial_penalty': r.adversarial_penalty,
        })
    except MatchResult.DoesNotExist:
        return JsonResponse({'message': 'Result not found.'}, status=404)


@login_required
@require_http_methods(['GET'])
def export_csv(request, run_id):
    try:
        run = MatchRun.objects.select_related('job').get(id=run_id)
        results = run.results.select_related('resume').order_by('rank')

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="run_{run_id}_results.csv"'

        writer = csv.writer(response)
        writer.writerow([
            'Rank', 'Name', 'Final Score',
            'D1 Skills', 'D2 Seniority', 'D3 Domain', 'D4 Constraints',
            'Confidence', 'Recommendation', 'Threat Level',
        ])
        for r in results:
            writer.writerow([
                r.rank,
                r.resume.name,
                round(r.final_score, 4),
                round(r.d1_skills, 4),
                round(r.d2_seniority, 4),
                round(r.d3_domain, 4),
                round(r.d4_constraints, 4),
                r.confidence,
                r.recommendation,
                r.threat_level,
            ])
        return response
    except MatchRun.DoesNotExist:
        return JsonResponse({'message': 'Run not found.'}, status=404)
