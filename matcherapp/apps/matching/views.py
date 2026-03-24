from django.shortcuts import render, get_object_or_404, redirect
from matcherapp.decorators import login_required
from matcherapp.models import MatchRun, MatchResult, Job

from matcherapp.apps.matching.api import unique_match_results_for_display


def redirect_to_dashboard(request):
    return redirect('dashboard')


@login_required
def dashboard(request):
    runs = MatchRun.objects.select_related('job').all()[:20]
    total_runs = MatchRun.objects.count()
    total_resumes = sum(r.total_resumes for r in MatchRun.objects.all())
    complete_runs = MatchRun.objects.filter(status='complete')
    avg_score = 0.0
    if complete_runs.exists():
        scores = []
        for run in complete_runs:
            top = run.results.order_by('-final_score').first()
            if top:
                scores.append(top.final_score)
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    last_run = runs.first()
    context = {
        'runs': runs,
        'total_runs': total_runs,
        'total_resumes': total_resumes,
        'avg_score': avg_score,
        'last_run': last_run,
    }
    return render(request, 'matching/dashboard.html', context)


@login_required
def new_run(request):
    return render(request, 'matching/new_run.html')


WEIGHT_PROFILE_DISPLAY_LABELS = {
    "junior": "Junior",
    "mid": "Mid-level",
    "senior": "Senior",
    "staff": "Staff / Lead",
    "executive": "Executive",
    "custom": "Custom",
}


def scoring_config_note(cfg: dict) -> str:
    if not cfg or not cfg.get("custom_dims") or not cfg.get("weights"):
        return ""
    profile = cfg.get("profile", "custom")
    label = WEIGHT_PROFILE_DISPLAY_LABELS.get(profile, profile.title())
    w = cfg["weights"]
    rp = cfg.get("weights_raw_pct")
    if rp and len(rp) == 4:
        return (
            f"{label} profile — Skills {rp[0]:.0f}% · Seniority {rp[1]:.0f}% · "
            f"Domain {rp[2]:.0f}% · Constraints {rp[3]:.0f}% (normalized for scoring)."
        )
    return (
        f"{label} profile — Skills {w[0]*100:.1f}% · Seniority {w[1]*100:.1f}% · "
        f"Domain {w[2]*100:.1f}% · Constraints {w[3]*100:.1f}%."
    )


@login_required
def run_detail(request, run_id):
    run = get_object_or_404(MatchRun.objects.select_related('job'), id=run_id)
    results = unique_match_results_for_display(run)
    resume_upload_count = run.resumes.count()
    raw_result_row_count = run.results.count()
    context = {
        'run': run,
        'results': results,
        'resume_upload_count': resume_upload_count,
        'raw_result_row_count': raw_result_row_count,
        'scoring_weights_note': scoring_config_note(run.scoring_config or {}),
    }
    return render(request, 'matching/run_detail.html', context)


@login_required
def candidate_detail(request, result_id):
    result = get_object_or_404(MatchResult.objects.select_related('resume', 'match_run__job'), id=result_id)
    sd = result.skill_detail or {}
    skills_list = sd.get('skills_checked') or sd.get('d1_breakdown') or []
    context = {'result': result, 'skills_list': skills_list}
    return render(request, 'matching/candidate_detail.html', context)
