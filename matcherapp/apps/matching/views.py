from django.shortcuts import render, get_object_or_404, redirect
from matcherapp.decorators import login_required
from matcherapp.models import MatchRun, MatchResult, Job


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


@login_required
def run_detail(request, run_id):
    run = get_object_or_404(MatchRun.objects.select_related('job'), id=run_id)
    results = run.results.select_related('resume').all()
    context = {'run': run, 'results': results}
    return render(request, 'matching/run_detail.html', context)


@login_required
def candidate_detail(request, result_id):
    result = get_object_or_404(MatchResult.objects.select_related('resume', 'match_run__job'), id=result_id)
    context = {'result': result}
    return render(request, 'matching/candidate_detail.html', context)
