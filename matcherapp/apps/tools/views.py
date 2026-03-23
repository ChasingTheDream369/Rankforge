from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from matcherapp.decorators import login_required
from .runner import run_tests_bg, get_test_state, run_ablation_bg, get_ablation_state


@login_required
def pipeline(request):
    return render(request, 'tools/pipeline.html')


@login_required
def roadmap(request):
    return render(request, 'tools/roadmap.html')


@login_required
def test_suite(request):
    return render(request, 'tools/test_suite.html')


@login_required
def ablation(request):
    approaches = [
        ("TF-IDF only",             "Baseline bag-of-words cosine similarity"),
        ("TF-IDF + BM25",           "Adds BM25 term-frequency ranking"),
        ("BM25 + Dense + RRF",      "Hybrid retrieval fused with RRF"),
        ("+ Cross-encoder",         "Re-ranks top candidates with cross-encoder"),
        ("Hybrid + Agentic (Ontology Grounding)", "ESCO ontology-grounded skill adjacency + agentic re-scoring"),
    ]
    return render(request, 'tools/ablation.html', {'approaches': approaches})


# ── Test runner API ───────────────────────────────────────────────────────────

@login_required
@require_http_methods(['POST'])
def api_run_tests(request):
    run_tests_bg()
    return JsonResponse({'status': 'started'})


@login_required
@require_http_methods(['GET'])
def api_test_status(request):
    return JsonResponse(get_test_state())


# ── Ablation API ──────────────────────────────────────────────────────────────

@login_required
@require_http_methods(['POST'])
def api_run_ablation(request):
    run_ablation_bg()
    return JsonResponse({'status': 'started'})


@login_required
@require_http_methods(['GET'])
def api_ablation_status(request):
    return JsonResponse(get_ablation_state())
