from django.urls import path
from . import views, api
from matcherapp.apps.tools import views as tool_views

urlpatterns = [
    path('', views.redirect_to_dashboard),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('matching/new/', views.new_run, name='new_run'),
    path('matching/run/<int:run_id>/', views.run_detail, name='run_detail'),
    path('matching/candidate/<int:result_id>/', views.candidate_detail, name='candidate_detail'),

    # Tool pages
    path('pipeline/', tool_views.pipeline, name='pipeline'),
    path('roadmap/',  tool_views.roadmap,  name='roadmap'),
    path('tests/',    tool_views.test_suite, name='test_suite'),
    path('ablation/', tool_views.ablation, name='ablation'),

    # Matching API
    path('api/matching/start/',                          api.start_run,       name='api_start_run'),
    path('api/matching/run/<int:run_id>/status/',        api.run_status,      name='api_run_status'),
    path('api/matching/candidate/<int:result_id>/',      api.candidate_api,   name='api_candidate'),
    path('api/matching/resume/<int:resume_id>/preview/', api.resume_preview,  name='api_resume_preview'),
    path('api/matching/run/<int:run_id>/rescore/',       api.rescore_run,     name='api_rescore_run'),
    path('api/matching/result/<int:result_id>/rescore/', api.rescore_single,  name='api_rescore_single'),
    path('api/matching/run/<int:run_id>/export/',        api.export_csv,      name='api_export_csv'),

    # Tools API
    path('api/tests/run/',       tool_views.api_run_tests,      name='api_run_tests'),
    path('api/tests/status/',    tool_views.api_test_status,    name='api_test_status'),
    path('api/ablation/run/',    tool_views.api_run_ablation,   name='api_run_ablation'),
    path('api/ablation/status/', tool_views.api_ablation_status, name='api_ablation_status'),
]
