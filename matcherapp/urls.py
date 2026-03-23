from django.urls import path, include

urlpatterns = [
    path('', include('matcherapp.apps.auth.urls')),
    path('', include('matcherapp.apps.matching.urls')),
]
