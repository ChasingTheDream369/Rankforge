import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'matcherserver.settings')

app = Celery('matcherserver')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
