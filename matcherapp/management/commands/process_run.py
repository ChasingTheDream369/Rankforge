"""
Management command to process a match run in its own process.

Launched via subprocess.Popen with start_new_session=True so it
survives Django dev-server auto-reloads (StatReloader kills daemon
threads but not detached child processes).
"""
import sys
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Process a match run by ID (called via subprocess from the API)'

    def add_arguments(self, parser):
        parser.add_argument('run_id', type=int)

    def handle(self, *args, **options):
        from matcherapp.apps.matching.services import process_match_run
        run_id = options['run_id']
        try:
            process_match_run(run_id)
        except Exception as e:
            self.stderr.write(f"Run {run_id} failed: {e}")
            sys.exit(1)
