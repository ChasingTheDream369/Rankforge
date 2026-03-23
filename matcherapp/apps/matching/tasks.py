from celery import shared_task


@shared_task(bind=True, max_retries=0)
def process_match_run_task(self, match_run_id):
    from matcherapp.apps.matching.services import process_match_run
    process_match_run(match_run_id)
