from django.apps import AppConfig


class MatcherappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'matcherapp'

    def ready(self):
        # On every server start (including auto-reloads) any run left in
        # 'processing' or 'pending' has no live thread behind it — mark it
        # failed so the UI stops spinning and the user can re-trigger.
        try:
            from matcherapp.models import MatchRun
            stuck = MatchRun.objects.filter(status__in=['processing', 'pending'])
            count = stuck.update(status='failed')
            if count:
                print(f"[startup] Reset {count} stuck run(s) to 'failed'.")
        except Exception:
            # DB may not be ready on first migrate — silently skip
            pass
