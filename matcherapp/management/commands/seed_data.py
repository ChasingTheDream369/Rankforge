import os
import sys
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from django.conf import settings


class Command(BaseCommand):
    help = 'Seed admin user + sample FinPay run from engine data'

    def handle(self, *args, **options):
        # Create admin user
        if not User.objects.filter(username='admin').exists():
            User.objects.create_superuser('admin', 'admin@example.com', 'admin123')
            self.stdout.write(self.style.SUCCESS('Created admin user (admin / admin123)'))
        else:
            self.stdout.write('Admin user already exists')

        # Locate engine data (now inside the project)
        data_dir = str(settings.BASE_DIR / 'data')
        jd_path = os.path.join(data_dir, 'job_descriptions', 'senior_backend_finpay.txt')
        resume_dir = os.path.join(data_dir, 'resumes')

        if not os.path.exists(jd_path):
            self.stdout.write(self.style.WARNING(f'JD not found at {jd_path} — skipping sample run'))
            return

        with open(jd_path, 'r') as f:
            jd_text = f.read()

        resume_files = sorted([
            f for f in os.listdir(resume_dir) if f.endswith('.txt')
        ]) if os.path.isdir(resume_dir) else []

        if not resume_files:
            self.stdout.write(self.style.WARNING('No resumes found — skipping sample run'))
            return

        from matcherapp.models import Job, MatchRun, Resume
        job = Job.objects.create(
            title='Senior Backend Engineer — FinPay',
            description=jd_text,
        )
        run = MatchRun.objects.create(
            job=job,
            scoring_mode='auto',
            total_resumes=len(resume_files),
            status='pending',
        )

        for fname in resume_files:
            name = fname.replace('.txt', '').replace('_', ' ').title()
            fpath = os.path.join(resume_dir, fname)
            with open(fpath, 'r') as f:
                text = f.read()
            Resume.objects.create(match_run=run, name=name, raw_text=text)

        self.stdout.write(f'Created run #{run.id} with {len(resume_files)} resumes')
        self.stdout.write('Processing now...')

        from matcherapp.apps.matching.services import process_match_run
        process_match_run(run.id)

        self.stdout.write(self.style.SUCCESS(f'Done! Visit /matching/run/{run.id}/ to see results'))
