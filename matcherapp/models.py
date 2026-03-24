from django.db import models
from django.utils import timezone


class Job(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField()
    raw_file = models.FileField(upload_to='jobs/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


class MatchRun(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('complete', 'Complete'),
        ('failed', 'Failed'),
    ]
    MODE_CHOICES = [
        ('auto', 'Auto'),
        ('llm', 'LLM'),
        ('deterministic', 'Deterministic'),
    ]
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name='runs')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    scoring_mode = models.CharField(max_length=20, choices=MODE_CHOICES, default='auto')
    total_resumes = models.IntegerField(default=0)
    processed = models.IntegerField(default=0)
    # custom_dims + weights [w_d1..w_d4] when user sets JD-specific dimension importance; else {}
    scoring_config = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    @property
    def progress_pct(self):
        if self.total_resumes == 0:
            return 0
        return int((self.processed / self.total_resumes) * 100)


class Resume(models.Model):
    match_run = models.ForeignKey(MatchRun, on_delete=models.CASCADE, related_name='resumes')
    name = models.CharField(max_length=200)
    file = models.FileField(upload_to='resumes/', null=True, blank=True)
    raw_text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class MatchResult(models.Model):
    match_run = models.ForeignKey(MatchRun, on_delete=models.CASCADE, related_name='results')
    resume = models.ForeignKey(Resume, on_delete=models.CASCADE)
    rank = models.IntegerField(default=0)
    final_score = models.FloatField(default=0.0)
    confidence = models.CharField(max_length=10, default='MEDIUM')
    recommendation = models.CharField(max_length=20, default='NO_MATCH')

    d1_skills = models.FloatField(default=0.0)
    d2_seniority = models.FloatField(default=0.0)
    d3_domain = models.FloatField(default=0.0)
    d4_constraints = models.FloatField(default=0.0)
    dim_composite = models.FloatField(default=0.0)
    ce_sigmoid = models.FloatField(default=0.0)
    ce_weight_used = models.FloatField(default=0.0)
    scoring_mode = models.CharField(max_length=20, default='auto')

    strengths = models.JSONField(default=list)
    gaps = models.JSONField(default=list)
    rationale = models.TextField(blank=True)
    skill_detail = models.JSONField(default=dict)
    seniority_detail = models.JSONField(default=dict)
    domain_detail = models.JSONField(default=dict)
    constraint_detail = models.JSONField(default=dict)
    stage_scores = models.JSONField(default=dict)

    threat_level = models.CharField(max_length=20, default='NONE')
    adversarial_penalty = models.FloatField(default=0.0)
    threat_flags = models.JSONField(default=list)

    class Meta:
        ordering = ['rank']

    @property
    def score_color(self):
        if self.final_score >= 0.70:
            return 'success'
        if self.final_score >= 0.40:
            return 'warning'
        return 'error'
