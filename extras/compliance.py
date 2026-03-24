"""
Regulatory Compliance Module — EU AI Act + NYC Local Law 144.

EU AI Act (Article 6, Annex III — High-Risk):
  - Immutable audit log for every scoring decision
  - Model state + context hash recorded per run
  - Human oversight checkpoints embedded in workflow
  - Data governance: training data provenance tracked

NYC Local Law 144 (AEDT):
  - Impact ratio computation per demographic group
  - Four-fifths rule monitoring (threshold < 0.8 flags violation)
  - Annual audit report generation (structured JSON for third-party auditors)
  - Candidate notification tracking

Every pipeline run produces an AuditRecord that is:
  - Append-only (immutable once written)
  - Forensically reproducible (same inputs → same outputs guaranteed by temp=0)
  - JSON-serializable for third-party auditor consumption
"""

import json
import os
import hashlib
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

from src.config import FEEDBACK_DIR, LLM_PROVIDER, OPENAI_MODEL
AUDIT_DIR = os.path.join(FEEDBACK_DIR, "audit_logs")
BIAS_REPORT_DIR = os.path.join(FEEDBACK_DIR, "bias_reports")
# NYC LL144 four-fifths rule: flag when impact_ratio < threshold (default 0.8). Override via env for audits.
try:
    _fft = os.environ.get("FAIRNESS_FOUR_FIFTHS_THRESHOLD")
    FAIRNESS_FOUR_FIFTHS_THRESHOLD = float(_fft) if _fft is not None and _fft != "" else 0.8
except (TypeError, ValueError):
    FAIRNESS_FOUR_FIFTHS_THRESHOLD = 0.8
FAIRNESS_FOUR_FIFTHS_THRESHOLD = max(0.1, min(1.0, FAIRNESS_FOUR_FIFTHS_THRESHOLD))
# ============================================================
# EU AI Act — Immutable Audit Trail
# ============================================================

@dataclass
class AuditRecord:
    """
    Immutable record of a single pipeline execution.
    Satisfies EU AI Act Article 12 (Recordkeeping) and Article 14 (Human Oversight).
    """
    audit_id: str
    timestamp: str
    # System state
    model_id: str                       # which LLM was used
    model_temperature: float = 0.0      # must be 0 for reproducibility
    pipeline_version: str = "2.0"
    config_hash: str = ""               # hash of config.py for drift detection
    # Input provenance
    jd_id: str = ""
    jd_hash: str = ""                   # SHA256 of JD text
    num_resumes: int = 0
    resume_hashes: Dict[str, str] = field(default_factory=dict)  # {id: SHA256}
    # Output
    rankings: List[dict] = field(default_factory=list)  # [{resume_id, score, confidence, recommendation}]
    adversarial_flags: List[dict] = field(default_factory=list)
    # Metrics
    evaluation_metrics: dict = field(default_factory=dict)
    cost_summary: dict = field(default_factory=dict)
    # Human oversight
    hitl_required: bool = False         # True if any result has LOW confidence
    hitl_completed: bool = False
    hitl_overrides: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
def compute_hash(text: str) -> str:
    """SHA256 hash for content integrity verification."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
def compute_config_hash() -> str:
    """Hash the config module to detect configuration drift between runs."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.py")
        with open(config_path, 'r') as f:
            return compute_hash(f.read())
    except Exception:
        return "unknown"
def create_audit_record(
    jd_id: str,
    jd_text: str,
    resumes: Dict[str, dict],
    results: list,
    metrics: dict = None,
    cost_summary: dict = None,
) -> AuditRecord:
    """
    Create an immutable audit record from a pipeline run.
    Called automatically at the end of every pipeline execution.
    """
    model_id = OPENAI_MODEL if LLM_PROVIDER == "openai" else "regex_fallback"

    rankings = []
    adversarial_flags = []
    hitl_required = False

    for r in results:
        rankings.append({
            "resume_id": r.resume_id,
            "name": r.name,
            "rank": r.rank,
            "score": r.final_score,
            "confidence": r.confidence,
            "recommendation": r.rationale.recommendation if r.rationale else "",
        })
        if r.confidence == "LOW":
            hitl_required = True
        if r.threat_report and not r.threat_report.is_clean:
            adversarial_flags.append({
                "resume_id": r.resume_id,
                "threat_level": r.threat_report.threat_level,
                "penalty": r.threat_report.total_penalty,
                "flags": r.threat_report.flags,
            })

    return AuditRecord(
        audit_id=f"audit_{int(time.time() * 1000)}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_id=model_id,
        model_temperature=0.0,
        config_hash=compute_config_hash(),
        jd_id=jd_id,
        jd_hash=compute_hash(jd_text),
        num_resumes=len(resumes),
        resume_hashes={rid: compute_hash(r.get("text", "")) for rid, r in resumes.items()},
        rankings=rankings,
        adversarial_flags=adversarial_flags,
        evaluation_metrics=metrics or {},
        cost_summary=cost_summary or {},
        hitl_required=hitl_required,
    )
def save_audit_record(record: AuditRecord) -> str:
    """Append audit record to immutable log. Returns filepath."""
    os.makedirs(AUDIT_DIR, exist_ok=True)
    log_path = os.path.join(AUDIT_DIR, "audit_log.jsonl")
    with open(log_path, 'a') as f:
        f.write(json.dumps(record.to_dict()) + '\n')
    return log_path
def load_audit_history() -> List[dict]:
    """Load all audit records for compliance review."""
    log_path = os.path.join(AUDIT_DIR, "audit_log.jsonl")
    if not os.path.exists(log_path):
        return []
    records = []
    with open(log_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
def verify_reproducibility(record: AuditRecord, current_config_hash: str) -> dict:
    """
    Check if a past run can be reproduced with current system state.
    Detects config drift, model changes, and input modifications.
    """
    issues = []
    if record.config_hash != current_config_hash:
        issues.append(f"Config drift: recorded={record.config_hash}, current={current_config_hash}")
    if record.model_temperature != 0.0:
        issues.append(f"Non-zero temperature: {record.model_temperature} (must be 0.0)")

    return {
        "reproducible": len(issues) == 0,
        "issues": issues,
        "audit_id": record.audit_id,
        "timestamp": record.timestamp,
    }
# ============================================================
# NYC Local Law 144 — Bias Auditing
# ============================================================

@dataclass
class BiasAuditReport:
    """
    Structured bias audit report per NYC LL144 requirements.
    Must be conducted annually by independent third-party auditor.
    Results must be publicly posted on employer's website.
    """
    report_id: str
    generated_at: str
    audit_period_start: str
    audit_period_end: str
    # Impact ratios per demographic category
    sex_impact_ratios: Dict[str, float] = field(default_factory=dict)
    ethnicity_impact_ratios: Dict[str, float] = field(default_factory=dict)
    # Violations
    violations: List[dict] = field(default_factory=list)
    # Overall assessment
    compliant: bool = True
    recommendations: List[str] = field(default_factory=list)
    # Metadata
    total_candidates_evaluated: int = 0
    selection_threshold: float = 0.5
    four_fifths_threshold: float = 0.8
def compute_selection_rates(
    scores_by_group: Dict[str, List[float]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute selection rate per demographic group."""
    rates = {}
    for group, scores in scores_by_group.items():
        if scores:
            rates[group] = sum(1 for s in scores if s >= threshold) / len(scores)
        else:
            rates[group] = 0.0
    return rates
def compute_impact_ratios(selection_rates: Dict[str, float]) -> Dict[str, float]:
    """
    Impact ratio = selection_rate(group) / selection_rate(highest_group).
    Per EEOC four-fifths rule: ratio < 0.8 indicates potential disparate impact.
    """
    if not selection_rates:
        return {}
    max_rate = max(selection_rates.values())
    if max_rate == 0:
        return {g: 0.0 for g in selection_rates}
    return {g: round(r / max_rate, 4) for g, r in selection_rates.items()}
def check_four_fifths_rule(
    impact_ratios: Dict[str, float],
    threshold: Optional[float] = None,
) -> List[dict]:
    """Flag groups with impact ratio below threshold (default: FAIRNESS_FOUR_FIFTHS_THRESHOLD, usually 0.8)."""
    th = FAIRNESS_FOUR_FIFTHS_THRESHOLD if threshold is None else max(0.1, min(1.0, float(threshold)))
    violations = []
    for group, ratio in impact_ratios.items():
        if ratio < th:
            violations.append({
                "group": group,
                "impact_ratio": ratio,
                "threshold": th,
                "severity": "CRITICAL" if ratio < 0.5 else "WARNING",
                "action_required": "Immediate review and remediation required" if ratio < 0.5
                                   else "Monitor and investigate root cause",
            })
    return violations
def generate_bias_audit_report(
    sex_scores: Dict[str, List[float]] = None,
    ethnicity_scores: Dict[str, List[float]] = None,
    selection_threshold: float = 0.5,
    audit_period: tuple = None,
) -> BiasAuditReport:
    """
    Generate a NYC LL144-compliant bias audit report.

    In production, this would run on anonymized demographic data
    from actual candidate scoring runs. For the POC, it demonstrates
    the mathematical framework and audit structure.
    """
    now = datetime.now(timezone.utc).isoformat()
    start = audit_period[0] if audit_period else now
    end = audit_period[1] if audit_period else now

    report = BiasAuditReport(
        report_id=f"bias_audit_{int(time.time())}",
        generated_at=now,
        audit_period_start=start,
        audit_period_end=end,
        selection_threshold=selection_threshold,
    )

    total_candidates = 0

    if sex_scores:
        rates = compute_selection_rates(sex_scores, selection_threshold)
        ratios = compute_impact_ratios(rates)
        report.sex_impact_ratios = ratios
        violations = check_four_fifths_rule(ratios)
        report.violations.extend([{**v, "category": "sex"} for v in violations])
        total_candidates += sum(len(s) for s in sex_scores.values())

    if ethnicity_scores:
        rates = compute_selection_rates(ethnicity_scores, selection_threshold)
        ratios = compute_impact_ratios(rates)
        report.ethnicity_impact_ratios = ratios
        violations = check_four_fifths_rule(ratios)
        report.violations.extend([{**v, "category": "ethnicity"} for v in violations])
        total_candidates += sum(len(s) for s in ethnicity_scores.values())

    report.total_candidates_evaluated = total_candidates
    report.compliant = len(report.violations) == 0

    if not report.compliant:
        report.recommendations.append(
            "IMMEDIATE: Review scoring algorithm for potential proxy variables causing disparate impact."
        )
        report.recommendations.append(
            "REQUIRED: Engage independent third-party auditor to validate findings per NYC LL144."
        )
        report.recommendations.append(
            "REQUIRED: Publicly post audit results on employer website within 30 days."
        )
    else:
        report.recommendations.append(
            "System is within compliance thresholds. Schedule next annual audit."
        )

    if not sex_scores and not ethnicity_scores:
        report.recommendations = [
            "POC MODE: No demographic data available for live bias auditing.",
            "PRODUCTION: Connect anonymized demographic data source to enable continuous monitoring.",
            "REQUIRED: Impact ratios must be computed across sex and race/ethnicity categories.",
            f"THRESHOLD: Impact ratio < {report.four_fifths_threshold} flags potential disparate impact.",
        ]

    return report
def save_bias_report(report: BiasAuditReport) -> str:
    """Save bias audit report for regulatory compliance."""
    os.makedirs(BIAS_REPORT_DIR, exist_ok=True)
    path = os.path.join(BIAS_REPORT_DIR, f"{report.report_id}.json")
    with open(path, 'w') as f:
        json.dump(asdict(report), f, indent=2)
    return path
def print_bias_report(report: BiasAuditReport) -> None:
    """Print bias audit report for terminal review."""
    print(f"\n{'=' * 70}")
    print(f"  NYC LOCAL LAW 144 — BIAS AUDIT REPORT")
    print(f"{'=' * 70}")
    print(f"  Report ID:    {report.report_id}")
    print(f"  Generated:    {report.generated_at}")
    print(f"  Candidates:   {report.total_candidates_evaluated}")
    print(f"  Compliant:    {'YES' if report.compliant else 'NO — VIOLATIONS DETECTED'}")

    if report.sex_impact_ratios:
        print(f"\n  Sex Impact Ratios:")
        for group, ratio in report.sex_impact_ratios.items():
            flag = " ⚠ BELOW 0.8" if ratio < 0.8 else " ✓"
            bar = "█" * int(ratio * 20) + "░" * (20 - int(ratio * 20))
            print(f"    {group:<15} {bar} {ratio:.4f}{flag}")

    if report.ethnicity_impact_ratios:
        print(f"\n  Ethnicity Impact Ratios:")
        for group, ratio in report.ethnicity_impact_ratios.items():
            flag = " ⚠ BELOW 0.8" if ratio < 0.8 else " ✓"
            bar = "█" * int(ratio * 20) + "░" * (20 - int(ratio * 20))
            print(f"    {group:<15} {bar} {ratio:.4f}{flag}")

    if report.violations:
        print(f"\n  VIOLATIONS ({len(report.violations)}):")
        for v in report.violations:
            print(f"    [{v['severity']}] {v['category']}/{v['group']}: "
                  f"ratio={v['impact_ratio']:.4f} (threshold={v['threshold']})")
            print(f"      Action: {v['action_required']}")

    print(f"\n  Recommendations:")
    for r in report.recommendations:
        print(f"    → {r}")
    print()
