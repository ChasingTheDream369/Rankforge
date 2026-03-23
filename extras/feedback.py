"""
Human-in-the-Loop Feedback System — captures, analyzes, and applies feedback.

Design: Feedback is CONTEXT, not TRAINING DATA.
  - Adjusts scoring weights, never retrains the model (avoids Amazon bias trap)
  - Recruiter remains sovereign — feedback influences, never overrides
  - All feedback is immutable JSON (EU AI Act audit compliance)
  - Weight adjustments bounded ±0.10 with re-normalization

Flow:
  AI scores → Recruiter reviews → Feedback stored → Patterns analyzed →
  Weight adjustments suggested → Applied on next scoring run
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from collections import Counter
from typing import List, Optional, Dict

from src.config import FEEDBACK_DIR
@dataclass
class RecruiterFeedback:
    """Single feedback entry from a recruiter."""
    job_id: str
    resume_id: str
    candidate_name: str
    ai_score: float
    ai_rank: int
    recruiter_decision: str     # ADVANCE | MAYBE | REJECT
    recruiter_relevance: float  # 0.0 to 1.0
    decision_reasons: List[str] = field(default_factory=list)
    notes: str = ""
    reviewer_id: str = "anonymous"
    timestamp: float = 0.0
    feedback_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.feedback_id:
            self.feedback_id = f"fb_{int(self.timestamp * 1000)}"

    def to_dict(self) -> dict:
        return asdict(self)
class FeedbackStore:
    """Immutable JSONL feedback log."""

    def __init__(self, store_dir: str = None):
        self.store_dir = store_dir or FEEDBACK_DIR
        os.makedirs(self.store_dir, exist_ok=True)
        self.log_path = os.path.join(self.store_dir, "feedback_log.jsonl")

    def record(self, fb: RecruiterFeedback) -> str:
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(fb.to_dict()) + '\n')
        return fb.feedback_id

    def load_all(self) -> List[dict]:
        if not os.path.exists(self.log_path):
            return []
        entries = []
        with open(self.log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def load_for_job(self, job_id: str) -> List[dict]:
        return [e for e in self.load_all() if e.get("job_id") == job_id]

    def count(self) -> int:
        return len(self.load_all())
class FeedbackAnalyzer:
    """Analyze feedback to find scoring calibration errors and patterns."""

    def __init__(self, store: FeedbackStore):
        self.store = store

    def compute_calibration(self, job_id: str = None) -> dict:
        entries = self.store.load_for_job(job_id) if job_id else self.store.load_all()
        if not entries:
            return {"status": "no_feedback", "entries": 0}

        def avg(group, key):
            vals = [e[key] for e in group]
            return round(sum(vals) / len(vals), 4) if vals else 0

        advanced = [e for e in entries if e["recruiter_decision"] == "ADVANCE"]
        rejected = [e for e in entries if e["recruiter_decision"] == "REJECT"]

        # Disagreement analysis
        disagreements = []
        for e in entries:
            delta = e["ai_score"] - e["recruiter_relevance"]
            if abs(delta) > 0.2:
                disagreements.append({
                    "resume_id": e["resume_id"],
                    "ai_score": e["ai_score"],
                    "recruiter_score": e["recruiter_relevance"],
                    "delta": round(delta, 4),
                    "direction": "OVERSCORED" if delta > 0 else "UNDERSCORED",
                })

        n = max(len(entries), 1)
        return {
            "total_entries": len(entries),
            "advanced_ai_avg": avg(advanced, "ai_score"),
            "advanced_recruiter_avg": avg(advanced, "recruiter_relevance"),
            "rejected_ai_avg": avg(rejected, "ai_score") if rejected else 0,
            "overscoring_rate": round(sum(1 for d in disagreements if d["direction"] == "OVERSCORED") / n, 4),
            "underscoring_rate": round(sum(1 for d in disagreements if d["direction"] == "UNDERSCORED") / n, 4),
            "disagreements": disagreements,
        }

    def extract_patterns(self, job_id: str = None) -> dict:
        entries = self.store.load_for_job(job_id) if job_id else self.store.load_all()
        advance_reasons = Counter()
        reject_reasons = Counter()
        for e in entries:
            reasons = e.get("decision_reasons", [])
            if e["recruiter_decision"] == "ADVANCE":
                advance_reasons.update(reasons)
            elif e["recruiter_decision"] == "REJECT":
                reject_reasons.update(reasons)
        return {
            "advance_reasons": dict(advance_reasons.most_common(10)),
            "reject_reasons": dict(reject_reasons.most_common(10)),
        }
def compile_feedback_context(store: FeedbackStore, job_id: str = None) -> dict:
    """Compile feedback into a context object for the next scoring run."""
    analyzer = FeedbackAnalyzer(store)
    cal = analyzer.compute_calibration(job_id)
    patterns = analyzer.extract_patterns(job_id)

    weight_adjustments = []
    if cal.get("overscoring_rate", 0) > 0.3:
        weight_adjustments.append({
            "target": "W_HARD_SKILLS", "delta": -0.05,
            "reason": f"AI overscores {cal['overscoring_rate']:.0%} of candidates"
        })
    if cal.get("underscoring_rate", 0) > 0.3:
        weight_adjustments.append({
            "target": "W_TRANSFERABILITY", "delta": +0.05,
            "reason": f"AI underscores {cal['underscoring_rate']:.0%} — expand adjacency"
        })
    if patterns.get("reject_reasons", {}).get("weak_experience", 0) > 3:
        weight_adjustments.append({
            "target": "W_EXPERIENCE_CONTEXT", "delta": +0.05,
            "reason": "Recruiters frequently cite 'weak experience'"
        })

    return {
        "feedback_available": cal.get("total_entries", 0) > 0,
        "calibration": cal,
        "patterns": patterns,
        "weight_adjustments": weight_adjustments,
    }
def apply_weight_adjustments(base_weights: dict, context: dict) -> dict:
    """
    Apply feedback-derived weight adjustments with safety bounds.
    Max ±0.10 per adjustment. Re-normalized to sum to 1.0.
    """
    MAX_ADJ = 0.10
    adjusted = dict(base_weights)

    for adj in context.get("weight_adjustments", []):
        target = adj.get("target", "")
        delta = max(-MAX_ADJ, min(MAX_ADJ, adj.get("delta", 0)))
        if target in adjusted:
            adjusted[target] = adjusted[target] + delta

    # Re-normalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: round(v / total, 4) for k, v in adjusted.items()}

    return adjusted
