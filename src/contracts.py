"""
Pipeline Data Contracts — enforced schemas at every stage boundary.
"""

from dataclasses import dataclass, field
from typing import List, Optional
@dataclass
class SkillEvidence:
    """Single evidence item linking a JD requirement to resume content."""
    requirement: str
    status: str             # MATCHED | ADJACENT | GROUP | MISSING | MISSING_CRITICAL
    evidence_text: str
    evidence_location: str  # ontology | esco | none
    confidence: str         # HIGH | MEDIUM | LOW
    strength: float = 1.0   # 1.0=exact, 0.6=adjacent, 0.3=group, 0.0=missing

    def to_dict(self):
        return {
            "requirement": self.requirement,
            "status": self.status,
            "evidence_text": self.evidence_text,
            "evidence_location": self.evidence_location,
            "confidence": self.confidence,
            "strength": self.strength,
        }
@dataclass
class MatchRationale:
    """Complete explanation for a single match."""
    summary: str
    recommendation: str
    strengths: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def to_recruiter_text(self, name, score, confidence):
        lines = [
            "=" * 70,
            f"  CANDIDATE: {name}",
            f"  SCORE: {score:.2f} / 1.00  |  CONFIDENCE: {confidence}  |  {self.recommendation}",
            "=" * 70,
            f"\n  {self.summary}\n",
        ]

        if self.strengths:
            lines.append("  STRENGTHS:")
            for s in self.strengths:
                lines.append(f"    + {s}")

        if self.gaps:
            lines.append("\n  GAPS:")
            for g in self.gaps:
                lines.append(f"    - {g}")

        if self.flags:
            lines.append("\n  FLAGS:")
            for f in self.flags:
                lines.append(f"    ! {f}")

        return "\n".join(lines)
@dataclass
class ThreatReport:
    """Adversarial assessment for a single resume."""
    resume_id: str
    threat_level: str = "NONE"
    total_penalty: float = 0.0
    is_clean: bool = True
    injection_attempts: List[dict] = field(default_factory=list)
    invisible_char_count: int = 0
    homoglyph_count: int = 0
    duplicate_content_ratio: float = 0.0
    keyword_stuffing_score: float = 0.0
    flags: List[str] = field(default_factory=list)
@dataclass
class MatchResult:
    """Complete scored output for a single resume against a JD."""
    resume_id: str
    name: str
    final_score: float = 0.0
    confidence: str = "MEDIUM"
    rationale: Optional[MatchRationale] = None
    threat_report: Optional[ThreatReport] = None
    skill_detail: dict = field(default_factory=dict)
    stage_scores: dict = field(default_factory=dict)
    rank: int = 0
    label: float = -1.0
