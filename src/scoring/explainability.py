"""
Explainability — generates recruiter-facing rationale from skill overlap data.
No LLM calls. Deterministic: same inputs produce same output.
"""

from src.contracts import MatchRationale


def generate_rationale(name, score, confidence, recommendation,
                       skill_detail=None, threat_flags=None, **kwargs):
    """Generate explanation from skill overlap analysis."""

    skill_detail = skill_detail or {}
    matched = skill_detail.get("matched", [])
    adjacent = skill_detail.get("adjacent_matched", [])
    group = skill_detail.get("group_matched", [])
    missing = skill_detail.get("missing", [])
    total = len(matched) + len(adjacent) + len(group) + len(missing)

    strengths = []
    if matched:
        strengths.append(f"Matches {len(matched)}/{total} required skills: {', '.join(matched[:5])}")
    if adjacent:
        strengths.append(f"Transferable skills: {', '.join(adjacent[:3])}")
    if group:
        strengths.append(f"Same skill group: {', '.join(group[:3])}")

    gaps = []
    if missing:
        gaps.append(f"Missing: {', '.join(missing[:5])}")

    flags = list(threat_flags) if threat_flags else []
    if confidence == "LOW" and score > 0.4:
        flags.append("High score but weak evidence — manual review recommended.")
    if len(adjacent) > len(matched) and total > 0:
        flags.append("More skills matched via adjacency than directly — possible career pivoter.")

    openers = {
        "STRONG_MATCH": f"{name} is a strong match.",
        "GOOD_MATCH": f"{name} shows solid alignment.",
        "PARTIAL_MATCH": f"{name} partially matches, with notable gaps.",
        "WEAK_MATCH": f"{name} has limited overlap.",
        "NO_MATCH": f"{name} does not align with core requirements.",
    }
    opener = openers.get(recommendation, f"{name} was evaluated.")

    if matched:
        detail = f"Directly matches {len(matched)}/{total} skills"
        if adjacent:
            detail += f", {len(adjacent)} via adjacent tools"
        detail += "."
    else:
        detail = f"No direct skill matches among {total} required."

    summary = f"{opener} {detail} (Confidence: {confidence})"

    return MatchRationale(
        summary=summary,
        recommendation=recommendation,
        strengths=strengths[:5],
        gaps=gaps[:5],
        flags=flags,
    )
