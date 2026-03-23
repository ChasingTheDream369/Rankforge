"""
Adversarial Sanitization Layer — runs BEFORE any semantic evaluation.

7 detectors (merged from both versions):
  1. Prompt injection (pattern-based, strips from text)
  2. Keyword stuffing (density + verb ratio + comma lists)
  3. Invisible text (zero-width, BOM, directional marks)
  4. Homoglyph normalization (Cyrillic → Latin)
  5. JD content duplication (n-gram overlap detection)
  6. Experience inflation (timeline vs claimed years)
  7. Credential density anomaly

Returns cleaned text + ThreatReport. Penalty is continuous (0.0-0.95),
not binary reject — degrades adversarial resumes proportionally.
"""

import re
import math
from typing import List, Tuple

from src.contracts import ThreatReport
# === Compiled Patterns ===

INJECTION_PATTERNS = [
    (r'(?i)ignore\s+(all\s+)?previous\s+instructions?', 0.9),
    (r'(?i)disregard\s+(all\s+)?(prior|previous|above)', 0.9),
    (r'(?i)forget\s+(everything|all)', 0.85),
    (r'(?i)(score|rank|rate)\s+(this|me|the\s+candidate)\s+.{0,20}(perfect|100|99|10/10|highest)', 0.95),
    (r'(?i)assign\s+(a\s+)?(perfect|maximum|highest)\s+(score|rank)', 0.9),
    (r'(?i)this\s+candidate\s+is\s+(a\s+)?perfect\s+match', 0.85),
    (r'(?i)override\s+(the\s+)?(system|scoring)', 0.9),
    (r'(?i)^system\s*:', 0.9),
    (r'(?i)\[system\]', 0.9),
    (r'(?i)you\s+are\s+now\s+(a|an)\s+', 0.7),
    (r'<!--.*?-->', 0.8),
    (r'(?i)\{\{.*?(system|instruction|prompt).*?\}\}', 0.85),
    (r'(?i)<\s*/?instruction\s*>', 0.9),
    (r'(?i)(output|print|show|reveal)\s+(the\s+)?(system\s+prompt|other\s+candidates?)', 0.8),
]

INVISIBLE_CHARS = re.compile(
    r'[\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064'
    r'\ufeff\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180e\uffa0]'
)

HOMOGLYPH_MAP = {
    '\u0430': 'a', '\u0435': 'e', '\u043e': 'o', '\u0440': 'p',
    '\u0441': 'c', '\u0443': 'y', '\u0445': 'x', '\u0456': 'i',
}
def detect_injections(text: str) -> Tuple[str, List[dict], float]:
    """Detect and strip prompt injections. Returns (cleaned, attempts, penalty)."""
    attempts = []
    penalty = 0.0
    cleaned = text

    for pattern, severity in INJECTION_PATTERNS:
        for m in re.finditer(pattern, cleaned, re.DOTALL):
            attempts.append({"matched": m.group()[:80], "severity": severity})
            penalty = max(penalty, severity * 0.5)

    # Strip HTML comments entirely
    cleaned = re.sub(r'<!--.*?-->', '', cleaned, flags=re.DOTALL)
    # Strip other injection patterns (replace with space)
    for pattern, _ in INJECTION_PATTERNS:
        cleaned = re.sub(pattern, ' ', cleaned, flags=re.DOTALL)

    return cleaned, attempts, penalty
def strip_invisible(text: str) -> Tuple[str, int]:
    """Remove zero-width and invisible unicode. Returns (cleaned, count)."""
    matches = INVISIBLE_CHARS.findall(text)
    return INVISIBLE_CHARS.sub('', text), len(matches)
def normalize_homoglyphs(text: str) -> Tuple[str, int]:
    """Replace Cyrillic homoglyphs with Latin equivalents."""
    count = 0
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch in HOMOGLYPH_MAP:
            chars[i] = HOMOGLYPH_MAP[ch]
            count += 1
    return ''.join(chars), count
def detect_jd_duplication(resume_text: str, jd_text: str) -> float:
    """Detect copy-pasted JD content via 4-gram overlap ratio."""
    if not jd_text.strip() or not resume_text.strip():
        return 0.0
    def ngrams(text, n=4):
        words = text.lower().split()
        return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))
    jd_ng = ngrams(jd_text)
    if not jd_ng:
        return 0.0
    return len(jd_ng & ngrams(resume_text)) / len(jd_ng)
def detect_keyword_stuffing(text: str) -> float:
    """Detect abnormal keyword density (0.0=normal, 1.0=heavily stuffed)."""
    words = text.lower().split()
    if len(words) < 20:
        return 0.0

    action_verbs = {'built', 'designed', 'implemented', 'developed', 'architected',
                    'managed', 'deployed', 'optimized', 'led', 'created', 'maintained',
                    'migrated', 'scaled', 'automated', 'configured', 'integrated'}
    from src.ingestion.ontology import SKILL_ONTOLOGY
    tech_keywords = set(SKILL_ONTOLOGY.keys())

    found = sum(1 for kw in tech_keywords if kw in text.lower())
    verbs = sum(1 for v in action_verbs if v in text.lower())
    ratio = found / max(verbs, 1)
    density = found / len(words)

    if density > 0.08 and ratio > 5:
        return min(0.9, density * 5)
    return 0.0
def detect_experience_inflation(text: str) -> float:
    """Detect claimed years vs actual timeline span."""
    text_lower = text.lower()
    claimed = re.findall(r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)', text_lower)
    max_claimed = max([int(y) for y in claimed], default=0)
    ranges = re.findall(r'(20\d{2})\s*[-–—to]+\s*(20\d{2}|present|current|now)', text_lower)

    if ranges and max_claimed > 0:
        years = [(int(s), 2026 if e in ('present', 'current', 'now') else int(e)) for s, e in ranges]
        actual_span = max(e for _, e in years) - min(s for s, _ in years)
        if max_claimed > actual_span + 2:
            return min(0.7, (max_claimed - actual_span) * 0.15)
    return 0.0
def detect_credential_anomaly(text: str) -> float:
    """Flag suspiciously dense certification listings (>7)."""
    certs = set()
    for p in [r'\baws\s*(?:saa|sap|dva|soa|dbs|mls|ans|scs)', r'\bcissp\b', r'\bccna\b',
              r'\bpmp\b', r'\bcka\b', r'\bckad\b', r'\bcks\b', r'\bgcp\s*(?:ace|pca)',
              r'\baz-\d{3}\b', r'\btogaf\b', r'\bscrum\s*master\b']:
        certs.update(re.findall(p, text.lower()))
    if len(certs) > 7:
        return min(0.6, (len(certs) - 7) * 0.1)
    return 0.0
RESUME_MARKERS = re.compile(
    r'(?i)\b(experience|education|skills|summary|objective|work history|employment|'
    r'projects?|certifications?|achievements?|bachelor|master|university|college|'
    r'github|linkedin|portfolio|contact|email|phone|\d{4}\s*[-–]\s*(present|\d{4}))\b'
)

def detect_non_resume(text: str) -> float:
    """Flag documents that lack basic resume markers — likely a wrong file upload."""
    words = text.split()
    if len(words) < 30:
        return 0.0  # too short to judge
    hits = len(RESUME_MARKERS.findall(text))
    # A genuine resume typically has 5+ of these markers
    if hits < 3:
        return 0.6  # strong signal: not a resume
    if hits < 5:
        return 0.25  # weak signal: possibly resume
    return 0.0


def sanitize(resume_text: str, jd_text: str, resume_id: str = "unknown") -> Tuple[str, ThreatReport]:
    """
    Full sanitization pipeline. Returns (cleaned_text, threat_report).
    Cleaned text has injections stripped and invisible chars removed.
    """
    report = ThreatReport(resume_id=resume_id)
    text = resume_text

    # Step 1: Strip invisible characters
    text, inv_count = strip_invisible(text)
    report.invisible_char_count = inv_count
    if inv_count > 0:
        report.flags.append(f"Removed {inv_count} invisible unicode characters")

    # Step 2: Normalize homoglyphs
    text, hg_count = normalize_homoglyphs(text)
    report.homoglyph_count = hg_count
    if hg_count > 5:
        report.flags.append(f"Normalized {hg_count} homoglyph characters")

    # Step 3: Detect + strip prompt injections
    text, injections, inj_penalty = detect_injections(text)
    report.injection_attempts = injections
    if injections:
        report.flags.append(f"CRITICAL: {len(injections)} prompt injection(s) detected and stripped")

    # Step 4: JD duplication
    dup_ratio = detect_jd_duplication(text, jd_text)
    report.duplicate_content_ratio = dup_ratio
    if dup_ratio > 0.3:
        report.flags.append(f"WARNING: {dup_ratio:.0%} of JD text found verbatim in resume")

    # Step 5: Keyword stuffing
    stuffing = detect_keyword_stuffing(text)
    report.keyword_stuffing_score = stuffing
    if stuffing > 0.4:
        report.flags.append(f"WARNING: Abnormal keyword density (score={stuffing:.2f})")

    # Step 6: Experience inflation
    inflation = detect_experience_inflation(text)
    if inflation > 0:
        report.flags.append(f"WARNING: Timeline inconsistency (penalty={inflation:.2f})")

    # Step 7: Credential anomaly
    cred = detect_credential_anomaly(text)
    if cred > 0:
        report.flags.append(f"WARNING: Excessive certifications (penalty={cred:.2f})")

    # Step 8: Not-a-resume detector
    non_resume = detect_non_resume(text)
    if non_resume >= 0.5:
        report.flags.append("WARNING: Document does not appear to be a resume (missing standard resume sections)")
    elif non_resume > 0:
        report.flags.append("INFO: Document has few resume markers — verify correct file was uploaded")

    # Compute total penalty (compound, capped at 0.95)
    penalties = [inj_penalty]
    if inv_count > 10: penalties.append(0.15)
    if hg_count > 5: penalties.append(0.1)
    if dup_ratio > 0.3: penalties.append(min(dup_ratio, 0.3))
    if stuffing > 0.4: penalties.append(stuffing * 0.4)
    if inflation > 0: penalties.append(inflation * 0.3)
    if cred > 0: penalties.append(cred * 0.2)
    if non_resume > 0: penalties.append(non_resume * 0.5)

    report.total_penalty = min(0.95, 1.0 - math.prod(1.0 - p for p in penalties))

    # Threat level
    tp = report.total_penalty
    if tp < 0.05: report.threat_level = "NONE"
    elif tp < 0.15: report.threat_level = "LOW"
    elif tp < 0.35: report.threat_level = "MEDIUM"
    elif tp < 0.60: report.threat_level = "HIGH"
    else: report.threat_level = "CRITICAL"

    report.is_clean = report.threat_level in ("NONE", "LOW")

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text, report
