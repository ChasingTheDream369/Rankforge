#!/usr/bin/env python3
# Run with: python3 demo.py  OR  /path/to/venv/bin/python demo.py
"""
Resume-JD Matching Engine v2 — Full Pipeline Runner

Usage:
  python demo.py                                          # Sample data (12 resumes)
  python demo.py --jd job.txt --resumes resumes.zip       # Zip of PDFs/DOCXs
  python demo.py --jd job.txt --resumes ./resume_folder/  # Folder of resumes
  python demo.py --jd job.txt --resumes candidate.pdf     # Single resume

Supported: .pdf .docx .doc .txt .html .rtf .png .jpg .jpeg .webp
"""

import sys
import os
import json
import zipfile
import tempfile
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import JD_DIR, RESUME_DIR, GOLDEN_DATASET_PATH
from src.ingestion.extractor import extract_text, EXTENSION_MAP
from src.pipeline import run_pipeline, load_sample_data
from src.evaluation.metrics import evaluate_full, load_golden_dataset

# Compliance module is in extras/ — import if available
try:
    from extras.compliance import (
        create_audit_record, save_audit_record, print_bias_report,
        generate_bias_audit_report,
    )
    HAS_COMPLIANCE = True
except ImportError:
    HAS_COMPLIANCE = False
# ============================================================
# Loaders — handle any input format
# ============================================================

def load_jd(jd_path):
    """Load JD from any supported file. Returns (jd_id, jd_text)."""
    p = Path(jd_path)
    if not p.exists():
        print(f"ERROR: JD not found: {jd_path}")
        sys.exit(1)
    text = extract_text(str(p))
    if not text or len(text.strip()) < 20:
        text = p.read_text(encoding='utf-8', errors='replace')
    return p.stem, text
def load_resumes(resume_path):
    """
    Load resumes from zip, directory, or single file.
    Returns {resume_id: {"text": str, "name": str}}
    """
    p = Path(resume_path)
    if not p.exists():
        print(f"ERROR: Resume path not found: {resume_path}")
        sys.exit(1)

    if p.suffix.lower() == '.zip':
        tmpdir = tempfile.mkdtemp(prefix="resumes_")
        print(f"  Extracting zip...")
        with zipfile.ZipFile(str(p), 'r') as zf:
            zf.extractall(tmpdir)
        return scan_dir(tmpdir)

    if p.is_dir():
        return scan_dir(str(p))

    if p.is_file():
        text = extract_text(str(p))
        if text and len(text.strip()) > 50:
            rid = p.stem
            return {rid: {"text": text, "name": rid.replace('_', ' ').replace('-', ' ').title()}}
        print(f"ERROR: Could not extract text from {p.name}")
        sys.exit(1)

    print(f"ERROR: Cannot process: {resume_path}")
    sys.exit(1)
def scan_dir(dir_path):
    """Recursively extract all supported files from a directory."""
    resumes = {}
    supported = set(EXTENSION_MAP.keys())
    for f in sorted(Path(dir_path).rglob('*')):
        if not f.is_file():
            continue
        if f.suffix.lower() not in supported:
            continue
        if any(part.startswith('.') or part.startswith('__') for part in f.parts):
            continue
        rid = f.stem
        if rid in resumes:
            rid = f"{f.parent.name}_{rid}"
        text = extract_text(str(f))
        if text and len(text.strip()) > 50:
            name = rid.replace('_', ' ').replace('-', ' ').title()
            resumes[rid] = {"text": text, "name": name}
            print(f"    + {f.name} ({len(text)} chars)")
        else:
            print(f"    - {f.name} (skip: extraction failed)")
    return resumes
# ============================================================
# Output
# ============================================================

def print_rankings(results):
    print(f"\n{'=' * 78}")
    print(f"  RANKINGS ({len(results)} candidates)")
    print(f"{'=' * 78}")
    print(f"  {'Rank':<6}{'Name':<28}{'Score':<10}{'Conf':<8}{'Recommendation':<18}{'Threat'}")
    print("  " + "-" * 72)
    for r in results:
        threat = ""
        if r.threat_report and not r.threat_report.is_clean:
            threat = f"{r.threat_report.threat_level} ({r.threat_report.total_penalty:.2f})"
        rec = r.rationale.recommendation if r.rationale else ""
        print(f"  #{r.rank:<5}{r.name:<28}{r.final_score:<10.4f}{r.confidence:<8}{rec:<18}{threat}")
def print_explanations(results, top_k=5):
    print(f"\n{'=' * 78}")
    print(f"  DETAILED EXPLANATIONS (Top {min(top_k, len(results))})")
    print(f"{'=' * 78}")
    W = {"d1_skills": 0.40, "d2_seniority": 0.35, "d3_domain": 0.15, "d4_constraints": 0.10}
    DIM_LABELS = {
        "d1_skills":      "Hard Skills     (D1)",
        "d2_seniority":   "Exp. Depth      (D2)",
        "d3_domain":      "Domain Fit      (D3)",
        "d4_constraints": "Constraints     (D4)",
    }
    for r in results[:top_k]:
        rec = r.rationale.recommendation if r.rationale else ""
        mode = (r.stage_scores or {}).get("scoring_mode", "?")
        print(f"\n  {'─' * 74}")
        print(f"  #{r.rank}  {r.name}   Score: {r.final_score:.4f}   Conf: {r.confidence}   [{rec}]   (mode: {mode})")
        print(f"  {'─' * 74}")

        # LLM rationale paragraph
        if r.rationale and r.rationale.summary:
            print(f"\n  LLM RATIONALE:")
            # word-wrap at ~72 chars
            words = r.rationale.summary.split()
            line, lines = "    ", []
            for w in words:
                if len(line) + len(w) + 1 > 76:
                    lines.append(line)
                    line = "    " + w
                else:
                    line += (" " if line.strip() else "") + w
            if line.strip():
                lines.append(line)
            for l in lines:
                print(l)

        # Score decomposition table
        ss = r.stage_scores or {}
        print(f"\n  SCORE BREAKDOWN:")
        print(f"  {'─' * 58}")
        dim_composite = 0.0
        for key, label in DIM_LABELS.items():
            score = ss.get(key, 0.0)
            w = W[key]
            contrib = score * w
            dim_composite += contrib
            print(f"  {label} ×{w:.2f}   {score:.4f}  →  {contrib:.4f}")
        print(f"  {'─' * 58}")
        print(f"  {'Dim. Composite':30s}         {ss.get('dim_composite', dim_composite):.4f}")
        alpha = ss.get("ce_weight", 0.0)
        ce = ss.get("ce_sigmoid", ss.get("cross_encoder", 0.0))
        if alpha > 0:
            print(f"  {'Cross-Encoder':30s} α={alpha:.2f}   {ce:.4f}   (sigmoid)")
        print(f"  {'─' * 58}")
        print(f"  {'FINAL SCORE':30s}         {r.final_score:.4f}")
        if ss.get("bm25") is not None:
            print(f"\n  Retrieval:  BM25={ss.get('bm25', 0):.3f}  Dense={ss.get('dense', 0):.3f}  RRF={ss.get('rrf', 0):.4f}  CE-logit={ss.get('ce_logit', 0):.3f}")

        # Strengths & Gaps
        if r.rationale:
            if r.rationale.strengths:
                print(f"\n  STRENGTHS:")
                for s in r.rationale.strengths[:4]:
                    print(f"    + {s}")
            if r.rationale.gaps:
                print(f"\n  GAPS:")
                for g in r.rationale.gaps[:4]:
                    print(f"    - {g}")

        # Per-skill evidence
        sd = r.skill_detail or {}
        skills_checked = sd.get("skills_checked", [])
        if skills_checked:
            print(f"\n  SKILL EVIDENCE (D1 — top {min(8, len(skills_checked))}):")
            print(f"  {'Skill':<20} {'Level':<14} Evidence")
            print(f"  {'─' * 70}")
            for sk in skills_checked[:8]:
                skill = str(sk.get("skill", ""))[:18]
                level = str(sk.get("level", ""))[:12]
                evidence = str(sk.get("evidence", ""))[:38].replace("\n", " ")
                print(f"  {skill:<20} {level:<14} {evidence}")
        elif sd.get("matched") or sd.get("missing"):
            # fallback for deterministic mode
            print(f"\n  SKILL SUMMARY:")
            if sd.get("matched"):
                print(f"    Exact:    {', '.join(list(sd['matched'])[:8])}")
            if sd.get("adjacent_matched"):
                print(f"    Adjacent: {', '.join(list(sd['adjacent_matched'])[:6])}")
            if sd.get("missing"):
                print(f"    Missing:  {', '.join(list(sd['missing'])[:6])}")
def print_adversarial(results):
    flagged = [r for r in results if r.threat_report and not r.threat_report.is_clean]
    if not flagged:
        print(f"\n  Adversarial: all {len(results)} resumes clean")
        return
    print(f"\n{'=' * 78}")
    print(f"  ADVERSARIAL DETECTION ({len(flagged)} flagged)")
    print(f"{'=' * 78}")
    for r in flagged:
        tr = r.threat_report
        print(f"\n  {r.name} — {tr.threat_level} (penalty: {tr.total_penalty:.2f})")
        for flag in tr.flags:
            print(f"    ! {flag}")
def print_evaluation(results, golden_path):
    if not os.path.exists(golden_path):
        print(f"\n  No golden dataset — skipping evaluation metrics.")
        return
    golden = load_golden_dataset(golden_path)
    for jd_id, labels in golden.items():
        labeled = [r for r in results if r.resume_id in labels]
        if not labeled:
            continue
        metrics = evaluate_full(results, labels)
        print(f"\n{'=' * 78}")
        print(f"  EVALUATION vs GOLDEN DATASET  ({jd_id})")
        print(f"{'=' * 78}")

        # Metrics table
        cols = [
            ("nDCG@3",    metrics.get("ndcg@3")),
            ("nDCG@5",    metrics.get("ndcg@5")),
            ("nDCG@10",   metrics.get("ndcg@10")),
            ("MRR",       metrics.get("mrr")),
            ("P@3",       metrics.get("precision@3")),
            ("P@5",       metrics.get("precision@5")),
            ("Spearman",  metrics.get("spearman_rho")),
        ]
        cols = [(k, v) for k, v in cols if v is not None]
        header = "  ┌─" + "─┬─".join("─" * 8 for _ in cols) + "─┐"
        row_h  = "  │ " + " │ ".join(f"{k:^8}" for k, _ in cols) + " │"
        sep    = "  ├─" + "─┼─".join("─" * 8 for _ in cols) + "─┤"
        row_v  = "  │ " + " │ ".join(f"{v:^8.4f}" for _, v in cols) + " │"
        footer = "  └─" + "─┴─".join("─" * 8 for _ in cols) + "─┘"
        print(header); print(row_h); print(sep); print(row_v); print(footer)

        # Score separation
        if any(k in metrics for k in ("avg_good", "avg_partial", "avg_poor")):
            print(f"\n  Score separation:")
            parts = []
            for k, label in [("avg_good", "good"), ("avg_partial", "partial"), ("avg_poor", "poor")]:
                if k in metrics:
                    parts.append(f"avg_{label}={metrics[k]:.3f}")
            print(f"    {' | '.join(parts)}")
            if "gap_good_partial" in metrics:
                print(f"    gap good→partial: +{metrics['gap_good_partial']:.3f}", end="")
            if "gap_partial_poor" in metrics:
                print(f"   gap partial→poor: +{metrics['gap_partial_poor']:.3f}", end="")
            print()

        # Per-resume table
        print(f"\n  {'Resume':<28}{'Gold':<10}{'Rank':<8}{'Score':<10}{'OK?'}")
        print("  " + "─" * 60)
        score_map = {r.resume_id: r for r in results}
        for rid, label in sorted(labels.items(), key=lambda x: x[1], reverse=True):
            label_str = {1.0: "GOOD", 0.5: "PARTIAL", 0.0: "POOR"}.get(label, "?")
            r = score_map.get(rid)
            if r:
                ok = "✓" if (label >= 0.5 and r.rank <= 6) or (label == 0.0 and r.rank > 6) else "✗"
                print(f"  {rid:<28}{label_str:<10}#{r.rank:<7}{r.final_score:<10.4f}{ok}")
def print_compliance(results, jd_id, jd_text, resumes):
    if not HAS_COMPLIANCE:
        print(f"\n  Compliance module available in extras/ — run with extras for audit trail.")
        return
    print(f"\n{'=' * 78}")
    print(f"  COMPLIANCE & AUDIT")
    print(f"{'=' * 78}")
    record = create_audit_record(jd_id, jd_text, resumes, results)
    log_path = save_audit_record(record)
    print(f"  Audit ID:     {record.audit_id}")
    print(f"  Timestamp:    {record.timestamp}")
    print(f"  Config hash:  {record.config_hash}")
    print(f"  JD hash:      {record.jd_hash}")
    print(f"  Resumes:      {record.num_resumes}")
    print(f"  HITL needed:  {'YES' if record.hitl_required else 'No'}")
    print(f"  Adversarial:  {len(record.adversarial_flags)} flagged")
    print(f"  Audit log:    {log_path}")
    report = generate_bias_audit_report()
    print_bias_report(report)
def export_results(results):
    os.makedirs("evaluation", exist_ok=True)
    output = {
        "rankings": [
            {
                "rank": r.rank, "name": r.name, "resume_id": r.resume_id,
                "score": r.final_score, "confidence": r.confidence,
                "recommendation": r.rationale.recommendation if r.rationale else "",
                "stage_scores": r.stage_scores,
                "skill_detail": r.skill_detail,
                "strengths": r.rationale.strengths if r.rationale else [],
                "gaps": r.rationale.gaps if r.rationale else [],
                "threat_level": r.threat_report.threat_level if r.threat_report else "NONE",
            }
            for r in results
        ],
    }
    with open("evaluation/results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results: evaluation/results.json")
# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Resume-JD Matching Engine v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                                                    Sample data (12 resumes)
  python demo.py --jd job.txt --resumes resumes.zip                 Zip of PDFs/DOCXs
  python demo.py --jd job.txt --resumes ./resume_folder/            Folder of resumes
  python demo.py --jd job.txt --resumes r1.pdf r2.pdf r3.pdf        Multiple individual files
  python demo.py --jd jd1.txt jd2.txt --resumes ./resume_folder/    Multiple JDs

Supported formats: .pdf .docx .doc .txt .html .rtf .png .jpg .jpeg .webp

Where to put data:
  data/job_descriptions/   Job description .txt files
  data/resumes/            Resume files (any supported format)
  data/golden_dataset.json Manual labels for evaluation (optional)
        """)
    parser.add_argument('--jd', nargs='+', help='Job description file(s)')
    parser.add_argument('--resumes', nargs='+', help='Resumes: zip, folder, or individual files')
    args = parser.parse_args()

    print()
    print("=" * 78)
    print("  Resume-JD Matching Engine v2 — Full Pipeline")
    print("  Extract → Sanitize → Ontology+ESCO → BM25+Dense → RRF → Cross-encoder")
    print("  → 4D Weighted Score → Explain → Comply → Audit")
    print("=" * 78)

    if args.jd and args.resumes:
        # Custom mode
        print(f"\n  Mode: CUSTOM")

        # Load resumes (merge all paths)
        print(f"\n  Loading resumes: {args.resumes}")
        resumes = {}
        for rpath in args.resumes:
            resumes.update(load_resumes(rpath))
        if not resumes:
            print("ERROR: No resumes extracted")
            sys.exit(1)
        print(f"    Total: {len(resumes)} resumes loaded")

        # Run pipeline for each JD
        for jd_path in args.jd:
            print(f"\n  Loading JD: {jd_path}")
            jd_id, jd_text = load_jd(jd_path)
            print(f"    {jd_id} ({len(jd_text)} chars)")

            results = run_pipeline(jd_text, resumes, verbose=True)
            print_rankings(results)
            print_explanations(results, top_k=min(5, len(results)))
            print_adversarial(results)
            export_results(results)

    elif args.jd or args.resumes:
        parser.error("Both --jd and --resumes required for custom data")

    else:
        # Sample mode
        print(f"\n  Mode: SAMPLE DATA")
        jd_files, resumes = load_sample_data()
        if not jd_files or not resumes:
            print("ERROR: No sample data in data/ directories")
            sys.exit(1)

        jd_id = list(jd_files.keys())[0]
        jd_text = jd_files[jd_id]
        print(f"  JD: {jd_id} ({len(jd_text)} chars)")
        print(f"  Resumes: {len(resumes)}")

        results = run_pipeline(jd_text, resumes, verbose=True)
        print_rankings(results)
        print_explanations(results, top_k=3)
        print_adversarial(results)
        print_evaluation(results, GOLDEN_DATASET_PATH)
        export_results(results)

    print(f"\n{'=' * 78}")
    print(f"  Done.")
    print(f"{'=' * 78}")
if __name__ == "__main__":
    main()
