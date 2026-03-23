#!/usr/bin/env python3
"""
Score all resumes in data/ablation_resumes (and data/resumes) against a JD using the full 4D algo.
No retrieval — direct scoring only. Outputs D1–D4, layer justification, and final scores.

Usage:
  python score_ablation_resumes.py                      # default JD, all resumes
  python score_ablation_resumes.py --jd job.txt         # custom JD
  python score_ablation_resumes.py --resumes ./folder/   # custom resume folder
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.extractor import extract_directory, extract_text, EXTENSION_MAP
from pathlib import Path


def load_resumes(resume_paths):
    """Load resumes from dir(s), individual files, or zip."""
    import zipfile
    import tempfile

    def _scan(dir_path):
        out = {}
        supported = set(EXTENSION_MAP.keys())
        for f in sorted(Path(dir_path).rglob("*")):
            if not f.is_file() or f.suffix.lower() not in supported:
                continue
            if any(x.startswith(".") or x.startswith("__") for x in f.parts):
                continue
            text = extract_text(str(f))
            if text and len(text.strip()) > 50:
                out[f.stem] = {"text": text, "name": f.stem}
        return out

    resumes = {}
    for rpath in resume_paths:
        p = Path(rpath)
        if p.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(str(p)) as zf:
                    zf.extractall(tmpdir)
                resumes.update(_scan(tmpdir))
        elif p.is_dir():
            resumes.update(_scan(str(p)))
        elif p.is_file():
            text = extract_text(str(p))
            if text and len(text.strip()) > 50:
                resumes[p.stem] = {"text": text, "name": p.stem}
    return resumes


def main():
    parser = argparse.ArgumentParser(description="Score resumes with full 4D algo (no retrieval)")
    parser.add_argument("--jd", help="Job description file (default: data/job_descriptions/senior_backend_finpay.txt)")
    parser.add_argument("--resumes", nargs="+", help="Resume folder(s) or file(s)")
    parser.add_argument("-o", "--output", default="evaluation/ablation_scores.json", help="Output JSON path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "data")

    # JD
    if args.jd:
        jd_path = Path(args.jd)
        jd_text = extract_text(str(jd_path)) or jd_path.read_text(encoding="utf-8", errors="replace")
        jd_id = jd_path.stem
    else:
        jd_path = Path(data_dir) / "job_descriptions" / "senior_backend_finpay.txt"
        jd_text = jd_path.read_text(encoding="utf-8", errors="replace")
        jd_id = "senior_backend_finpay"

    # Resumes
    if args.resumes:
        resumes = load_resumes(args.resumes)
    else:
        resumes = {}
        resume_dir = os.path.join(data_dir, "resumes")
        ablation_dir = os.path.join(data_dir, "ablation_resumes")
        if os.path.isdir(resume_dir):
            extracted = extract_directory(resume_dir)
            for stem, text in extracted.items():
                if text and len(text.strip()) > 50:
                    resumes[stem] = {"text": text, "name": stem}
        if os.path.isdir(ablation_dir):
            extracted = extract_directory(ablation_dir)
            for stem, text in extracted.items():
                if text and len(text.strip()) > 50:
                    resumes[stem] = {"text": text, "name": stem}

    if not resumes:
        print("No resumes found. Use --resumes to specify folder(s).")
        return 1

    from src.scoring.scorer import score_resume

    print(f"JD: {jd_id} | Resumes: {len(resumes)}")
    print("Scoring each resume (4D: D1 skills, D2 seniority, D3 domain, D4 constraints)...\n")

    results = []
    for rid, rdata in sorted(resumes.items()):
        if args.verbose:
            print(f"  {rid}...", end=" ", flush=True)
        try:
            scored = score_resume(
                jd_text=jd_text,
                resume_text=rdata["text"],
                ce_logit=0.0,
                n_candidates=1,
                adversarial_penalty=0.0,
                verbose=False,
            )
            row = {
                "resume_id": rid,
                "name": rdata.get("name", rid),
                "final_score": scored["final_score"],
                "d1_skills": scored["d1_skills"],
                "d2_seniority": scored["d2_seniority"],
                "d3_domain": scored["d3_domain"],
                "d4_constraints": scored["d4_constraints"],
                "confidence": scored["confidence"],
                "recommendation": scored["recommendation"],
                "rationale": scored.get("rationale", ""),
                "strengths": scored.get("strengths", []),
                "gaps": scored.get("gaps", []),
                "skill_detail": scored.get("skill_detail", {}),
                "seniority_detail": scored.get("seniority_detail", {}),
                "domain_detail": scored.get("domain_detail", {}),
                "constraint_detail": scored.get("constraint_detail", []),
            }
            results.append(row)
            if args.verbose:
                print(f"D1={scored['d1_skills']:.2f} D2={scored['d2_seniority']:.2f} D3={scored['d3_domain']:.2f} D4={scored['d4_constraints']:.2f} → {scored['final_score']:.2f}")
        except Exception as e:
            if args.verbose:
                print(f"FAIL: {e}")
            results.append({
                "resume_id": rid,
                "name": rdata.get("name", rid),
                "error": str(e),
                "final_score": 0.0,
            })

    # Sort by final_score descending
    results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i

    out = {"jd_id": jd_id, "n_resumes": len(resumes), "results": results}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved: {args.output}")
    print("\nTop 5:")
    for r in results[:5]:
        err = f" [ERROR: {r['error']}]" if "error" in r else ""
        print(f"  #{r['rank']} {r['resume_id']}: {r.get('final_score', 0):.2f}{err}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
