#!/usr/bin/env python3
"""
Ablation Study — Progressive Improvement Comparison

Runs the golden dataset through 5 increasingly sophisticated approaches
and compares nDCG, MRR, P@k, and Spearman at each stage.

Approach 1: Baseline — TF-IDF cosine similarity only
Approach 2: TF-IDF + BM25 (dual sparse)
Approach 3: BM25 + Bi-encoder + RRF (hybrid retrieval)
Approach 4: Approach 3 + Cross-encoder reranking
Approach 5: Approach 4 + Ontology graph (our full system)

Usage:
  python ablation.py                                    # sample data
  python ablation.py --jd job.txt --resumes ./resumes/ # custom data
  python ablation.py --jd job.txt --resumes r1.pdf r2.pdf r3.pdf
"""

import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import GOLDEN_DATASET_PATH, JD_DIR, RESUME_DIR
from src.evaluation.metrics import ndcg_at_k, mrr, precision_at_k, spearman_rho
from src.pipeline import load_sample_data


def _load_jsonl(path: str) -> dict:
    """Load a file that is either a single JSON object or multiple JSON objects (JSONL)."""
    with open(path) as f:
        content = f.read().strip()
    result = {}
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        remaining = content[pos:].lstrip()
        if not remaining:
            break
        # Skip comma separators between objects (non-standard but common)
        if remaining.startswith(','):
            pos += len(content[pos:]) - len(remaining) + 1
            continue
        skip = len(content[pos:]) - len(remaining)
        try:
            obj, end = decoder.raw_decode(remaining)
        except Exception:
            break
        result.update(obj)
        pos += skip + end
    return result

def load_data(jd_path=None, resume_paths=None):
    """Load JD + resumes + labels. Accepts custom paths or falls back to sample data."""
    if jd_path and resume_paths:
        # Custom mode — mirror demo.py loaders
        from src.ingestion.extractor import extract_text, EXTENSION_MAP
        from pathlib import Path
        import zipfile, tempfile

        p = Path(jd_path)
        jd_text = extract_text(str(p))
        if not jd_text or len(jd_text.strip()) < 20:
            jd_text = p.read_text(encoding='utf-8', errors='replace')
        jd_id = p.stem

        def _scan(dir_path):
            out = {}
            supported = set(EXTENSION_MAP.keys())
            for f in sorted(Path(dir_path).rglob('*')):
                if not f.is_file() or f.suffix.lower() not in supported:
                    continue
                if any(x.startswith('.') or x.startswith('__') for x in f.parts):
                    continue
                text = extract_text(str(f))
                if text and len(text.strip()) > 50:
                    out[f.stem] = {"text": text, "name": f.stem}
            return out

        resumes = {}
        for rpath in resume_paths:
            p2 = Path(rpath)
            if p2.suffix.lower() == '.zip':
                tmpdir = tempfile.mkdtemp()
                with zipfile.ZipFile(str(p2)) as zf:
                    zf.extractall(tmpdir)
                resumes.update(_scan(tmpdir))
            elif p2.is_dir():
                resumes.update(_scan(str(p2)))
            elif p2.is_file():
                text = extract_text(str(p2))
                if text and len(text.strip()) > 50:
                    resumes[p2.stem] = {"text": text, "name": p2.stem}
    else:
        # Load sample resumes (txt) + ablation_resumes (any format), pick JD by best label coverage
        from src.ingestion.extractor import extract_directory

        jd_files, txt_resumes = load_sample_data()

        # Scan data/ablation_resumes/ if it exists
        ablation_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ablation_resumes")
        ablation_resumes = {}
        if os.path.isdir(ablation_dir):
            extracted = extract_directory(ablation_dir)
            for stem, text in extracted.items():
                ablation_resumes[stem] = {"text": text, "name": stem}

        all_resumes = {**txt_resumes, **ablation_resumes}

        if os.path.exists(GOLDEN_DATASET_PATH):
            golden = _load_jsonl(GOLDEN_DATASET_PATH)
            best_jd = max(
                (k for k in jd_files if k in golden),
                key=lambda k: len(set(golden[k].keys()) & set(all_resumes.keys())),
                default=None,
            )
            jd_id = best_jd or list(jd_files.keys())[0]
        else:
            jd_id = list(jd_files.keys())[0]

        jd_text = jd_files[jd_id]
        resumes = all_resumes

    # Load labels and attach to resume dicts so pipeline.py can use them
    labels = {}
    if os.path.exists(GOLDEN_DATASET_PATH):
        golden = _load_jsonl(GOLDEN_DATASET_PATH)
        labels = golden.get(jd_id, {})
        for rid, lv in labels.items():
            if rid in resumes:
                resumes[rid]["label"] = lv

    if not labels:
        print(f"  WARNING: No golden labels found for JD '{jd_id}' in {GOLDEN_DATASET_PATH}")
        print(f"  Add an entry like: {{\"{jd_id}\": {{\"resume_id\": 1.0, ...}}}}")

    return jd_text, resumes, labels, jd_id

def evaluate_ranking(ranked_ids, labels):
    """Given a list of resume_ids in ranked order, compute metrics against gold labels."""
    relevances = [labels.get(rid, 0.0) for rid in ranked_ids]
    n = len(ranked_ids)

    ideal_order = sorted(labels.keys(), key=lambda k: labels[k], reverse=True)
    id_to_pred = {rid: rank + 1 for rank, rid in enumerate(ranked_ids)}
    pred_ranks = [id_to_pred.get(rid, n) for rid in ideal_order]
    true_ranks = list(range(1, len(ideal_order) + 1))

    return {
        "ndcg@3":  ndcg_at_k(relevances, 3),
        "ndcg@5":  ndcg_at_k(relevances, 5),
        "ndcg@10": ndcg_at_k(relevances, 10),
        "mrr":     mrr(relevances),
        "p@3":     precision_at_k(relevances, 3),
        "p@5":     precision_at_k(relevances, 5),
        "spearman": spearman_rho(pred_ranks, true_ranks),
    }
# ============================================================
# Approach 1: TF-IDF cosine similarity only
# ============================================================
def run_tfidf_only(jd_text, resumes):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    doc_ids = list(resumes.keys())
    texts = [resumes[rid]["text"] for rid in doc_ids]

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                            sublinear_tf=True, stop_words='english')
    all_texts = [jd_text] + texts
    matrix = tfidf.fit_transform(all_texts)
    scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
    return [rid for rid, _ in ranked], {rid: round(float(s), 4) for rid, s in ranked}
# ============================================================
# Approach 2: TF-IDF + BM25
# ============================================================
def run_tfidf_bm25(jd_text, resumes):
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from rank_bm25 import BM25Okapi

    doc_ids = list(resumes.keys())
    texts = [resumes[rid]["text"] for rid in doc_ids]

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                            sublinear_tf=True, stop_words='english')
    all_texts = [jd_text] + texts
    matrix = tfidf.fit_transform(all_texts)
    tfidf_scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

    # BM25
    tokenized = [re.findall(r'\b\w+\b', t.lower()) for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(re.findall(r'\b\w+\b', jd_text.lower()))

    # Simple RRF on both
    n = len(doc_ids)
    tfidf_ranks = {doc_ids[i]: rank + 1 for rank, i in enumerate(np.argsort(-tfidf_scores))}
    bm25_ranks = {doc_ids[i]: rank + 1 for rank, i in enumerate(np.argsort(-bm25_scores))}

    rrf = {}
    for rid in doc_ids:
        rrf[rid] = 1.0 / (60 + tfidf_ranks[rid]) + 1.0 / (60 + bm25_ranks[rid])

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    return [rid for rid, _ in ranked], {rid: round(float(s), 4) for rid, s in ranked}
# ============================================================
# Approach 3: BM25 + Bi-encoder + RRF
# ============================================================
def run_hybrid_rrf(jd_text, resumes):
    import re
    from rank_bm25 import BM25Okapi

    doc_ids = list(resumes.keys())
    texts = [resumes[rid]["text"] for rid in doc_ids]

    # BM25
    tokenized = [re.findall(r'\b\w+\b', t.lower()) for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(re.findall(r'\b\w+\b', jd_text.lower()))

    # Bi-encoder (or TF-IDF fallback)
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        doc_embs = encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        q_emb = encoder.encode([jd_text], show_progress_bar=False, normalize_embeddings=True)
        dense_scores = np.dot(doc_embs, q_emb.T).flatten()
    except ImportError:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
        mat = tfidf.fit_transform([jd_text] + texts)
        dense_scores = cos_sim(mat[0:1], mat[1:]).flatten()

    # RRF
    n = len(doc_ids)
    bm25_ranks = {doc_ids[i]: rank + 1 for rank, i in enumerate(np.argsort(-bm25_scores))}
    dense_ranks = {doc_ids[i]: rank + 1 for rank, i in enumerate(np.argsort(-dense_scores))}

    rrf = {}
    for rid in doc_ids:
        rrf[rid] = 1.0 / (60 + bm25_ranks[rid]) + 1.0 / (60 + dense_ranks[rid])

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    return [rid for rid, _ in ranked], {rid: round(float(s), 4) for rid, s in ranked}
# ============================================================
# Approach 4: BM25 + Bi-encoder + RRF + Cross-encoder
# ============================================================
def run_hybrid_crossencoder(jd_text, resumes):
    ranked_ids, rrf_scores = run_hybrid_rrf(jd_text, resumes)

    # Take top-20 from RRF, rerank with cross-encoder
    top_ids = ranked_ids[:20]
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [[jd_text, resumes[rid]["text"]] for rid in top_ids]
        ce_scores = ce.predict(pairs, show_progress_bar=False)
        ce_norm = [1.0 / (1.0 + np.exp(-s)) for s in ce_scores]

        reranked = sorted(zip(top_ids, ce_norm), key=lambda x: x[1], reverse=True)
        final_order = [rid for rid, _ in reranked]
        final_scores = {rid: round(float(s), 4) for rid, s in reranked}

        # Append remaining
        for rid in ranked_ids[20:]:
            final_order.append(rid)
            final_scores[rid] = 0.0

        return final_order, final_scores
    except ImportError:
        print("  (cross-encoder not available, using RRF scores)")
        return ranked_ids, rrf_scores
# ============================================================
# Approach 5: Full system (hybrid + cross-encoder + ontology graph)
# ============================================================
def run_full_system(jd_text, resumes):
    from src.pipeline import run_pipeline
    results = run_pipeline(jd_text, resumes, verbose=False)
    ranked_ids = [r.resume_id for r in results]
    scores = {r.resume_id: r.final_score for r in results}
    return ranked_ids, scores
# ============================================================
# Main
# ============================================================
METRIC_COLS = [
    ("nDCG@3",   "ndcg@3"),
    ("nDCG@5",   "ndcg@5"),
    ("nDCG@10",  "ndcg@10"),
    ("MRR",      "mrr"),
    ("P@3",      "p@3"),
    ("P@5",      "p@5"),
    ("Spearman", "spearman"),
]

def _fmt(v):
    return f"{v:.4f}" if v is not None else "  —   "

def _box_table(all_metrics):
    """Print a box-bordered comparison table with winner markers."""
    col_w = 8
    name_w = 32
    # find best per metric
    best = {}
    for _, key in METRIC_COLS:
        vals = [m[key] for m in all_metrics if key in m]
        best[key] = max(vals) if vals else None

    hdr  = "  ┌─" + "─" * name_w + "─┬─" + "─┬─".join("─" * col_w for _ in METRIC_COLS) + "─┐"
    row0 = "  │ " + f"{'Approach':<{name_w}}" + " │ " + " │ ".join(f"{h:^{col_w}}" for h, _ in METRIC_COLS) + " │"
    sep  = "  ├─" + "─" * name_w + "─┼─" + "─┼─".join("─" * col_w for _ in METRIC_COLS) + "─┤"
    foot = "  └─" + "─" * name_w + "─┴─" + "─┴─".join("─" * col_w for _ in METRIC_COLS) + "─┘"

    print(hdr); print(row0); print(sep)
    for m in all_metrics:
        cells = []
        for _, key in METRIC_COLS:
            v = m.get(key)
            mark = "*" if (v is not None and best.get(key) is not None and abs(v - best[key]) < 1e-9) else " "
            cells.append(f"{_fmt(v):>{col_w-1}}{mark}")
        name = m["name"][:name_w]
        print(f"  │ {name:<{name_w}} │ " + " │ ".join(cells) + " │")
    print(foot)
    print("  * = best in column")

def _delta_table(all_metrics):
    """Print improvement deltas vs baseline for every metric."""
    base = all_metrics[0]
    col_w = 9
    name_w = 32
    hdr  = "  ┌─" + "─" * name_w + "─┬─" + "─┬─".join("─" * col_w for _ in METRIC_COLS) + "─┐"
    row0 = "  │ " + f"{'Approach':<{name_w}}" + " │ " + " │ ".join(f"{h:^{col_w}}" for h, _ in METRIC_COLS) + " │"
    sep  = "  ├─" + "─" * name_w + "─┼─" + "─┼─".join("─" * col_w for _ in METRIC_COLS) + "─┤"
    foot = "  └─" + "─" * name_w + "─┴─" + "─┴─".join("─" * col_w for _ in METRIC_COLS) + "─┘"

    print(hdr); print(row0); print(sep)
    for m in all_metrics[1:]:
        cells = []
        for _, key in METRIC_COLS:
            d = m.get(key, 0) - base.get(key, 0)
            sign = "+" if d >= 0 else ""
            cells.append(f"{sign}{d:.4f}".center(col_w))
        name = m["name"][:name_w]
        print(f"  │ {name:<{name_w}} │ " + " │ ".join(cells) + " │")
    print(foot)

def main():
    parser = argparse.ArgumentParser(description="Ablation Study — 5-approach metric comparison")
    parser.add_argument('--jd', help='Job description file')
    parser.add_argument('--resumes', nargs='+', help='Resumes: folder, zip, or individual files')
    args = parser.parse_args()

    print("=" * 80)
    print("  ABLATION STUDY — Progressive Improvement vs Golden Dataset")
    print("  5 increasingly sophisticated approaches compared on nDCG / MRR / P@k")
    print("=" * 80)

    jd_text, resumes, labels, jd_id = load_data(args.jd, args.resumes)
    print(f"\n  JD: {jd_id} | Resumes: {len(resumes)} | Labels: {len(labels)}")
    if not labels:
        print("  Cannot run without golden labels. Exiting.")
        return

    label_str = {1.0: "GOOD", 0.5: "PARTIAL", 0.0: "POOR"}

    approaches = [
        ("1. TF-IDF cosine only",      run_tfidf_only),
        ("2. TF-IDF + BM25 (RRF)",     run_tfidf_bm25),
        ("3. BM25 + Bi-encoder (RRF)", run_hybrid_rrf),
        ("4. + Cross-encoder rerank",  run_hybrid_crossencoder),
        ("5. + Ontology graph (full)", run_full_system),
    ]

    all_metrics = []

    for name, fn in approaches:
        print(f"\n  ── {name} ──")
        ranked_ids, scores = fn(jd_text, resumes)
        metrics = evaluate_ranking(ranked_ids, labels)
        metrics["name"] = name
        all_metrics.append(metrics)

        # Compact table: only show labeled resumes in their predicted position
        labeled_rows = [(ranked_ids.index(rid) + 1 if rid in ranked_ids else 999,
                         rid, labels[rid], scores.get(rid, 0.0))
                        for rid in labels]
        labeled_rows.sort(key=lambda x: x[0])
        print(f"  {'Rank':<6}{'Resume':<26}{'Gold':<10}{'Score':<8}{'OK?'}")
        print(f"  {'─' * 54}")
        for pred_rank, rid, gold_label, score in labeled_rows:
            ok = "✓" if (gold_label >= 0.5 and pred_rank <= 6) or (gold_label == 0.0 and pred_rank > 6) else "✗"
            print(f"  #{pred_rank:<5}{rid:<26}{label_str.get(gold_label,'?'):<10}{score:<8.4f}{ok}")

        # Inline metric snapshot
        m = metrics
        print(f"\n  nDCG@3={m['ndcg@3']:.4f}  nDCG@5={m['ndcg@5']:.4f}  nDCG@10={m['ndcg@10']:.4f}"
              f"  MRR={m['mrr']:.4f}  P@3={m['p@3']:.4f}  P@5={m['p@5']:.4f}  Spearman={m['spearman']:.4f}")

    # ── Comparison table ──
    print(f"\n\n{'=' * 80}")
    print(f"  METRIC COMPARISON  (vs golden dataset: {jd_id})")
    print(f"{'=' * 80}\n")
    _box_table(all_metrics)

    # ── Delta table ──
    print(f"\n  IMPROVEMENT OVER BASELINE (approach 1 = 0)\n")
    _delta_table(all_metrics)

    # Save
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/ablation_results.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved: evaluation/ablation_results.json")
    print(f"\n{'=' * 80}")

if __name__ == "__main__":
    main()
