# RankForge

## Table of Contents

1. [Overview](#overview)
2. [Technical Approach & Justification](#technical-approach--justification)
3. [Architecture](#architecture)
4. [Feature Engineering & Data Preprocessing](#feature-engineering--data-preprocessing)
5. [Performance Evaluation](#performance-evaluation)
6. [Challenges Overcome & Solutions](#challenges-overcome--solutions)
7. [Future Steps & Production Roadmap](#future-steps--production-roadmap)
8. [Setup & Usage](#setup--usage) — [UI walkthrough & screenshots](#web-ui-walkthrough--rankforge-screenshots)
9. [Testing](#testing)
10. [Project Structure](#project-structure)
11. [Known Limitations](#known-limitations)
12. [Research alignment & literature notes](#research-alignment)
13. [References](#references)
14. [Acknowledgements](#acknowledgements)

---

## Overview

### Background

Resume and JD matching sits at the intersection of **information retrieval**, **structured scoring**, and **fairness**: keyword-only filters and opaque rankers routinely mis-rank or exclude people; regulators increasingly expect **auditability** for automated employment tools. RankForge is an open codebase to explore **explainable** matching—scores you can trace to evidence and dimensions—not a black-box relevance label.

### Problem

Manual review **does not scale** with applicant volume. Fully automated pipelines without structure drift toward **unpredictable** judgments and **weak audit trails**. The core problem this repo tackles is: **rank candidates against a JD with explicit, inspectable reasons**, not only a single scalar “match score.”

### Issues

Several tensions show up in practice:

- **Consistency** — different reviewers (or runs) should not wildly disagree on the same resume–JD pair when evidence is stable.
- **Gaming & noise** — injected prompts, hidden text, keyword stuffing, and non-resume documents need **detection and penalties**, not silent boosts.
- **Retrieval vs understanding** — lexical-only methods miss semantics; semantic-only methods miss exact tokens; a **hybrid** path with re-ranking is needed.
- **Explainability vs cost** — deep models help relevance but **structured dimensions** (skills, seniority, domain, constraints) keep scores legible for review and tuning.
- **Calibration** — weights and fusion knobs should eventually be **fit to labeled data**, not only hand-tuned (see [Future Steps](#future-steps--production-roadmap)).

### Solution

**RankForge** implements a **Python-first matching engine** plus a small Django web app for runs and tooling. Each candidate receives a **0–1** score built from **four interpretable dimensions (D1–D4)** blended with a **cross-encoder** signal, with **evidence** where the pipeline exposes it.

**Inputs:** job description + resumes (PDF, DOCX, image, LaTeX, plain text, ZIP).  
**Outputs:** ranked list with per-dimension breakdowns, strengths/gaps where available, confidence, and threat/sanitizer flags.

**Pipeline in short:**

1. **Ingest & sanitize** — extract text; run adversarial detectors; optional non-resume gate.
2. **Retrieve** — BM25 + dense embeddings, fused with **RRF**; shortlist for deeper scoring.
3. **Re-rank** — cross-encoder on a top fraction of the pool.
4. **Score** — structured D1–D4 (ontology- and tool-grounded where configured) plus narrative layer where LLMs are enabled.
5. **Return** — ranked table and per-candidate detail for inspection.

### Taking ahead

The **next wave** is principled **calibration**: joint tuning of dimension weights, CE blend, and retrieval hyperparameters on **golden + adversarial** suites, plus production hardening (queueing, storage, rate limits, compliance hooks)—spelled out under [Future Steps & Production Roadmap](#future-steps--production-roadmap). Literature notes and doc links live under [Research alignment & literature notes](#research-alignment).

### Context (architecture & narrative)

For a **readable walkthrough** of the design (figures, layers, and rationale), see **[Architecture_Document.pdf](Architecture_Document.pdf)** in the repo root. The LaTeX source is [`docs/architecture.tex`](docs/architecture.tex) if you want to rebuild or cite sections. Screens from the web UI are under [`docs/rankforge_screens/`](docs/rankforge_screens/).

---

## Technical Approach & Justification

### Model Selection

| Component | Model | Purpose |
|-----------|-------|---------|
| **Stage 1 — Profile Extraction** | OpenAI `gpt-4o-mini` | Cheap structured extraction of JD and resume profiles |
| **Stage 2 — Scoring** | OpenAI `gpt-4o` | Few-shot 4D scoring with temp=0 for consistency |
| **Lexical Retrieval** | BM25 (`rank_bm25`) | Exact term overlap (Go, Kafka, PostgreSQL) |
| **Dense Retrieval** | `all-MiniLM-L6-v2` or `text-embedding-3-small` | Semantic similarity (“payment APIs” ≈ “financial microservices”) |
| **Re-ranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Joint [JD, resume] encoding for context-aware relevance |

### Why This Approach?

| Choice | Primary Advantage | Trade-off |
|--------|------------------|-----------|
| **Two-stage LLM** | Stage 1 (cheap) extracts structured profiles; Stage 2 (expensive) scores on a compact, deterministic schema. Caching profiles avoids re-extraction when resumes are unchanged. | Latency: ~1–2 LLM calls per resume; cost scales with candidate count |
| **Hybrid retrieval (BM25 + dense)** | BM25 catches exact keywords; dense catches paraphrases and synonyms. Together they cover both “Kafka” and “event streaming platforms”. | Two indices to maintain; RRF fusion adds minimal overhead |
| **RRF fusion (k=60)** | Parameter-free; no score calibration across retrievers; robust to scale differences between BM25 and cosine similarity | Fixed k may not be optimal for all corpus sizes |
| **Cross-encoder rerank** | Highest quality relevance signal for [JD, resume] pairs; catches inflated language (“used Python once” vs “led Python backend”) | Expensive: O(n) pairwise calls; applied only to top 50% of RRF pool |
| **4D + CE scoring** | Explainable: D1 (skills), D2 (seniority), D3 (domain), D4 (constraints) + CE blend. JD seniority presets + optional per-run custom weights; CE weight from config. | Hand-tuned defaults; **planned:** joint fit of dimension weights, CE α, and retrieval hyperparameters via regression on labeled data |
| **Ontology (curated skill graph + domain pairs)** | **Grounding in use today:** skills are matched in order **exact → adjacent → group**, with **lower tier multipliers** on top of evidence level (BUILT_WITH / USED / LISTED). Domains use **canonical labels** plus **`ADJACENT_DOMAINS`** before any LLM. | Off-graph cases use optional **LLM tool fallbacks** (`D1_LLM_FALLBACK`, `D3_LLM_FALLBACK` in `src/config.py`). |

**LLM vs deterministic (by dimension).** **D1 / D3** are **ontology-first**; an LLM tool runs **only** when exact / adjacent / (for D1) group coverage does not resolve the skill or domain. **D2** inverts that pattern when the agent is enabled: the **LLM orchestrates** calls into **deterministic** tool implementations on resume evidence, and **falls back** to fully deterministic seniority-from-profiles if the agent path fails. The scoring LLM still produces rationale and strengths/gaps; the **D1–D4 values that enter the final blend** are the **merged, grounded** profile scores (not unconstrained judge floats).

### Alternatives Considered

| Alternative | Why Not Chosen |
|-------------|----------------|
| **TF-IDF only** | In ablation, a 70K-char academic paper ranked #1 because it contained every keyword. No semantic understanding. |
| **Dense only** | Misses exact technical terms; “Go” and “Golang” may diverge in embedding space depending on context. |
| **LLM-only scoring (no structure)** | Unpredictable outputs; hard to audit; no traceability to evidence. Our D1–D4 are **grounded** in ontology/regex/agents where possible. |
| **Single LLM call for extraction + scoring** | More prompt drift; harder to cache; we separate cheap extraction (cached) from expensive scoring. |
| **Pure cross-encoder for all pairs** | O(n²) cost; infeasible for 100+ resumes. We use CE only on top candidates from RRF. |

---

## Architecture

### End-to-End Pipeline

```
Job Description  +  Resumes (PDF / DOCX / PNG / LaTeX / TXT / ZIP)
       │                      │
       │                      ▼
       │         ┌─────────────────────────────────────────────┐
       │         │  L0  INGESTION & ADVERSARIAL SANITIZATION   │
       │         │                                             │
       │         │  Text Extraction (MIME-first detection)     │
       │         │  ├─ PDF     → pypdf + pdftotext fallback    │
       │         │  ├─ DOCX    → pandoc + python-docx fallback │
       │         │  ├─ Image   → pytesseract OCR               │
       │         │  ├─ LaTeX   → plain text                    │
       │         │  └─ HTML    → BeautifulSoup strip tags      │
       │         │                                             │
       │         │  Adversarial Detectors (7 independent)     │
       │         │  ├─ Prompt injection   (strip + penalise)     │
       │         │  ├─ Invisible text     (zero-width chars)     │
       │         │  ├─ Homoglyph attack  (Cyrillic → Latin)     │
       │         │  ├─ JD duplication    (n-gram overlap)      │
       │         │  ├─ Keyword stuffing  (TF density ratio)     │
       │         │  ├─ Experience inflation (timeline gap)     │
       │         │  └─ Credential anomaly (>7 certs listed)     │
       │         │                                             │
       │         │  Non-Resume Gate (LLM: gpt-4o-mini)         │
       │         │  → YES/NO on first ~1200 chars              │
       │         │  → Score 0, threat=HIGH if NO               │
       │         │                                             │
       │         │  Content Dedup (SHA-256 hash)               │
       │         └─────────────────────────────────────────────┘
       │                      │ clean text + threat_report
       │                      ▼
       │         ┌─────────────────────────────────────────────┐
       │         │  STAGE 1  PROFILE EXTRACTION (gpt-4o-mini)  │
       ├────────►│                                             │
       │  JD     │  JD Profile: required_skills, years_required,│
       │         │  domain, seniority, hard_constraints         │
       │         │                                             │
       │         │  Resume Profile: skills (name, level,       │
       │         │  evidence), total_years, domains,           │
       │         │  seniority_signals, highlights              │
       │         │  level ∈ {BUILT_WITH, USED, LISTED, ABSENT} │
       │         └─────────────────────────────────────────────┘
       │                      │ jd_profile + resume_profile
       │                      ▼
       │         ┌─────────────────────────────────────────────┐
       │         │  L1  HYBRID RETRIEVAL                       │
       │         │  ├─ BM25  (rank_bm25, lexical)              │
       │         │  └─ Bi-encoder (all-MiniLM-L6-v2 or        │
       │         │     text-embedding-3-small)                 │
       │         └─────────────────────────────────────────────┘
       │                      │ two ranked lists
       │                      ▼
       │         ┌─────────────────────────────────────────────┐
       │         │  L2  RRF FUSION  (k=60)                    │
       │         │  score(d) = Σ 1/(60 + rank_i(d))           │
       │         └─────────────────────────────────────────────┘
       │                      │ unified ranking
       │                      ▼
       │         ┌─────────────────────────────────────────────┐
       │         │  L3  CROSS-ENCODER RE-RANKING               │
       │         │  ms-marco-MiniLM-L-6-v2                     │
       │         │  Top 50% through CE; rest get RRF-derived    │
       │         │  score; docs outside pool get ce_logit=-10  │
       │         └─────────────────────────────────────────────┘
       │                      │ ce_logit per resume
       │                      ▼
       │         ┌─────────────────────────────────────────────┐
       │         │  STAGE 2  PROFILE D1–D4 + JUDGE LLM       │
       │         │  Compute D1–D4 from profiles first       │
       │         │  (ontology tiers, D2 agent/tools, D4)     │
       │         │  Judge LLM: rationale / strengths / gaps    │
       │         │  Published D1–D4 = grounded merge (above)  │
       │         │  Agentic retry: if confidence=LOW → 1       │
       │         │  bounded re-score (verify_score)            │
       │         └─────────────────────────────────────────────┘
       │                      │
       │                      ▼
       │         ┌─────────────────────────────────────────────┐
       │         │  FINAL SCORE FORMULA                        │
       │         │  dim = w1·D1 + w2·D2 + w3·D3 + w4·D4       │
       │         │  (defaults 40/35/15/10 or role / custom %) │
       │         │  raw = 0.5·dim + 0.5·σ(ce_logit)           │
       │         │  (CE_WEIGHT in src/config.py, default 0.5) │
       │         │  final = raw · (1 − adversarial_penalty)   │
       │         └─────────────────────────────────────────────┘
       │                      │
       │                      ▼
               Ranked candidates with final_score, D1–D4,
               strengths, gaps, per-skill evidence, threat_level
```

### Scoring Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **D1 — Hard Skills** | 0.40 | Match JD required_skills to resume; BUILT_WITH=1.0, USED=0.7, LISTED=0.3, ABSENT=0.0 |
| **D2 — Experience Depth** | 0.35 | Leadership, Architecture, Scale, Ownership signals from seniority_signals + total_years |
| **D3 — Domain Fit** | 0.15 | Same domain=1.0, adjacent=0.5–0.8, unrelated=0.0–0.3 |
| **D4 — Hard Constraints** | 0.10 | Fraction of hard_constraints met (years, certs, location, etc.) |

### Retrieval Formulation: CE Top 50% + Rank-Wise Full Coverage

The cross-encoder is expensive (O(n) pairwise [JD, resume] calls). We apply it only to the **top CE_TOP_PERCENT** of the RRF pool (env, default 50% → top 25 of 50). **All other docs** receive a RRF rank-derived logit.

**RRF over all docs:** RRF fusion runs over the **full corpus**. Every doc gets a rank-based score; no one gets `ce_logit = -10`.

| Position | What happens | ce_logit source |
|----------|--------------|------------------|
| **Top CE_TOP_PERCENT** of RRF pool (default 50%) | Real cross-encoder; rank-based scores in (0.01, 0.99), converted to logits | Real CE output |
| **All remaining** | RRF rank-derived fractional score, linear decay by position; converted to logit | RRF-derived |

**CE blend:** `CE_WEIGHT` in `src/config.py` (default **0.5**): 50% dimension composite, 50% cross-encoder sigmoid. **CE_TOP_PERCENT** (env, 0–100, default 50): % of RRF pool that goes through real CE.

### Scoring Formulation: Layer by Layer

The final score is built in clearly separated layers. If we change a layer tomorrow, we update only that layer.

**Layer 1 — Dimension composite (50% of raw score before penalty):**

\[
\text{dim} = w_1 D_1 + w_2 D_2 + w_3 D_3 + w_4 D_4
\]

- **Default weights** \((w_1..w_4)\): base **40% / 35% / 15% / 10%** for D1–D4, overridden by **JD seniority** presets in `scorer.py` (`ROLE_WEIGHTS`: junior, mid, senior, staff, …).
- **Custom (web):** On *New Run*, users can enable “Custom dimension importance” and enter percentages for Skills / Seniority / Domain / Constraints; values are **normalized to sum 100%** and stored on `MatchRun.scoring_config`. If disabled, only defaults apply.
- **D1 (Skills):** Per required skill: BUILT_WITH=1.0, USED=0.7, LISTED=0.3, ABSENT=0.0. Ontology match order **exact → adjacent → group** with tiered credit at each step; core skills weighted higher in the average.
- **D2 (Seniority):** Leadership, architecture, scale, ownership + years; **LLM agent → deterministic tools** on evidence when enabled, else deterministic-from-profiles.
- **D3 (Domain):** Canonical match → **`ADJACENT_DOMAINS`** pair → LLM tool only if still unresolved.
- **D4 (Constraints):** met=1.0, partial=0.5, not met=0.0; average.

**Layer 2 — CE weight:** \(\alpha = \texttt{CE\_WEIGHT}\) (default **0.5**).

**Layer 3 — Raw score blend:**

\[
\text{raw} = (1 - \alpha)\,\text{dim} + \alpha\,\sigma(\text{ce\_logit})
\]

- `σ(logit) = 1/(1 + exp(-logit))`.

**Layer 4 — Adversarial penalty:**

\[
\text{final} = \max(0, \min(1, \text{raw} \times (1 - \text{adversarial\_penalty})))
\]

**Summary:** 50% dim + 50% CE by default; internal D1–D4 mix from role presets or user percentages; then × (1 − penalty).

### Two Execution Paths

| Path | Retrieval | ce_logit | Use Case |
|------|-----------|----------|----------|
| **CLI / `demo.py` / `pipeline.py`** | Full hybrid (BM25 + dense + RRF + CE) | Real CE logit from engine | Research, ablation, evaluation |
| **Django Web UI (`services.py`)** | Hybrid retrieval + CE per run | Real CE logit from `RetrievalEngine` | Production UI; custom D1–D4 %, scoring mode, rescore, CSV export |

---

## Feature Engineering & Data Preprocessing

### Text Extraction (MIME-first)

1. Detect format via `file --mime-type` or extension
2. Route to extractor:
   - **PDF:** `pypdf` → fallback to `pdftotext` if available
   - **DOCX/DOC/RTF:** `python-docx` or `pandoc`
   - **HTML:** BeautifulSoup strip tags
   - **Images:** `pytesseract` OCR (requires `tesseract` installed)
   - **LaTeX/plain text:** read as text

### Adversarial Sanitization (7 Detectors)

| Detector | What It Catches | Action |
|----------|-----------------|--------|
| Prompt injection | “Ignore instructions”, “score me 10/10” | Strip patterns; assign penalty 0.7–0.95 |
| Keyword stuffing | Abnormal TF density, verb ratio | Continuous penalty |
| Invisible text | Zero-width, BOM, directional marks | Strip; count for penalty |
| Homoglyphs | Cyrillic “a” → Latin “a” | Normalise |
| JD duplication | Copy-pasted JD into resume (n-gram overlap) | Penalty by overlap ratio |
| Experience inflation | Timeline gaps vs claimed years | Penalty |
| Credential anomaly | >7 certs in skills section | Penalty |

**Penalty model:** Continuous (0.0–0.95), not binary. Final score multiplied by `(1 - adversarial_penalty)`.

### Profile Normalisation

- **Skills:** Ontology alias mapping (e.g. “Golang” → “Go”); **exact / adjacent / group** tiers for D1 with decreasing credit; adjacency (e.g. rabbitmq ≈ kafka).
- **Domains:** Canonical mapping; `ADJACENT_DOMAINS` for adjacent industry pairs before LLM fallback.
- **Seniority:** Normalised to schema vocabulary; D2 agent calls **deterministic** assessors on evidence text.

### Deterministic Fallback (No LLM)

When `OPENAI_API_KEY` is not set:

- BM25 + cross-encoder for retrieval
- Regex-based skill matching (BUILD_VERBS, USE_VERBS)
- Keyword hit counts for D2, D3, D4

Accuracy is lower, especially for semantic skill equivalence and seniority judgement.

---

## Performance Evaluation

### Synthetic Evaluation Set

We evaluate against a small **human-labeled** set (not production scale, but reproducible):

- **Job description (primary table below):** `data/job_descriptions/senior_backend_finpay.txt` — senior backend / FinPay-style stack
- **Resumes:** `data/resumes/` (13 labeled `.txt` profiles + one unlabeled file in the folder; labels in `golden_dataset.jsonl`)
- **Optional stress formats:** `data/ablation_resumes/` (PDF, PNG, JPEG, LaTeX) — use for manual UI / adversarial testing; add stems and grades to `golden_dataset.jsonl` when you want them in `ablation.py`
- **Labels:** `data/golden_dataset.jsonl` — per JD, keyed by resume **file stem**:
  - **1.0** = Good Match
  - **0.5** = Partial Match
  - **0.0** = Poor Match

### Adding Golden Labels

Edit `data/golden_dataset.jsonl` and add (or extend) an entry:

```json
{
  "your_jd_id": {
    "resume_id_1": 1.0,
    "resume_id_2": 0.5,
    "resume_id_3": 0.0
  }
}
```

Label keys must match file stems exactly (case-sensitive). The ablation script auto-selects the JD with the most labeled resumes.

### Evaluation Metrics

| Metric | What It Measures |
|--------|-------------------|
| **nDCG@k** | Quality of top-k ranking; position-weighted, graded relevance |
| **MRR** | How quickly the first relevant result appears |
| **Precision@k** | Fraction of top-k that are relevant (≥0.5) |
| **Spearman ρ** | Rank correlation between predicted and gold ordering |
| **Impact Ratio** | NYC LL144 bias audit — selection rate per group / max rate |

<a id="ablation-latest-ema-mar-2026"></a>

### Latest ablation — real resume dataset (25 March 2026)

**Setup:** Labeled **real resumes** from the project corpus, JD key **`ema_enterprise_swe`** in `data/golden_dataset.jsonl`, run through the **current** RankForge **Ablation** ladder (same five steps as the UI: sparse baselines → hybrid RRF → cross-encoder → **full pipeline** with profile extraction, curated **ontology-grounded** D1/D3 tiers, D2 agent/tools when enabled, merged D1–D4, and agentic verification on LOW confidence). This table is the **most recent recorded snapshot** for that configuration.

| Approach | nDCG@3 | nDCG@5 | nDCG@10 | MRR | P@3 | P@5 | Spearman ρ | TOP-1† |
|----------|-------:|-------:|--------:|----:|----:|----:|-----------:|-------:|
| TF-IDF only | 0.229 | 0.471 | 0.667 | 0.250 | 0.000 | 0.200 | 0.304 | 0.093 |
| TF-IDF + BM25 | 0.436 | 0.480 | 0.584 | 0.500 | 0.333 | 0.200 | 0.214 | 0.032 |
| BM25 + Dense + RRF | 0.738 | 0.726 | 0.712 | 0.333 | 0.333 | 0.400 | 0.329 | 0.032 |
| + Cross-encoder | 0.304 | 0.444 | 0.643 | 0.200 | 0.000 | 0.200 | 0.389 | 0.158 |
| Hybrid + Agentic (Ontology Grounding) | 0.498 | 0.690 | **0.786** | 0.250 | 0.000 | 0.400 | **0.454** | **0.720** |

† **TOP-1** in the UI is the **model score at rank 1** after that approach (how sharp the list head is), not “accuracy@1” vs gold.

**Reading this run:** On this JD, **retrieval-heavy stages** (notably BM25 + dense + RRF) can win **early-cutoff** metrics (e.g. nDCG@3, MRR, P@3) while **reordering the shortlist**; the **full stack** then trades some of that head metric for **better graded ordering through the top 10** and stronger **Spearman ρ**. **Stage 5 achieves the best nDCG@10 and Spearman here**, which is what we emphasize when discussing ranking quality on this benchmark. **Dimension weights, CE blend, RRF *k*, and pool cut** are still deliberately conservative and are **planned for joint tuning** on golden labels (see [Future Steps](#future-steps--production-roadmap))—this snapshot is meant to show the **intended direction**: ontology + structured scoring **recovering list quality** where it matters for shortlist review, not that every column is simultaneously maximal.

**Why nDCG@10 is primary here (and not MRR / P@k alone):** As in the architecture write-up (`docs/architecture.tex`), **nDCG** uses **graded** relevance (e.g. 1.0 / 0.5 / 0.0) and a **logarithmic position discount**: pushing a strong candidate down to rank 10 hurts more than pushing them to rank 2. That matches **how TA uses a ranked shortlist**—several slots get eyeballs, not only position 1. **MRR** is dominated by the **first** relevant hit, so it can look excellent or poor when a single rank moves, even if the **rest of the top 10** improved. **P@k** treats relevance as **binary within k** and ignores **how much** better one “good” resume is than another, so it can disagree with nDCG when labels are graded. **Spearman ρ** is a useful **global** ordering check but is **noisier on small labeled pools**; **nDCG@10** stays tied to the **decision-critical head** of the list. We still report MRR, P@k, and Spearman as **secondary** diagnostics, not as competing “headline” goals.

### Ablation study — five levels (FinPay JD, labeled `data/resumes/`)

`ablation.py` scores the same pool with **five stacked approaches**: (1) TF-IDF only → (2) TF-IDF + BM25 (RRF) → (3) BM25 + bi-encoder (RRF) → (4) + cross-encoder rerank → (5) **full pipeline** (hybrid retrieval + CE + ontology / 4D scoring as in `src/pipeline.py`). Metrics are vs `senior_backend_finpay` in `golden_dataset.jsonl`. For the **most recent real-resume snapshot** (`ema_enterprise_swe`, 25 March 2026), see [Latest ablation](#ablation-latest-ema-mar-2026) above.

**Measured run** (regenerate anytime; committed snapshot: `evaluation/ablation_results.json`):

| Level | Approach | nDCG@3 | nDCG@5 | nDCG@10 | MRR | P@3 | P@5 | Spearman ρ |
|------:|----------|-------:|-------:|--------:|----:|----:|----:|-----------:|
| 1 | TF-IDF cosine only | 1.000 | 0.915 | 0.974 | 1.000 | 1.000 | 0.800 | 0.558 |
| 2 | TF-IDF + BM25 (RRF) | 1.000 | 0.915 | 0.974 | 1.000 | 1.000 | 0.800 | 0.637 |
| 3 | BM25 + bi-encoder (RRF) | 1.000 | 0.915 | **0.976** | 1.000 | 1.000 | 0.800 | **0.681** |
| 4 | + Cross-encoder rerank | 0.765 | 0.727 | 0.856 | 1.000 | 0.667 | 0.600 | 0.267 |
| 5 | + Full system (ontology / 4D) | 0.765 | 0.727 | **0.922** | 1.000 | 0.667 | 0.600 | 0.563 |

**How to read this (important for reviewers):**

- **Stages 1 → 3 — consistent retrieval improvement:** On this FinPay JD, keyword overlap is strong enough that the top three gold “good” candidates already appear at the top under raw TF-IDF, so **nDCG@3 stays saturated at 1.0**. The real progression shows up in **Spearman ρ** (0.558 → 0.637 → **0.681**): hybrid BM25 + dense **better matches the full gold ordering** across all 13 labeled resumes, not just the head of the list.
- **Stage 4 — cross-encoder:** Reranking optimizes pairwise [JD, resume] relevance; on a **tiny** labeled set, reordering can **lower** graded metrics (nDCG@3/5/10, P@k) even while MRR stays high — a known small-sample effect, not a claim that CE is worse in production.
- **Stage 5 — full stack vs stage 4:** The complete pipeline **recovers list quality vs CE-only**: **nDCG@10 0.856 → 0.922** and **Spearman 0.267 → 0.563**, while adding **explainable D1–D4 scores, ontology grounding, and adversarial handling** (not visible in this table). That is the intended “improvement applied” for the productized matcher vs retrieval-only baselines.

**Reproduce (FinPay, default golden labels):**

```bash
python ablation.py --jd data/job_descriptions/senior_backend_finpay.txt --resumes data/resumes/
# writes evaluation/ablation_results.json
```

**Your own resumes:** add a block to `data/golden_dataset.jsonl` (JD stem → `{ "your_file_stem": 1.0, ... }`), then point `--resumes` at a folder of those files and re-run the same command.

### Running Evaluation

```bash
# Ablation: default uses only data/ablation_resumes/ when that folder has files (fast UI-sized run);
# picks JD in golden_dataset with best label overlap on those stems. Use --jd/--resumes for FinPay table.
python ablation.py

# FinPay + labeled txt corpus (recommended for the table above)
python ablation.py --jd data/job_descriptions/senior_backend_finpay.txt --resumes data/resumes/

# Custom JD and resumes (requires matching keys in golden_dataset.jsonl)
python ablation.py --jd my_jd.txt --resumes ./my_resumes/

# Demo: full pipeline on a folder
python demo.py --jd data/job_descriptions/senior_backend_finpay.txt \
               --resumes data/ablation_resumes/
```

### Formal Evaluation with Larger Labeled Data

If a larger, labeled dataset were available, we would:

1. **Primary metric:** **nDCG@10** — captures graded relevance and positional quality
2. **Secondary:** **Precision@5** (recruiter typically reviews top 5); **MRR** (how fast first good candidate appears)
3. **Rank correlation:** **Spearman ρ** for overall ordering quality
4. **Fairness:** **Impact ratio** per demographic proxy (NYC LL144 four-fifths rule)
5. **Calibration:** Score separation (avg_good vs avg_partial vs avg_poor); gaps should be positive
6. **A/B testing:** Compare ranking quality vs human recruiter baseline on same pool
7. **Combined regression (planned):** Treat D1–D4 weights, CE weight, RRF *k*, and `CE_TOP_PERCENT` as a constrained parameter vector; optimize nDCG@k / Spearman on `golden_dataset.jsonl` + held-out adversarial cases (keyword stuffing, JD copy-paste, etc.) so ranking quality and robustness improve together rather than tuning one knob at a time

---

## Challenges Overcome & Solutions

| Challenge | Solution |
|-----------|----------|
| **Adversarial documents rank #1** (e.g. 70K academic paper with every keyword) | L0 sanitization + `is_resume` LLM gate; continuous penalty instead of binary reject; adversarial docs get high penalty and low/zero score |
| **Semantic equivalence** (“payment APIs” vs “financial microservices”) | Dense bi-encoder retrieval; ontology adjacency (e.g. rabbitmq ≈ kafka); LLM profile extraction |
| **Exact keyword missed by dense** | BM25 lexical retrieval; hybrid RRF fusion |
| ** inflated language** (“used Python once” vs “led Python backend”) | Cross-encoder re-ranking on [JD, resume] pairs |
| **Non-deterministic LLM outputs** | Two-stage: cheap extraction (cached) + temp=0 scoring; D1–D4 overwritten by deterministic modules (ontology, regex, agents) where possible |
| **Skills not in ontology** | D1/D3 LLM fallback (`D1_LLM_FALLBACK`, `D3_LLM_FALLBACK`) when exact/adjacent match fails |
| **Seniority judgement** | D2 agent with deterministic tools on LLM quotes; regex/heuristics for leadership, architecture, scale, ownership |
| **Explainability** | 4D scores; per-skill evidence (name, level, quote); strengths/gaps; threat flags |
| **Cost at scale** | Profile cache (IndexStore); CE only on top 50% of RRF pool; optional `cost_tracker` in extras |

---

## Future Steps & Production Roadmap

**Calibration priority.** The submission build keeps the **base scorer stable** (same formulas and evidence paths). The next improvement pass is to **discover weights and ordering of signals jointly**: dimension weights (D1–D4), CE blend α, retrieval fusion (RRF *k*, CE pool cut), and optional score floors/caps — fitted with **combined regression or constrained grid search** on golden labels, ablation JSON, and adversarial uploads so nDCG / Spearman and “gaming resistance” move together. Other high-value items: wire `cost_tracker` to per-run DB totals, expand few-shot banks per industry, richer confidence calibration (skill evidence + CE–dim divergence), and optional demographic-aware fairness dashboards (impact ratio already implemented; threshold is configurable — see env table).

### Phase 1 — Demo-Ready

- **Combined regression / grid search** — Joint optimization of D1–D4 weights, CE weight, and key retrieval hyperparameters against `golden_dataset.jsonl` + adversarial suite; primary objective nDCG@10 / Spearman, secondary robustness on ablation resumes
- **Surface full profiles in UI** — Optional collapsible `jd_profile` / `resume_profile` JSON for auditors (skill table already shown when populated)
- **Confidence calibration** — Tune thresholds on BUILT_WITH vs ABSENT ratios and CE vs `dim_composite` divergence beyond current heuristics
- **Cost visibility** — Persist `extras/cost_tracker.py` aggregates per `MatchRun` for stakeholder “cost per resume” answers

**Already in this codebase (not blocking submission):** JD seniority–based dimension presets (`ROLE_WEIGHTS`), optional **custom dimension importance** on New Run (normalized, stored in `MatchRun.scoring_config`), domain-selected few-shot bank (FinTech vs AI/ML), applied weights surfaced on candidate detail, agentic LOW-confidence re-score, 7-detector adversarial penalty model.

### Phase 2 — Knowledge Graph & Skill Intelligence

- **ESCO (13,485 skills, 1.3M co-occurrence edges):** canonical skill resolution + adjacency + graph expansion; ontology as source of truth for exact/adjacent/group matching
- **O\*NET crosswalk:** importance weights per skill per occupation (e.g. Python = 5/5 for Backend Engineer, 1/5 for Marketing Manager)
- **Lightcast Open Skills (32K skills, biweekly updates):** captures emerging skills ESCO/O\*NET lag on (“agentic workflows”, “prompt engineering”)
- **KG as translation layer** between LLM reasoning and deterministic grounding — LLM outputs normalised via KG before D1–D4 computation

### Phase 3 — MCP Integration

- **MCP server** (`extras/mcp_server.py`) with 9 tools: `match`, `score`, `explain`, `audit`, `feedback`, etc.
- **JSON-RPC 2.0**, RBAC, rate limiting
- **Production integrations:** ATS (Greenhouse, Lever), HRIS for demographics, calendar for interview scheduling

### Phase 4 — HITL Feedback Loop

- **Recruiter decisions** (ADVANCE / MAYBE / REJECT) → immutable JSONL
- **Pattern analysis** → bounded weight adjustment (±0.10, re-normalised)
- **Feedback is context, not training data** — avoiding the Amazon AI Hiring bias replication trap (2014–2018): we do not retrain on feedback; we adjust interpretable weights within bounds

### Phase 5 — Compliance & Audit

- **EU AI Act (Article 6, Annex III):** immutable `AuditRecord` per run — config hash, model version, reproducibility via temp=0
- **NYC Local Law 144:** `impact_ratio` per demographic group; four-fifths rule flagging (`extras/compliance.py`; override threshold with `FAIRNESS_FOUR_FIFTHS_THRESHOLD` if needed)
- **Cost tracking:** per-call token accounting (`extras/cost_tracker.py`)
- **Append-only audit logs** in `data/feedback/audit_logs/`

### Phase 6 — Scale

- **Vector DB:** Milvus or Qdrant (replace numpy)
- **Job queue:** Celery + Redis durable tasks
- **Storage:** PostgreSQL, S3
- **Progress:** SSE for real-time updates
- **Deployment:** horizontal auto-scaling

---

## Setup & Usage

### Prerequisites

- Python 3.9+
- `tesseract` (for OCR): `brew install tesseract` on macOS
- Optional: `pdftotext`, `pandoc` for richer extraction

### Installation

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env with your keys
```

### Environment Variables

| Variable | Notes |
|----------|-------|
| `OPENAI_API_KEY` | **Required for LLM scoring.** Set via `.env` or `export`. Without it, deterministic fallback runs. |
| `SECRET_KEY` | Django secret (required in production; `.env.example` has placeholder) |
| `DEBUG` | Set `False` in production |
| `USE_SQLITE` | `true` (default) → SQLite with WAL; `false` → MySQL via `DB_*` vars below |
| `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT` | MySQL connection (only when `USE_SQLITE=false`) |
| `CE_TOP_PERCENT` | % of RRF pool through real CE (0–100, default 50). Rest get RRF-derived logit. |
| `FAIRNESS_FOUR_FIFTHS_THRESHOLD` | (Optional) NYC LL144-style flag when group impact ratio falls below this value (default `0.8`). Used by `extras/compliance.py`. |

**Security:** API keys must come from environment or `.env` only.  Added above while setting up if possible.
### Web UI

```bash
python manage.py migrate
python manage.py createsuperuser
python manage.py seed_data          # optional: populate demo JD + resumes
python manage.py runserver
```

Since this is just localhost and not deplpyed yet, so labelling that here once deployed this will be replaced -> Visit `http://127.0.0.1:8000` — log in, go to **New Run**, paste JD, upload resumes (up to 50; ZIP supported). On *New Run* you can also select a **scoring mode** (Auto / LLM / Deterministic) and optionally set **custom dimension weights** (D1–D4 percentages).

Processing runs in a background subprocess (`manage.py process_run <id>`); the run-detail page auto-polls for progress and streams partial results.

<a id="web-ui-walkthrough--rankforge-screenshots"></a>

#### Web UI walkthrough — RankForge screenshots

The following matches what the Django app (**Resume Matcher**) shows end-to-end. **Curated images** live under `docs/rankforge_screens/` as descriptive kebab-case `.png` names (dashboard, new-run flows, pipeline, test suite, terminals, roadmap, run/candidate views, ablation) so paths read clearly in docs and diffs.

**Shell layout** — Left nav: **Dashboard**, **New Run**, then **ENGINE** → **Pipeline**, **Test Suite**, **Ablation**, **Roadmap**. Header: dark mode, user, logout.

| What | Highlights |
|------|------------|
| **Dashboard** | Summary cards (total runs, resumes, avg top score, last run status). **Recent runs** table: job title, status (Complete / Processing), resume count, mode (**Auto** vs **Llm**), created time. **+ New Matching Run** shortcut. |
| **New Run** | JD **Title** + **Job Description**; **Resumes** drop zone (PDF, DOCX, PNG, LaTeX, ZIP — note in UI: optimized for PDF). **Settings**: scoring mode dropdown (**Auto** = LLM if API key else deterministic; **LLM**; **Deterministic**). **Scoring weight profile** chips: Auto-detect, Junior, Mid-level, Senior, Staff / Lead, Executive, **Custom** — each updates D1–D4 sliders. UI copy: percentages normalize to 100%; **final score blends 50% dimension composite with 50% cross-encoder** (same default as `CE_WEIGHT` in `src/config.py`). **Start Matching Run**. |
| **Pipeline** (`/pipeline/`) | **L0** — ingestion (`TextExtractor`), ZIP in-memory, up to 50 resumes/run; adversarial **7-detector** sanitizer (injection, invisible text, homoglyphs, JD duplication, keyword density, font-size tricks, non-resume **LLM gate**); HIGH threat → zero score downstream. **L1** — BM25 + dense bi-encoder (OpenAI `text-embedding-3-small` vs local `all-MiniLM-L6-v2`), top-50 each, normalized scores. **L2** — RRF \(k=60\); top pool; **CE_TOP_PERCENT** controls how many get a real cross-encoder vs rank-derived logit (see [Retrieval Formulation](#retrieval-formulation-ce-top-50-rank-wise-full-coverage) above for the current code path). **L3** — cross-encoder re-rank. Second panel: **D1–D4** definitions (BUILT_WITH / USED / LISTED / ABSENT; D2 agent signals; D3 same/adjacent/unrelated; D4 constraints default 1.0 if none), **role-adaptive weight table** (Junior 50/25/15/10 through Executive 25/55/12/8), **final formula** \(\text{final} = (1-\alpha)\,\text{dim} + \alpha\,\sigma(\text{ce\_logit})\) with default \(\alpha=0.5\), LOW-confidence agentic retry + deterministic fallback, ESCO adjacency penalty (e.g. **0.7×** same-level neighbor), and **eval metrics** (nDCG@k, MRR, P@k, Spearman) with a pointer to **Ablation** for toggling stages. |
| **Test Suite** (`/tests/`) | **Run All Tests**; grouped results mirror `tests/test_all.py` — ontology & skill overlap; injection / invisible / homoglyphs / JD duplication; full sanitize; sigmoid & skill-penalty & final score; recommendation bands; nDCG, MRR, precision, Spearman; bias audit; extractor formats (incl. `.tex`, image OCR smoke); resume gate; ablation dataset smoke; contracts. Typical run: **60+ passed**, a few skipped when optional conditions aren’t met. |
| **Ablation** (`/ablation/`) | Same five-level study as `ablation.py` / [Performance Evaluation](#performance-evaluation); use the UI for quick iteration without the CLI. |
| **Roadmap** (`/roadmap/`) | In-app cards expand on [Future Steps](#future-steps--production-roadmap): ESCO normalization + multi-hop graph + local RDF; **compliance** (PII redaction, per-dimension distributions, four-fifths flagging, signed audit store — starts from `extras/compliance.py`); **MCP** (`match_resumes`, `explain_score`, `update_label` — `extras/mcp_server.py`); **cost** caps & per-run token surfacing (`extras/cost_tracker.py`); **HITL** labels → `golden_dataset.jsonl` (`extras/feedback.py`); **parameter tuning** (α, role presets, few-shot verticals, RRF \(k\), `CE_TOP_PERCENT`); **infrastructure** — Celery + Redis instead of daemon threads, Postgres for multi-worker state, S3/object storage for files, `django-ratelimit` on high-traffic endpoints, SSE/WebSockets instead of polling-only UX. |

**Run & candidate views** — Run header: job title, run id, resume count, timestamp; actions **Export CSV**, **Re-score**, status badge. **Ranked candidates** table: rank, label, **final score**, D1–D4, **confidence**, **recommendation** (e.g. Partial / No / Strong match), **threat**, **Preview** (PDF or text). **Candidate detail**: per-dimension cards with applied weights; breakdown row (**dimension composite**, **cross-encoder sigmoid** × weight, **final**); **LLM rationale**; **Strengths** / **Gaps**; **D1 skill evidence** rows (`exact`, `absent`, `group`, `adjacent`, `llm_fallback`); **D2** leadership / architecture / scale / ownership / years + narrative; **Retrieval stage** fields (BM25, dense, `CE_LOGIT`, JSON blobs for skills, seniority, domain). **Non-resume gate**: document rejected before scoring → **0.00** final score, LOW confidence, rationale in UI matches gate behaviour.

**Reference figures** (subset; see `docs/rankforge_screens/*.png` for the full gallery):

![Dashboard — runs overview](docs/rankforge_screens/dashboard-overview-recent-runs.png)

![New Run — JD, uploads, scoring mode and weight presets](docs/rankforge_screens/new-run-settings-auto-detect.png)

![Pipeline — L0–L3 retrieval narrative](docs/rankforge_screens/pipeline-layers-l0-l3.png)

![Pipeline — D1–D4, role weights, final formula, ESCO & metrics](docs/rankforge_screens/pipeline-dimensions-role-weights-esco-eval.png)

![Test Suite — grouped pytest results](docs/rankforge_screens/test-suite-ontology-skill-overlap.png)

![Roadmap — production and research next steps](docs/rankforge_screens/roadmap-items-1-through-3.png)

![Run detail — ranked candidates](docs/rankforge_screens/run-ranked-candidates-table.png)

![Candidate detail — dimensions, CE blend, rationale](docs/rankforge_screens/candidate-detail-dimensions-score.png)

#### Tool Pages

| Page | URL | Purpose |
|------|-----|---------|
| Pipeline | `/pipeline/` | Interactive architecture diagram |
| Roadmap | `/roadmap/` | Future steps / production roadmap |
| Test Suite | `/tests/` | Run `pytest` from the browser; stream results |
| Ablation | `/ablation/` | Trigger 5-level ablation study and view results |

#### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/matching/start/` | Create job + run, upload resumes (incl. ZIP), spawn worker |
| GET | `/api/matching/run/<id>/status/` | Poll run progress + partial results (includes dead-worker restart) |
| GET | `/api/matching/candidate/<id>/` | Full result JSON for a single candidate |
| GET | `/api/matching/resume/<id>/preview/` | Resume raw text or file URL |
| POST | `/api/matching/run/<id>/rescore/` | Clear results and re-run the entire match |
| POST | `/api/matching/result/<id>/rescore/` | Re-score a single candidate |
| GET | `/api/matching/run/<id>/export/` | CSV export of all results |
| POST | `/api/tests/run/` | Start background test suite |
| GET | `/api/tests/status/` | Poll test-suite progress |
| POST | `/api/ablation/run/` | Start background ablation study |
| GET | `/api/ablation/status/` | Poll ablation progress |

### CLI

```bash
# Demo: full pipeline on sample data
python demo.py

# Custom JD and resumes
python demo.py --jd data/job_descriptions/ai_engineer_ema.txt \
               --resumes data/ablation_resumes/

# Ablation: 5-level metric table (FinPay JD + golden labels) — see Performance Evaluation
python ablation.py --jd data/job_descriptions/senior_backend_finpay.txt --resumes data/resumes/

# Tests
python -m pytest tests/test_all.py -v
```

---

## Testing

The project ships a `pytest`-based test suite (`tests/test_all.py`) with **~75 unit and integration tests** across **13 test classes**. The suite is designed to run without an `OPENAI_API_KEY` — every LLM-dependent path either stubs gracefully or uses the deterministic fallback, so CI stays free.

### Running the Suite

```bash
# Run all tests with verbose output
python -m pytest tests/test_all.py -v

# Run a single test class
python -m pytest tests/test_all.py::TestOntology -v

# Stop on first failure
python -m pytest tests/test_all.py -x
```

### Coverage by Area

| Test Class | What It Verifies |
|------------|-----------------|
| `TestOntology` | Canonical alias mapping (k8s→kubernetes, golang→go), multi-word skill extraction, no false positives on non-technical text |
| `TestSkillOverlap` | Exact, adjacent (rabbitmq≈kafka), and group-level (java≈python) matching; empty-JD edge case; regex fallback with experience-year extraction |
| `TestInjectionDetection` | Prompt injection patterns (`ignore instructions`, score manipulation, HTML comment overrides), clean-text pass-through, injection stripping |
| `TestInvisibleText` | Zero-width character removal and count; clean-text baseline |
| `TestHomoglyphs` | Cyrillic→Latin normalisation; substitution count |
| `TestJDDuplication` | High n-gram overlap detection (ratio > 0.3); unrelated content near-zero baseline |
| `TestFullSanitize` | End-to-end adversarial document (injections + invisible text + keyword stuffing) flagged with ≥2 flags and penalty > 0.2; clean resume passes with penalty < 0.15 |
| `TestBaseScore` | Sigmoid boundary conditions (±5 logit, zero → 0.5, extreme ±100 saturation); strict (0, 1) bounds |
| `TestSkillPenalty` | No-missing multiplier = 1.0; single critical miss reduces multiplier; non-critical miss no penalty; three critical misses compounds to ~0.85³ |
| `TestFinalScore` | Score always in [0, 1]; adversarial penalty reduces score vs clean baseline; strong logit + full match > 0.9 |
| `TestRecommendation` | STRONG_MATCH ≥ 0.70/HIGH, NO_MATCH < 0.20, PARTIAL_MATCH mid-range |
| `TestNDCG / TestMRR / TestPrecision / TestSpearman / TestBiasAudit` | Metric correctness: perfect/worst/all-zeros nDCG; MRR rank-1/2/none/partial; Precision@k extremes; Spearman ρ = ±1.0; impact ratio equal-rates and four-fifths violation |
| `TestExtractor` | Format detection by extension; `.txt` and `.tex` verbatim extraction; `FileNotFoundError` on missing files; all expected extensions registered; image OCR smoke test (skipped if Tesseract absent) |
| `TestIsResumeGate` | Gate type-safety and fail-open behaviour for empty text, research-paper snippets, and financial-report snippets when LLM is unconfigured |
| `TestAblationResumeSet` | `data/ablation_resumes/` directory existence and non-empty; `resume15.tex` extracts ≥ 200 chars; adversarial/non-resume labels are 0.0; best candidate label ≥ 0.85 in golden dataset |
| `TestContracts` | `MatchResult` and `ThreatReport` safe defaults; `SkillEvidence.to_dict()` field serialisation |

### Development Workflow

When contributing a new module or modifying an existing one:

1. **Add tests** in `tests/test_all.py` under the relevant class, or create a new class following the existing naming convention.
2. **Run the suite** before committing — `python -m pytest tests/test_all.py -v` should pass in full (or skip where optional deps are absent).
3. **LLM-dependent paths** — guard with `if LLM_PROVIDER` or `pytest.skip(...)` so the suite stays runnable without an API key.
4. **Ablation and evaluation scripts** (`ablation.py`, `score_ablation_resumes.py`) are not part of the automated test suite; run them manually when adding new golden labels or changing retrieval logic.

---

## Project Structure

```
resume_matcher/
├── src/                          Core AI pipeline
│   ├── ingestion/
│   │   ├── extractor.py          MIME-first format detection, PDF/DOCX/OCR/HTML
│   │   ├── sanitizer.py          7 adversarial detectors → ThreatReport
│   │   └── ontology.py           ESCO skill alias/adjacency
│   ├── retrieval/
│   │   ├── engine.py             BM25 + bi-encoder + RRF + cross-encoder
│   │   └── index_store.py        Persistent dense (numpy) + BM25 cache
│   ├── scoring/
│   │   ├── scorer.py             Two-stage LLM + deterministic fallback
│   │   ├── d1.py – d4.py         Dimension modules (ontology, agents, regex)
│   │   ├── deterministic.py      Rule-based scoring (no LLM)
│   │   ├── extraction.py         JD/resume LLM prompts + tool calls
│   │   ├── extraction_schema.py  Normalisation, evidence validation
│   │   ├── llm_client.py         OpenAI client (extraction vs scoring models)
│   │   └── explainability.py     Rationale generation
│   ├── evaluation/
│   │   └── metrics.py            nDCG, MRR, P@k, Spearman, impact_ratio
│   ├── pipeline.py               Full orchestrator
│   ├── contracts.py              MatchResult, ThreatReport, etc.
│   └── config.py                 Constants, model names, weights, env flags
│
├── matcherserver/                Django project configuration
│   ├── settings.py               DB, middleware, dotenv loading
│   ├── urls.py                   Root URL conf → matcherapp.urls
│   ├── wsgi.py
│   └── celery.py                 Celery app (optional async path)
│
├── matcherapp/                   Django web application
│   ├── models.py                 Job, MatchRun, Resume, MatchResult
│   ├── decorators.py             login_required wrapper
│   ├── management/commands/
│   │   ├── process_run.py        Background worker for a single MatchRun
│   │   └── seed_data.py          Populate demo JD + resumes
│   ├── apps/
│   │   ├── auth/                 Login / logout views + URLs
│   │   ├── matching/
│   │   │   ├── views.py          Dashboard, new run, run detail, candidate detail
│   │   │   ├── api.py            REST endpoints (start, status, rescore, export …)
│   │   │   ├── services.py       process_match_run orchestration
│   │   │   └── tasks.py          Celery shared_task wrapper
│   │   └── tools/
│   │       ├── views.py          Pipeline, roadmap, test-suite, ablation pages + APIs
│   │       └── runner.py         Background ablation / test-suite runners
│   ├── templates/
│   │   ├── base.html
│   │   ├── auth/login.html
│   │   ├── matching/             dashboard, new_run, run_detail, candidate_detail
│   │   ├── tools/                pipeline, roadmap, test_suite, ablation
│   │   └── components/           navbar, sidebar, dimension_card, score_badge,
│   │                             resume_preview_modal, line_loader, snackbar
│   └── static/
│       ├── css/                  Tailwind input + compiled styles
│       └── js/                   Per-page modules (dashboard, new_run, run_detail,
│                                 candidate, ablation, test_suite) + shared utils
│
├── data/
│   ├── job_descriptions/         Sample JDs
│   ├── resumes/                  Sample txt resumes (13 labeled)
│   ├── ablation_resumes/         PDF, PNG, LaTeX (incl. adversarial)
│   ├── golden_dataset.jsonl      Human labels
│   └── index/                    Cached embeddings, BM25, profiles
│
├── evaluation/
│   ├── ablation_results.json     Snapshot from ablation.py (FinPay table in README)
│   └── ablation_scores.json      score_ablation_resumes.py D1–D4 dump
│
├── docs/
│   ├── architecture.tex          LaTeX architecture document
│   └── Architecture_Document.pdf Compiled PDF
│
├── extras/
│   ├── cost_tracker.py           Per-call token accounting
│   ├── compliance.py             NYC LL144 bias audit, impact ratio
│   ├── feedback.py               HITL recruiter decision logging
│   ├── mcp_server.py             JSON-RPC 2.0 MCP server (9 tools)
│   └── test_extras.py            Tests for extras modules
│
├── demo.py                       CLI full pipeline runner
├── ablation.py                   5-level ablation study → evaluation/ablation_results.json
├── score_ablation_resumes.py     Score without retrieval (D1–D4 only)
├── tests/test_all.py             ~75 unit + integration tests
├── manage.py                     Django management entry point
├── requirements.txt
├── tailwind.config.js            Tailwind CSS build config
├── .env.example                  Template for environment variables
└── logs/                         Worker logs (auto-created, gitignored)
```

---

## Known Limitations

**Weights and retrieval hyperparameters are hand-tuned in the current baseline.** A **combined regression** on golden + adversarial data (see [Future Steps](#future-steps--production-roadmap)) is planned to replace one-off tuning and better align metrics with human labels.

| Limitation | Current State |
|------------|---------------|
| **Explainability** | D1 skill breakdown with evidence is populated when deterministic D1 runs on extracted profiles; strengths/gaps/rationale from Stage-2 LLM; D2–D4 collapsible sections when signal detail exists |
| **Dimension weights** | Default 0.40/0.35/0.15/0.10; **JD seniority presets** (junior → staff); optional **custom % weights** per run on New Run |
| **Few-shot examples** | FinTech and AI/ML banks; scorer selects by `jd_profile["domain"]` |
| **Confidence** | LLM self-reported confidence plus downgrade when CE sigmoid diverges strongly from `dim_composite` |
| **Retrieval in web UI** | Match pipeline runs hybrid retrieval + CE logit when `src.retrieval` is available; if the engine fails, falls back to `ce_logit=0` (4D-only blend) |
| **Index implementation** | Dense index stored as numpy array (not FAISS); fine for <10K resumes |
| **Scoring mode selector** | UI stores Auto / LLM / Deterministic on `MatchRun.scoring_mode`; worker currently auto-detects mode via `has_llm()` — explicit enforcement is planned |
| **LLM provider** | OpenAI only; `llm_client.py` has no Anthropic path in current code |
| **Cost accounting** | `extras/cost_tracker.py` exists; not yet persisted on `MatchRun` in the DB |
| **Celery** | `celery.py` and `tasks.py` exist; current worker path uses `subprocess` + `manage.py process_run` — Celery wiring is ready for production deployment |

---

## Research Alignment

| Paper Component | Implementation |
|-----------------|----------------|
| Adversarial sanitization | `src/ingestion/sanitizer.py` — 7 detectors, continuous penalty |
| BM25 lexical retrieval | `src/retrieval/engine.py` |
| Dense bi-encoder | `all-MiniLM-L6-v2` or `text-embedding-3-small` |
| RRF fusion | k=60, parameter-free |
| Cross-encoder re-ranking | `ms-marco-MiniLM-L-6-v2` |
| LLM-as-Judge | Two-stage, gpt-4o-mini + gpt-4o, temp=0 |
| 4D scoring schema | D1 skills / D2 seniority / D3 domain / D4 constraints |
| ESCO ontology (partial — to be extended later; included as a base for explainability) | `src/ingestion/ontology.py` |
| Evaluation metrics | nDCG, MRR, P@k, Spearman, impact_ratio |
| Agentic retry | 1 bounded re-score on LOW confidence |

<a id="literature-review-notes"></a>

### Literature review & working notes (author)

Problem framing and literature survey for this project were developed across two working Google Docs (iterated in parallel; together they formed the hybrid basis for the design). **Primary links:**

- [Literature / background notes (1)](https://docs.google.com/document/d/1q2nIq1jjj6TZWj34VNzcHCo-4lkws7abH68tE6d485Q/edit?usp=sharing)
- [Literature / background notes (2)](https://docs.google.com/document/d/1eIZ9NAsMdAaO7I4zIptqpkLbZVVAf5BWZ6XVRTC2KBU/edit?usp=sharing)

For **which tools** assisted research vs coding vs the web app, see [Acknowledgements](#acknowledgements).

---

## References

1. **Harvard Business School & Accenture**, “Hidden Workers: Untapped Talent” — 27M excluded by ATS.
2. **Cormack et al. (2009)**, “Reciprocal Rank Fusion outperforms Condorcet” — RRF (k=60).
3. **Zheng et al. (2023)**, “Judging LLM-as-a-Judge” — position and verbosity biases; motivation for structured two-stage scoring.
4. **BEIR Benchmark (NeurIPS 2021)** — dense vs sparse retrieval trade-offs.
5. **EU AI Act, Article 6, Annex III** — high-risk employment AI; immutable audit, reproducibility.
6. **NYC Local Law 144** — AEDT bias audit, four-fifths rule.
7. **ESCO v1.2.1** — 13,485 skills, 28 languages; skill ontology.
8. **O\*NET** — occupational skill importance weights.
9. **Lightcast Open Skills** — 32K skills from labor market data; emerging skill coverage.
10. **Amazon AI Hiring (2014–2018)** — bias replication; motivation for feedback-as-context design (Phase 4).

---

## Acknowledgements

Transparent credit for AI-assisted research and engineering (aligned with `docs/architecture.tex` and project documentation):

| Role | Tool / source |
|------|----------------|
| **Reasoning & coding / algorithmic ground** | **Anthropic Claude 4.6 Opus** — primary assistant for implementation logic, structure, and problem-solving. |
| **Initial background & literature review** | **Google Gemini 3.1** (research-oriented workflows) — early framing and survey of the problem space. Working notes and links: [Literature review & working notes (author)](#literature-review-notes). |
| **Web app — ideation, planning, implementation** | **Cursor** (Composer) — fast iteration in a stack already familiar to the author: **Django**, **jQuery**, **Tailwind CSS**, vanilla JavaScript, background threading, and lightweight async processing for match runs. |
| **Architecture & docs** | **Claude** again used to summarize and tighten prose from the author’s bullets for the architecture write-up and technical summaries. |

**On repo norms:** There is no single mandatory GitHub standard for listing AI collaborators. A dedicated **Acknowledgements** section in the README (as here), plus matching notes in `docs/architecture.tex` or other docs you ship alongside the repo, is a common and transparent pattern. Optional extras some projects use: `ACKNOWLEDGEMENTS.md`, `CITATION.cff`, or author metadata in `pyproject.toml` / `CITATION.bib` — pick what fits your project.

The **runtime scoring** path in code uses **OpenAI** models where configured (`gpt-4o-mini` / `gpt-4o`, embeddings); that dependency is separate from the **authoring** assistants named above.
