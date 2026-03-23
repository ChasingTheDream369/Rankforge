# Resume-JD Matching Engine — AI-Powered Talent Screening POC

**Take-Home Assignment: Client Presentation**

An AI-powered resume screening system built as a proof-of-concept (POC) for enterprise Talent Acquisition. The core proposition is **explainable scoring** — every resume receives a 0–1 relevance score broken into four interpretable dimensions (plus CE blend), each traceable to specific evidence. Future: optimize dimension and CE weights jointly from labeled data.

---

## Table of Contents

1. [Overview](#overview)
2. [Technical Approach & Justification](#technical-approach--justification)
3. [Architecture](#architecture)
4. [Feature Engineering & Data Preprocessing](#feature-engineering--data-preprocessing)
5. [Performance Evaluation](#performance-evaluation)
6. [Challenges Overcome & Solutions](#challenges-overcome--solutions)
7. [Future Steps & Production Roadmap](#future-steps--production-roadmap)
8. [Setup & Usage](#setup--usage)
9. [Project Structure](#project-structure)
10. [Known Limitations](#known-limitations)
11. [References](#references)

---

## Overview

### Scenario

The Talent Acquisition team struggles with manual resume screening: it is **slow**, **inconsistent**, and likely **overlooks qualified candidates**. This POC demonstrates an automated system that scores and ranks resumes based on their relevance to a specific job description.

### Audience

This work is presented to the **Technical Lead of the AI Solutions team**, who cares about:

- **Technical approach and reasoning** — why we chose this design
- **Performance and limitations** — how well it works and where it falls short
- **Code quality and reproducibility** — clear, runnable, maintainable
- **Path to production** — what it takes to deploy at scale

### Core Deliverable

A **Python-based matching engine** that:

- **Input:** Job description + set of resumes (PDF, DOCX, image, LaTeX, plain text, ZIP)
- **Output:** Relevance score (0.0–1.0) per resume, plus per-dimension breakdowns, evidence, strengths, gaps, and confidence

### What the System Does

1. **Extracts and sanitizes** each resume against adversarial manipulation
2. **Retrieves** candidates using hybrid BM25 + semantic search, fused via Reciprocal Rank Fusion (RRF)
3. **Re-ranks** the shortlist with a cross-encoder
4. **Scores** each candidate on four structured dimensions using an LLM-as-Judge
5. **Returns** a ranked list with per-dimension scores, evidence quotes, strengths, gaps, and confidence

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
| **4D + CE scoring** | Explainable: D1 (skills), D2 (seniority), D3 (domain), D4 (constraints) + CE at fixed 25%. All candidates get normalized CE score. | Fixed weights; future: optimize D1–D4 and CE blend jointly from labels |
| **Ontology (ESCO-style)** | Skill alias normalisation (e.g. “rabbitmq” ≈ “kafka” as adjacent); partial credit for related skills | Ontology coverage limited; LLM fallback when skill not in graph |

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
       │         │  STAGE 2  LLM SCORING (gpt-4o, temp=0)      │
       │         │  Few-shot grounded examples                │
       │         │  D1–D4 overwritten by deterministic        │
       │         │  modules (ontology, regex, agents)          │
       │         │  Agentic retry: if confidence=LOW → 1       │
       │         │  bounded re-score (verify_score)            │
       │         └─────────────────────────────────────────────┘
       │                      │
       │                      ▼
       │         ┌─────────────────────────────────────────────┐
       │         │  FINAL SCORE FORMULA                        │
       │         │  dim = 0.40·D1 + 0.35·D2 + 0.15·D3 + 0.10·D4│
       │         │  raw = 0.75·dim + 0.25·σ(ce_logit)        │
       │         │  raw = (1−α)·dim + α·sigmoid(ce_logit)      │
       │         │  final = raw · (1 − adversarial_penalty)     │
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

**CE blend:** Fixed 75% dim + 25% CE. **CE_TOP_PERCENT** (env, 0–100, default 50): % of RRF pool that goes through real CE.

### Scoring Formulation: Layer by Layer

The final score is built in clearly separated layers. If we change a layer tomorrow, we update only that layer.

**Layer 1 — Dimension composite (75–100% of raw score):**

\[
\text{dim} = 0.40 \cdot D1 + 0.35 \cdot D2 + 0.15 \cdot D3 + 0.10 \cdot D4
\]

- **D1 (Skills, 40%):** Per required skill: BUILT_WITH=1.0, USED=0.7, LISTED=0.3, ABSENT=0.0. Ontology gives exact/adjacent/group credit. Weighted average; core skills weighted higher.
- **D2 (Seniority, 35%):** Four signals (leadership, architecture, scale, ownership) from `seniority_signals` + `total_years`. Agent + deterministic tools when enabled; else regex/heuristics.
- **D3 (Domain, 15%):** Same domain=1.0, adjacent=0.5–0.8, unrelated=0.0–0.3. Ontology + LLM fallback when domain missing.
- **D4 (Constraints, 10%):** Per hard constraint: met=1.0, partial=0.5, not met=0.0. Average across constraints.

**Layer 2 — CE weight (fixed):**

\[
\alpha = 0.25
\]

**Layer 3 — Raw score blend:**

\[
\text{raw} = 0.75 \cdot \text{dim} + 0.25 \cdot \sigma(\text{ce\_logit})
\]

- `σ(logit) = 1/(1 + exp(-logit))`. CE contributes 25% for all candidates.

**Layer 4 — Adversarial penalty:**

\[
\text{final} = \max(0, \min(1, \text{raw} \times (1 - \text{adversarial\_penalty})))
\]

- Penalty (0–0.95) comes from the 7 sanitizer detectors. Multiplicative; does not reject, but reduces score.

**Summary:** 75% dim + 25% CE (fixed); then multiplied by (1 − penalty). Each layer is independent; changes are local.

### Two Execution Paths

| Path | Retrieval | ce_logit | Use Case |
|------|-----------|----------|----------|
| **CLI / `demo.py` / `pipeline.py`** | Full hybrid (BM25 + dense + RRF + CE) | Real CE logit from engine | Research, ablation, evaluation |
| **Django Web UI (`services.py`)** | None (per-resume scoring only) | 0.0 → sigmoid=0.5, CE term = 12.5% | Operational simplicity; no shared index |

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

- **Skills:** Ontology alias mapping (e.g. “Golang” → “Go”); adjacency (e.g. rabbitmq ≈ kafka)
- **Domains:** Canonical mapping; `ADJACENT_DOMAINS` in schema
- **Seniority:** Normalised to schema vocabulary; deterministic tool implementations for D2 agent

### Deterministic Fallback (No LLM)

When `OPENAI_API_KEY` is not set:

- BM25 + cross-encoder for retrieval
- Regex-based skill matching (BUILD_VERBS, USE_VERBS)
- Keyword hit counts for D2, D3, D4

Accuracy is lower, especially for semantic skill equivalence and seniority judgement.

---

## Performance Evaluation

### Synthetic Evaluation Set

Since no labeled dataset exists, we create a small synthetic one:

- **Job description:** `data/job_descriptions/senior_backend_finpay.txt` (fintech backend) or `ema_enterprise_swe.txt`
- **Resumes:** 5–15 samples in `data/resumes/` (txt) and `data/ablation_resumes/` (PDF, PNG, LaTeX)
- **Labels:** `data/golden_dataset.jsonl` — manual labels per JD:
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

### Running Evaluation

```bash
# Ablation: compare 5 approaches on golden dataset
python ablation.py

# Custom JD and resumes
python ablation.py --jd my_jd.txt --resumes ./my_resumes/

# Demo: full pipeline + evaluation (when golden labels exist)
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

### Phase 1 — Demo-Ready

- **Optimized weights** — Joint optimization of dimension weights (D1–D4) and CE blend from golden labels; grid or gradient-based search for best nDCG
- **Surface `skill_detail` and profiles** in candidate detail UI — per-skill evidence table, `jd_profile`, `resume_profile`
- **JD-adaptive weights** — Derive D1–D4 weights from `jd_profile["seniority"]` (junior vs staff)
- **Domain-matched few-shot bank** — Add AI/ML examples; select by `jd_profile["domain"]` at scoring time
- **Confidence calibration** — BUILT_WITH ratio + ABSENT ratio + CE divergence from `dim_composite`

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
- **NYC Local Law 144:** `impact_ratio` per demographic group; four-fifths rule flagging
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
```

### Environment Variables

| Variable | Notes |
|----------|-------|
| `OPENAI_API_KEY` | **Required for LLM scoring.** Set via `.env` or `export`. Without it, deterministic fallback runs. |
| `CE_TOP_PERCENT` | % of RRF pool through real CE (0–100, default 50). Rest get RRF-derived logit. |
| `SECRET_KEY` | Django secret (required in production) |
| `DEBUG` | Set `False` in production |

**Security:** API keys must come from environment or `.env` only. Do not commit keys.

### Web UI

```bash
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Visit `http://127.0.0.1:8000` — log in, go to **New Run**, paste JD, upload resumes (up to 50; ZIP supported).

### CLI

```bash
# Demo: full pipeline on sample data
python demo.py

# Custom JD and resumes
python demo.py --jd data/job_descriptions/ai_engineer_ema.txt \
               --resumes data/ablation_resumes/

# Ablation: 5-approach metric comparison
python ablation.py

# Tests
python -m pytest tests/test_all.py -v
```

---

## Project Structure

```
resume_matcher/
├── src/                      Core AI pipeline
│   ├── ingestion/
│   │   ├── extractor.py      MIME-first format detection, PDF/DOCX/OCR/HTML
│   │   ├── sanitizer.py      7 adversarial detectors → ThreatReport
│   │   └── ontology.py       ESCO skill alias/adjacency
│   ├── retrieval/
│   │   ├── engine.py         BM25 + bi-encoder + RRF + cross-encoder
│   │   └── index_store.py   Persistent dense (numpy) + BM25 cache
│   ├── scoring/
│   │   ├── scorer.py         Two-stage LLM + deterministic fallback
│   │   ├── d1.py – d4.py     Dimension modules (ontology, agents, regex)
│   │   ├── extraction.py    JD/resume LLM prompts
│   │   ├── extraction_schema.py  Normalisation, evidence validation
│   │   ├── llm_client.py     OpenAI client
│   │   └── explainability.py Rationale generation
│   ├── evaluation/
│   │   └── metrics.py        nDCG, MRR, P@k, Spearman, impact_ratio
│   ├── pipeline.py           Full orchestrator
│   ├── contracts.py          MatchResult, ThreatReport, etc.
│   └── config.py             Constants, model names, weights
│
├── matcherapp/               Django web application
│   ├── models.py             Job, MatchRun, Resume, MatchResult
│   ├── apps/matching/        views, api, services
│   └── apps/tools/           pipeline, ablation, test suite pages
│
├── data/
│   ├── job_descriptions/     Sample JDs
│   ├── resumes/              Sample txt resumes
│   ├── ablation_resumes/     PDF, PNG, LaTeX (incl. adversarial)
│   ├── golden_dataset.jsonl  Human labels
│   └── index/                Cached embeddings, BM25, profiles
│
├── demo.py                   CLI full pipeline runner
├── ablation.py               5-approach ablation study
├── score_ablation_resumes.py Score without retrieval (D1–D4 only)
├── tests/test_all.py         ~75 unit + integration tests
├── extras/                   Optional: cost_tracker, feedback, compliance
└── requirements.txt
```

---

## Known Limitations

| Limitation | Current State |
|------------|---------------|
| **Explainability** | `skill_detail` (per-skill evidence table) only populated in deterministic fallback; LLM mode has strengths/gaps but not full evidence table in candidate UI |
| **Dimension weights** | Fixed 0.40/0.35/0.15/0.10 regardless of role level |
| **Few-shot examples** | FinPay-domain only; AI/ML JDs primed with fintech context |
| **Confidence** | `compute_confidence()` stub returns "MEDIUM" |
| **Django retrieval** | Web UI does not run hybrid retrieval; `ce_logit=0`; ranking is 4D blend only |
| **Index implementation** | Dense index stored as numpy array (not FAISS); fine for <10K resumes |
| **LLM provider** | OpenAI only; `llm_client.py` has no Anthropic path in current code |

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
| ESCO ontology | `src/ingestion/ontology.py` |
| Evaluation metrics | nDCG, MRR, P@k, Spearman, impact_ratio |
| Agentic retry | 1 bounded re-score on LOW confidence |

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
