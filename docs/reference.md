# RankForge — extended reference

Benchmarks, evaluation, roadmap, limitations, research alignment, citations, and AI-tooling disclosure. For **overview, technical approach, architecture, setup, tests, and repo layout**, see the main **[README](../README.md)**.

## Table of contents

1. [Feature Engineering & Data Preprocessing](#feature-engineering--data-preprocessing)
2. [Performance Evaluation](#performance-evaluation)
3. [Challenges Overcome & Solutions](#challenges-overcome--solutions)
4. [Future Steps & Production Roadmap](#future-steps--production-roadmap)
5. [Known Limitations](#known-limitations)
6. [Research alignment & literature notes](#research-alignment)
7. [References](#references)
8. [Acknowledgements](#acknowledgements)

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

Evaluation uses small **human-labeled** pools (reproducible, not large-scale). **Ablation** results below split into **benchmark A** (15 real resumes, EMA backend JD) and **benchmark B** (13 FinPay-style `.txt` profiles) — see [Ablation results](#ablation-results-two-separate-benchmarks).

**Labels:** `data/golden_dataset.jsonl` maps each JD key → `{ resume_stem: grade }`:

- **1.0** — strong match  
- **0.5** — partial  
- **0.0** — poor  

**Corpora:** **A** — `ema_enterprise_swe` + 15 real resume files; **B** — `senior_backend_finpay` + `data/resumes/` (13 labeled `.txt`). **`data/ablation_resumes/`** — PDF/PNG/LaTeX etc. for UI and adversarial checks; add stems to `golden_dataset.jsonl` when you want them in `ablation.py`.

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

### Ablation results (two separate benchmarks)

The tables below are **not** comparable row-for-row: they use **different JDs**, **different resume pools**, and **different label sets**. Use **A** for a **small real-resume** slice; use **B** for the **FinPay-style** labeled `.txt` corpus.

---

<a id="ablation-latest-ema-mar-2026"></a>

#### A — EMA backend engineer JD, **15 real resumes** (recorded 25 March 2026)

**Corpus:** **15 real resume files** (project corpus) with golden labels under JD key **`ema_enterprise_swe`** in `data/golden_dataset.jsonl` — aligned with an **EMA / enterprise backend software engineer** role profile.

**Ladder:** Same five steps as the in-app **Ablation** UI: sparse baselines → TF-IDF + BM25 → BM25 + dense + RRF → cross-encoder on the shortlist → **full pipeline** (profile extraction, ontology-grounded D1/D3, D2 agent/tools when enabled, merged D1–D4, LOW-confidence verification).

| Approach | nDCG@3 | nDCG@5 | nDCG@10 | MRR | P@3 | P@5 | Spearman ρ | TOP-1† |
|----------|-------:|-------:|--------:|----:|----:|----:|-----------:|-------:|
| TF-IDF only | 0.229 | 0.471 | 0.667 | 0.250 | 0.000 | 0.200 | 0.304 | 0.093 |
| TF-IDF + BM25 | 0.436 | 0.480 | 0.584 | 0.500 | 0.333 | 0.200 | 0.214 | 0.032 |
| BM25 + Dense + RRF | 0.738 | 0.726 | 0.712 | 0.333 | 0.333 | 0.400 | 0.329 | 0.032 |
| + Cross-encoder | 0.304 | 0.444 | 0.643 | 0.200 | 0.000 | 0.200 | 0.389 | 0.158 |
| Hybrid + Agentic (Ontology Grounding) | 0.498 | 0.690 | **0.786** | 0.250 | 0.000 | 0.400 | **0.454** | **0.720** |

† **TOP-1** (UI): model score at rank 1 after that stage — a sharpness readout, not gold **accuracy@1**.

**How reviewers should read A:** Hybrid retrieval can lead **early-cut** metrics (e.g. nDCG@3, MRR) while the **full stack (stage 5)** leads **nDCG@10** and **Spearman ρ** on this run — consistent with graded shortlist review rather than “winner takes all” on every column. Weights, CE blend, RRF *k*, and pool cut remain **conservative**; joint tuning on labels is tracked under [Future Steps](#future-steps--production-roadmap). **Primary headline on this table:** **nDCG@10** (graded + position discount); MRR / P@k / Spearman are **supporting** signals (see `docs/architecture.tex` for the metric definitions).

---

#### B — FinPay-style senior backend JD, **13 labeled `.txt` profiles** (`data/resumes/`)

**Corpus:** **`data/job_descriptions/senior_backend_finpay.txt`** + **13 labeled** resume stems in `data/resumes/` (golden key **`senior_backend_finpay`** in `data/golden_dataset.jsonl`). This is a **FinPay-flavour** backend stack scenario — **not** the same pool as **A**.

**Ladder:** `ablation.py` — (1) TF-IDF only → (2) TF-IDF + BM25 (RRF) → (3) BM25 + bi-encoder (RRF) → (4) + cross-encoder → (5) full pipeline (`src/pipeline.py`). Regenerate anytime; committed snapshot: **`evaluation/ablation_results.json`**.

| Level | Approach | nDCG@3 | nDCG@5 | nDCG@10 | MRR | P@3 | P@5 | Spearman ρ |
|------:|----------|-------:|-------:|--------:|----:|----:|----:|-----------:|
| 1 | TF-IDF cosine only | 1.000 | 0.915 | 0.974 | 1.000 | 1.000 | 0.800 | 0.558 |
| 2 | TF-IDF + BM25 (RRF) | 1.000 | 0.915 | 0.974 | 1.000 | 1.000 | 0.800 | 0.637 |
| 3 | BM25 + bi-encoder (RRF) | 1.000 | 0.915 | **0.976** | 1.000 | 1.000 | 0.800 | **0.681** |
| 4 | + Cross-encoder rerank | 0.765 | 0.727 | 0.856 | 1.000 | 0.667 | 0.600 | 0.267 |
| 5 | + Full system (ontology / 4D) | 0.765 | 0.727 | **0.922** | 1.000 | 0.667 | 0.600 | 0.563 |

**How reviewers should read B:**

- **Stages 1 → 3:** On this JD, **nDCG@3 stays at 1.0** because the top three gold “good” profiles already sit at the head under raw TF-IDF. **Spearman ρ** (0.558 → **0.681**) shows hybrid retrieval **aligning the full ordering** across all 13 labeled files.
- **Stage 4:** CE rerank can **dip** graded nDCG / P@k on a **small** label set while MRR stays high — a sample-size artefact, not a general claim that CE is harmful.
- **Stage 5 vs 4:** Full stack **recovers** **nDCG@10** (0.856 → 0.922) and **Spearman** (0.267 → 0.563) while adding **D1–D4, ontology grounding, and sanitizer path** (not shown in the metric columns).

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

# FinPay + labeled txt corpus (reproduces benchmark **B**)
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

The table below maps **documented failure modes** in keyword-only ATS paths and naive LLM-only wrappers—**synonym blindness**, **adversarial gaming**, **historical bias in training targets**, **cost/latency at scale**—to concrete mechanisms in this repo. The intent is the same as in the research notes: **defensible ingestion**, **hybrid retrieval**, **grounded scoring**, and **inspectable outputs** rather than a single end-to-end generative grade.

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

**Calibration priority.** The current codebase keeps the **base scorer stable** (same formulas and evidence paths). The next improvement pass is to **discover weights and ordering of signals jointly**: dimension weights (D1–D4), CE blend α, retrieval fusion (RRF *k*, CE pool cut), and optional score floors/caps — fitted with **combined regression or constrained grid search** on golden labels, ablation JSON, and adversarial uploads so nDCG / Spearman and “gaming resistance” move together. Other high-value items: wire `cost_tracker` to per-run DB totals, expand few-shot banks per industry, richer confidence calibration (skill evidence + CE–dim divergence), and optional demographic-aware fairness dashboards (impact ratio already implemented; threshold is configurable — see env table).

### Phase 1 — Calibration & visibility

- **Combined regression / grid search** — Joint optimization of D1–D4 weights, CE weight, and key retrieval hyperparameters against `golden_dataset.jsonl` + adversarial suite; primary objective nDCG@10 / Spearman, secondary robustness on ablation resumes
- **Surface full profiles in UI** — Optional collapsible `jd_profile` / `resume_profile` JSON for auditors (skill table already shown when populated)
- **Confidence calibration** — Tune thresholds on BUILT_WITH vs ABSENT ratios and CE vs `dim_composite` divergence beyond current heuristics
- **Cost visibility** — Persist `extras/cost_tracker.py` aggregates per `MatchRun` for stakeholder “cost per resume” answers

**Already in this codebase:** JD seniority–based dimension presets (`ROLE_WEIGHTS`), optional **custom dimension importance** on New Run (normalized, stored in `MatchRun.scoring_config`), domain-selected few-shot bank (FinTech vs AI/ML), applied weights surfaced on candidate detail, agentic LOW-confidence re-score, 7-detector adversarial penalty model.

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


## Known Limitations

**Tuning:** Dimension weights, CE blend, and retrieval hyperparameters are **hand-set** today; joint fit on golden + adversarial data is outlined under [Future Steps](#future-steps--production-roadmap).

| Area | Current state |
|------|----------------|
| **Scoring mode (UI vs worker)** | New Run stores Auto / LLM / Deterministic on `MatchRun`; the worker still **auto-picks** capability from `has_llm()` — strict per-run enforcement is not wired end-to-end yet. |
| **Retrieval in web runs** | Full hybrid + CE logit when `src.retrieval` loads; on failure, **`ce_logit=0`** and scores rely on the 4D blend only. |
| **Explainability** | D1 evidence when deterministic D1 runs on profiles; strengths/gaps/rationale from the scoring LLM when enabled; D2–D4 detail when those modules emit structured fields. |
| **Weights & presets** | Defaults 0.40 / 0.35 / 0.15 / 0.10; JD seniority presets; optional **custom %** per run in the UI. |
| **Few-shot banks** | Domain-tagged examples (e.g. FinTech, AI/ML); chosen from `jd_profile["domain"]`. |
| **Confidence** | LLM-reported band, downgraded when CE sigmoid and `dim_composite` disagree strongly. |
| **Dense index** | NumPy-backed in-process store — adequate for **small** corpora; not a FAISS-scale service yet. |
| **LLM vendor (runtime)** | OpenAI only in `llm_client.py` — no Anthropic path in code. |
| **Cost persistence** | `extras/cost_tracker.py` is not written onto `MatchRun` rows. |
| **Background jobs** | `manage.py process_run` via **subprocess**; Celery modules exist but are not the default path. |

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

### Literature review & working notes

Background synthesis and survey material are maintained in two working Google Docs (parallel iterations; both informed the design). **Links:**

- [Literature / background notes (1)](https://docs.google.com/document/d/1q2nIq1jjj6TZWj34VNzcHCo-4lkws7abH68tE6d485Q/edit?usp=sharing)
- [Literature / background notes (2)](https://docs.google.com/document/d/1eIZ9NAsMdAaO7I4zIptqpkLbZVVAf5BWZ6XVRTC2KBU/edit?usp=sharing)

For **AI-assisted tooling** disclosure (research vs implementation vs web stack), see [Acknowledgements](#acknowledgements).

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

**AI-assisted tooling** used in building this repository (for reviewer transparency):

| Role | Tool / source |
|------|----------------|
| **Reasoning, implementation, algorithms** | **Anthropic Claude 4.6 Opus** — structure, coding, and design reasoning. |
| **Background synthesis & literature-oriented drafting** | **Google Gemini 3.1** — early framing and survey work. Consolidated links: [Literature review & working notes](#literature-review-notes). |
| **Web application iteration** | **Cursor** (Composer) — planning and implementation on **Django**, **jQuery**, **Tailwind CSS**, vanilla JavaScript, background workers, and lightweight async handling for match runs. |
| **Architecture narrative** | **Claude** — distillation of design notes into `docs/architecture.tex` and long-form technical summaries in this repo. |

**Runtime scoring** (the match pipeline) calls **OpenAI** when configured (`gpt-4o-mini` / `gpt-4o`, embeddings). That production API usage is **separate** from the assistants listed above.
