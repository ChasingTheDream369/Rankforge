"""
Configuration — single source of truth for all pipeline parameters.
All constants in-code. No CLI arguments. Environment variables for secrets only.
OpenAI only — Anthropic removed.
"""

import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

LLM_PROVIDER = "openai" if OPENAI_API_KEY else "regex"

OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = 0.0

# Two-stage scorer models
EXTRACTION_MODEL = "gpt-4o-mini"   # Stage 1: cheap profile extraction
SCORING_MODEL = "gpt-4o"           # Stage 2: few-shot scoring with reasoning
EXTRACTION_USE_TOOLS = True        # Use canonicalize_skill/canonicalize_domain tools for deterministic LLM output
D2_AGENT_ENABLED = True            # Use agent + tools for D2 (Seniority) scoring; fallback to deterministic if disabled or fails
D3_LLM_FALLBACK = True            # When domain not in ontology (exact/adjacent), use LLM tool
D1_LLM_FALLBACK = True             # When skill not in ontology (exact/adjacent/group), use LLM tool

# Embedding Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIM = 1536
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 384
USE_OPENAI_EMBEDDINGS = bool(OPENAI_API_KEY)

# Retrieval
BM25_TOP_K = 50
DENSE_TOP_K = 50
RRF_K = 60
RRF_TOP_K = 50          # top 50 from RRF form candidate pool before CE
# Top % of RRF pool that goes through real CE. Env: CE_TOP_PERCENT (0-100, default 50)
try:
    _ce_top = os.environ.get("CE_TOP_PERCENT")
    CE_TOP_PERCENT = max(0.0, min(1.0, float(_ce_top) / 100.0)) if _ce_top is not None else 0.50
except (TypeError, ValueError):
    CE_TOP_PERCENT = 0.50
MAX_RESUMES_PER_RUN = 50

# Scoring
CRITICAL_SKILL_PENALTY = 0.85
CRITICAL_IMPORTANCE_THRESHOLD = 0.8  # importance >= 4 out of 5
CE_WEIGHT = 0.25  # (1-CE_WEIGHT)*dim + CE_WEIGHT*sigmoid(ce_logit)

# Paths
JD_DIR = "data/job_descriptions"
RESUME_DIR = "data/resumes"
GOLDEN_DATASET_PATH = "data/golden_dataset.jsonl"
FEEDBACK_DIR = "data/feedback"

# Determinism
RANDOM_SEED = 42
SCORE_PRECISION = 4
