"""
Hybrid Retrieval Engine — four-stage pipeline.

Stage 1a: BM25 sparse lexical retrieval (exact keyword matching)
Stage 1b: Dense bi-encoder semantic retrieval
           Priority: OpenAI text-embedding-3-small > sentence-transformers > TF-IDF
Stage 2:  Reciprocal Rank Fusion (RRF, k=60)
Stage 3:  Cross-encoder reranking on top-N candidates (always local)
"""

import re
import numpy as np
from typing import List, Tuple, Dict

from rank_bm25 import BM25Okapi
from src.config import (
    BM25_TOP_K, DENSE_TOP_K, RRF_K, RRF_TOP_K, CE_TOP_PERCENT, MAX_RESUMES_PER_RUN,
    USE_OPENAI_EMBEDDINGS, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL,
)

# Logit value for docs that never passed through CE → sigmoid ≈ 0 (no CE contribution)
NO_CE_LOGIT = -10.0
def tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())

# OpenAI text-embedding-3-small: 8191 tokens per input. ~1 token ≈ 4 chars (English),
# so 8191 × 4 ≈ 32k chars max. We use 6000 (~1500 tokens) to stay well under the limit
# and handle tokenizer variance (e.g. code, non-ASCII).
OPENAI_EMBED_MAX_CHARS = 6000
OPENAI_EMBED_BATCH_SIZE = 50  # process in batches to avoid huge single requests


def openai_embed(texts: List[str]) -> np.ndarray:
    """Get embeddings from OpenAI API. Returns normalized numpy array.
    Processes in batches of OPENAI_EMBED_BATCH_SIZE; truncates each text to OPENAI_EMBED_MAX_CHARS."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    truncated = [t[:OPENAI_EMBED_MAX_CHARS] for t in texts]
    all_embeddings = []
    for i in range(0, len(truncated), OPENAI_EMBED_BATCH_SIZE):
        batch = truncated[i : i + OPENAI_EMBED_BATCH_SIZE]
        resp = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=batch)
        batch_emb = np.array([d.embedding for d in resp.data])
        all_embeddings.append(batch_emb)
    embeddings = np.vstack(all_embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms
class RetrievalEngine:
    """Hybrid retrieval with BM25 + dense bi-encoder + RRF + cross-encoder."""

    def __init__(self):
        self.bm25 = None
        self.dense_embeddings = None
        self.doc_ids = []
        self.doc_texts = []
        self.last_ce_scores = {}
        self.last_ce_logits = {}  # raw logits before sigmoid

        # Cached query embedding for get_stage_scores (avoids N extra API calls per search)
        self._cached_query: str | None = None
        self._cached_query_emb: np.ndarray | None = None

        # Embedding provider
        self.embed_provider = None  # "openai" | "sentence-transformers" | "tfidf"
        self.encoder = None
        self.cross_encoder = None

        if USE_OPENAI_EMBEDDINGS:
            self.embed_provider = "openai"
            print(f"  Bi-encoder: OpenAI {OPENAI_EMBEDDING_MODEL}")
        else:
            try:
                from sentence_transformers import SentenceTransformer
                from src.config import EMBEDDING_MODEL
                self.encoder = SentenceTransformer(EMBEDDING_MODEL)
                self.embed_provider = "sentence-transformers"
                print(f"  Bi-encoder: {EMBEDDING_MODEL} (local)")
            except ImportError:
                self.embed_provider = "tfidf"
                print("  Bi-encoder: TF-IDF fallback (no sentence-transformers)")

        # Cross-encoder is always local — no API equivalent
        try:
            from sentence_transformers import CrossEncoder
            from src.config import CROSS_ENCODER_MODEL
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            print(f"  Cross-encoder: {CROSS_ENCODER_MODEL} (local)")
        except ImportError:
            self.cross_encoder = None
            print("  Cross-encoder: unavailable (sentence-transformers not installed)")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed texts using the best available provider."""
        if self.embed_provider == "openai":
            return openai_embed(texts)
        elif self.embed_provider == "sentence-transformers":
            return self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        else:
            return None  # TF-IDF handled separately

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        if self.embed_provider == "openai":
            return openai_embed([query])
        elif self.embed_provider == "sentence-transformers":
            return self.encoder.encode([query], show_progress_bar=False, normalize_embeddings=True)
        else:
            return None

    def index(self, documents: Dict[str, str],
              dense_documents: Dict[str, str] = None) -> None:
        """Build BM25 and dense indices from {doc_id: text}.

        documents       — clean sanitized text, used for BM25 (exact tokens only)
        dense_documents — ontology-expanded text for bi-encoder (optional).
                          Falls back to documents if not provided.
        """
        self.doc_ids = list(documents.keys())
        self.doc_texts = list(documents.values())

        # BM25 — raw clean text, no expansion (generic parent terms kill IDF)
        tokenized = [tokenize(t) for t in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized)

        # Dense — expanded text if provided, else same clean text
        dense_texts = [dense_documents[did] for did in self.doc_ids] \
            if dense_documents else self.doc_texts

        if self.embed_provider in ("openai", "sentence-transformers"):
            self.dense_embeddings = self.embed_texts(dense_texts)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                         sublinear_tf=True, stop_words='english')
            self.tfidf_matrix = self.tfidf.fit_transform(dense_texts)

    def search(self, query: str, dense_query: str = None) -> List[Tuple[str, float]]:
        """
        Full retrieval pipeline per the research architecture:
          1. BM25 → top BM25_TOP_K candidates (sparse lexical)
          2. Dense bi-encoder → top DENSE_TOP_K candidates (semantic)
          3. RRF fusion on the UNION of those two sets (~100 unique candidates)
          4. Cross-encoder reranking on top RERANK_TOP_N from RRF
        Returns [(doc_id, cross_encoder_score)] sorted descending.
        """
        n = len(self.doc_ids)
        if n == 0:
            return []

        # Stage 1a: BM25 — raw query, clean corpus (exact token matching)
        bm25_scores = self.bm25.get_scores(tokenize(query))

        # Stage 1b: Dense — ontology-expanded query against expanded corpus
        dq = dense_query if dense_query is not None else query
        if self.embed_provider in ("openai", "sentence-transformers") and self.dense_embeddings is not None:
            q_emb = self.embed_query(dq)
            dense_scores = np.dot(self.dense_embeddings, q_emb.T).flatten()
        elif self.embed_provider == "tfidf":
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec = self.tfidf.transform([dq])
            dense_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        else:
            dense_scores = np.zeros(n)

        # Stage 2: RRF fusion over ALL docs (rank-wise full coverage)
        # Every doc gets a rank-derived score; no one gets NO_CE_LOGIT
        bm25_rank_order = np.argsort(-bm25_scores)
        dense_rank_order = np.argsort(-dense_scores)
        bm25_rank_map = {int(idx): rank + 1 for rank, idx in enumerate(bm25_rank_order)}
        dense_rank_map = {int(idx): rank + 1 for rank, idx in enumerate(dense_rank_order)}

        rrf_results = []
        for idx in range(n):
            bm25_rank = bm25_rank_map.get(idx, n)
            dense_rank = dense_rank_map.get(idx, n)
            rrf_score = (1.0 / (RRF_K + bm25_rank)) + (1.0 / (RRF_K + dense_rank))
            rrf_results.append((idx, rrf_score))

        rrf_results.sort(key=lambda x: x[1], reverse=True)

        # Stage 3: RRF top-K = candidate pool
        # Top CE_TOP_PERCENT: real CE reranking (rank-based score)
        # Bottom (1-CE_TOP_PERCENT): fractional score by RRF position (marks 25% CE weight)
        rrf_pool_size = min(RRF_TOP_K, len(rrf_results))
        rrf_pool_indices = [idx for idx, _ in rrf_results[:rrf_pool_size]]
        n_ce = max(1, int(CE_TOP_PERCENT * rrf_pool_size))
        ce_indices = rrf_pool_indices[:n_ce]

        if self.cross_encoder and ce_indices:
            pairs = [(query, self.doc_texts[i]) for i in ce_indices]
            ce_raw_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)

            # Sort by CE raw score descending → rank 1 = best, rank n = worst
            ce_rank_order = np.argsort(-np.array(ce_raw_scores))
            # Rank-based score: rank 1 → 1.0, rank 2 → (n-1)/n, ..., rank n → 1/n
            n_ce_docs = len(ce_indices)
            rank_scores = np.zeros(n_ce_docs)
            for pos, orig_idx in enumerate(ce_rank_order):
                rank = pos + 1
                rank_score = (n_ce_docs - rank + 1) / n_ce_docs
                rank_scores[orig_idx] = max(0.01, min(0.99, rank_score))

            # Convert to logit for scorer: logit(p) = log(p/(1-p))
            ce_logits = np.log(rank_scores / (1.0 - rank_scores))

            results = []
            for i, idx in enumerate(ce_indices):
                doc_id = self.doc_ids[idx]
                results.append((doc_id, float(rank_scores[i])))
            results.sort(key=lambda x: x[1], reverse=True)

            self.last_ce_logits = {self.doc_ids[idx]: float(ce_logits[i])
                                   for i, idx in enumerate(ce_indices)}
            self.last_ce_scores = {self.doc_ids[idx]: float(rank_scores[i])
                                   for i, idx in enumerate(ce_indices)}
        else:
            self.last_ce_logits = {}
            self.last_ce_scores = {}
            results = []
            n_ce = 0

        # All remaining docs: RRF rank-derived score (connect rankwise fully, no -10)
        # Positions n_ce..rrf_pool_size: fractional score below worst CE doc
        # Positions rrf_pool_size..n: continue rank decay so lowest RRF rank gets smallest score
        worst_ce = 1.0 / n_ce if n_ce > 0 else 0.5
        n_after_ce = len(rrf_results) - n_ce
        for pos, (idx, _) in enumerate(rrf_results[n_ce:]):
            doc_id = self.doc_ids[idx]
            # Linear decay: rank 0 (pos 0) = just below worst CE, rank n_after_ce-1 = ~0.01
            frac_score = worst_ce * (1.0 - pos / max(n_after_ce, 1)) * 0.95
            frac_score = max(0.01, min(worst_ce - 0.01, frac_score))
            frac_logit = np.log(frac_score / (1.0 - frac_score))
            results.append((doc_id, float(frac_score)))
            self.last_ce_logits[doc_id] = float(frac_logit)
            self.last_ce_scores[doc_id] = float(frac_score)

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_cross_encoder_score(self, doc_id: str) -> float:
        """Get the sigmoid-normalized cross-encoder score."""
        return self.last_ce_scores.get(doc_id, 0.0)

    def get_cross_encoder_logit(self, doc_id: str) -> float:
        """Get the raw cross-encoder logit (before sigmoid). Used by scorer."""
        return self.last_ce_logits.get(doc_id, NO_CE_LOGIT)

    def get_stage_scores(self, query: str, doc_id: str) -> dict:
        """Get per-stage scores for a specific document (for explainability)."""
        idx = self.doc_ids.index(doc_id) if doc_id in self.doc_ids else -1
        if idx < 0:
            return {}

        bm25_scores = self.bm25.get_scores(tokenize(query))
        if self.embed_provider in ("openai", "sentence-transformers") and self.dense_embeddings is not None:
            if self._cached_query != query:
                self._cached_query = query
                self._cached_query_emb = self.embed_query(query)
            q_emb = self._cached_query_emb
            dense_scores = np.dot(self.dense_embeddings, q_emb.T).flatten()
        else:
            dense_scores = np.zeros(len(self.doc_ids))

        return {
            "bm25": round(float(bm25_scores[idx]), 4),
            "dense": round(float(dense_scores[idx]), 4),
            "embed_provider": self.embed_provider,
        }
