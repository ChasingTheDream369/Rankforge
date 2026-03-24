"""
Persistent Index Store — FAISS index save/load for production memory.

Instead of rebuilding the index on every run, persist it to disk.
When a new JD comes in, search against an already-indexed pool of
resumes rather than re-encoding everything.

Stores:
  - Dense embeddings (FAISS index file)
  - BM25 tokenized corpus (JSON)
  - Document metadata (ID → text mapping)
  - Index version + embedding model hash for invalidation

Cache invalidation: if the embedding model changes or corpus changes,
the index is automatically rebuilt.
"""

import os
import json
import hashlib
import pickle
import time
from typing import Dict, Optional, List
from pathlib import Path

from src.config import EMBEDDING_MODEL
INDEX_DIR = "data/index"


def compute_corpus_hash(documents: Dict[str, str]) -> str:
    """Hash the corpus to detect changes since last index build."""
    sorted_items = sorted(documents.items())
    content = json.dumps(sorted_items)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class IndexStore:
    """
    Persistent index with automatic cache invalidation.

    Usage:
        store = IndexStore()

        # First run: builds and saves
        if not store.is_valid(documents):
            store.build(documents)
            store.save()

        # Subsequent runs: load from disk
        store.load()
        results = store.search(query)
    """

    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = index_dir
        self.meta_path = os.path.join(index_dir, "index_meta.json")
        self.bm25_path = os.path.join(index_dir, "bm25_corpus.pkl")
        self.dense_path = os.path.join(index_dir, "dense_index.npy")
        self.docs_path = os.path.join(index_dir, "documents.json")
        self.skills_path = os.path.join(index_dir, "skills_cache.json")

        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.dense_embeddings = None
        self.bm25_tokens = None
        self.skills_cache: Dict[str, dict] = {}  # {resume_id: structured extraction result}
        self.meta = {}

    def is_valid(self, documents: Dict[str, str]) -> bool:
        """Check if saved index matches current corpus + model."""
        if not os.path.exists(self.meta_path):
            return False
        try:
            with open(self.meta_path, 'r') as f:
                meta = json.load(f)
            corpus_hash = compute_corpus_hash(documents)
            return (meta.get("corpus_hash") == corpus_hash and
                    meta.get("embedding_model") == EMBEDDING_MODEL)
        except Exception:
            return False

    def build(self, documents: Dict[str, str]) -> None:
        """Build BM25 + dense index from documents."""
        import re
        import numpy as np

        self.doc_ids = list(documents.keys())
        self.doc_texts = list(documents.values())

        # BM25 tokens
        self.bm25_tokens = [re.findall(r'\b\w+\b', t.lower()) for t in self.doc_texts]

        # Dense embeddings
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(EMBEDDING_MODEL)
            self.dense_embeddings = encoder.encode(
                self.doc_texts, show_progress_bar=False, normalize_embeddings=True
            )
        except ImportError:
            # TF-IDF fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
            self.dense_embeddings = tfidf.fit_transform(self.doc_texts).toarray()

        self.meta = {
            "corpus_hash": compute_corpus_hash(documents),
            "embedding_model": EMBEDDING_MODEL,
            "num_documents": len(documents),
            "built_at": time.time(),
            "embedding_dim": self.dense_embeddings.shape[1] if self.dense_embeddings is not None else 0,
        }

        print(f"  Index built: {len(self.doc_ids)} docs, "
              f"dim={self.meta['embedding_dim']}")

    def save(self) -> str:
        """Persist index + skills cache to disk."""
        import numpy as np

        os.makedirs(self.index_dir, exist_ok=True)

        with open(self.meta_path, 'w') as f:
            json.dump(self.meta, f, indent=2)

        with open(self.docs_path, 'w') as f:
            json.dump({"doc_ids": self.doc_ids, "doc_texts": self.doc_texts}, f)

        with open(self.bm25_path, 'wb') as f:
            pickle.dump(self.bm25_tokens, f)

        if self.dense_embeddings is not None:
            np.save(self.dense_path, self.dense_embeddings)

        if self.skills_cache:
            with open(self.skills_path, 'w') as f:
                json.dump(self.skills_cache, f, indent=2)

        print(f"  Index saved to {self.index_dir}/")
        return self.index_dir

    def load(self) -> bool:
        """Load index + skills cache from disk. Returns True if successful."""
        import numpy as np

        try:
            with open(self.meta_path, 'r') as f:
                self.meta = json.load(f)

            with open(self.docs_path, 'r') as f:
                data = json.load(f)
                self.doc_ids = data["doc_ids"]
                self.doc_texts = data["doc_texts"]

            with open(self.bm25_path, 'rb') as f:
                self.bm25_tokens = pickle.load(f)

            if os.path.exists(self.dense_path):
                self.dense_embeddings = np.load(self.dense_path)

            if os.path.exists(self.skills_path):
                with open(self.skills_path, 'r') as f:
                    self.skills_cache = json.load(f)

            print(f"  Index loaded: {len(self.doc_ids)} docs from {self.index_dir}/")
            return True
        except Exception as e:
            print(f"  Index load failed: {e}")
            return False

    def get_stats(self) -> dict:
        """Index statistics for monitoring."""
        return {
            "num_documents": len(self.doc_ids),
            "embedding_model": self.meta.get("embedding_model", "unknown"),
            "embedding_dim": self.meta.get("embedding_dim", 0),
            "corpus_hash": self.meta.get("corpus_hash", ""),
            "built_at": self.meta.get("built_at", 0),
            "index_dir": self.index_dir,
            "dense_index_exists": self.dense_embeddings is not None,
            "bm25_index_exists": self.bm25_tokens is not None,
        }

    def invalidate(self) -> None:
        """Force index rebuild on next run."""
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)
        print("  Index invalidated — will rebuild on next run")
