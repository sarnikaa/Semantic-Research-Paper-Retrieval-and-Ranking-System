"""
Okapi BM25 Lexical Retriever implementation.
Supports tokenization, indexing, fast candidate scoring, and standard IR run generation.
"""

import re
import time
import logging
from typing import Dict, List, Tuple, Optional
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.data_loader import get_document_text

logger = logging.getLogger(__name__)


def simple_tokenizer(text: str) -> List[str]:
    """Lightweight, fast regex tokenizer with lowercasing."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Retriever:
    """Okapi BM25 sparse lexical search engine."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.bm25: Optional[BM25Okapi] = None

    def index(self, corpus: Dict[str, Dict[str, str]]) -> None:
        """
        Indexes a document collection.

        Args:
            corpus: Dict of {doc_id: {'title': str, 'text': str}}
        """
        logger.info(f"Indexing {len(corpus):,} documents with Okapi BM25 (k1={self.k1}, b={self.b})...")
        start_time = time.time()
        self.doc_ids = list(corpus.keys())
        
        tokenized_corpus = [
            simple_tokenizer(get_document_text(corpus[doc_id]))
            for doc_id in tqdm(self.doc_ids, desc="Tokenizing documents")
        ]
        
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        elapsed = time.time() - start_time
        logger.info(f"BM25 index built in {elapsed:.2f} seconds.")

    def search(self, query: str, top_k: int = 100) -> Dict[str, float]:
        """
        Retrieves top_k documents for a single query.

        Returns:
            Dict mapping doc_id -> score
        """
        if self.bm25 is None:
            raise ValueError("BM25 index has not been built. Call index(corpus) first.")

        tokenized_query = simple_tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices using argpartition for efficiency
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return {
            self.doc_ids[idx]: float(scores[idx])
            for idx in top_indices
            if scores[idx] > 0
        }

    def retrieve_all(
        self, queries: Dict[str, str], top_k: int = 100
    ) -> Tuple[Dict[str, Dict[str, float]], float]:
        """
        Searches for all queries and logs latency statistics.

        Returns:
            run: Dict mapping query_id -> {doc_id: score}
            avg_latency_ms: Average query execution latency in milliseconds
        """
        run: Dict[str, Dict[str, float]] = {}
        latencies_ms: List[float] = []

        logger.info(f"Executing BM25 search over {len(queries):,} queries...")
        for q_id, q_text in tqdm(queries.items(), desc="BM25 Queries"):
            t0 = time.perf_counter()
            ranked_docs = self.search(q_text, top_k=top_k)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            run[q_id] = ranked_docs

        avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
        logger.info(f"BM25 search completed. Avg latency: {avg_latency:.2f} ms/query.")
        return run, avg_latency
