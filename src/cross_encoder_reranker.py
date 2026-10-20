"""
Neural Cross-Encoder Re-ranker module.
Uses ms-marco-MiniLM-L-6-v2 to re-score top candidate (query, document) pairs
with joint cross-attention for high top-10 precision.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import time
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

from src.data_loader import get_document_text

logger = logging.getLogger(__name__)


class NeuralReRanker:
    """Cross-encoder re-ranker scoring joint query-document text pairs."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 256,
        quantize: bool = False
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.is_quantized = False
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading CrossEncoder '{model_name}' on {self.device} (quantize={quantize})...")
        self.model = CrossEncoder(model_name, max_length=max_length, device=self.device)

        if quantize and self.device == "cpu":
            self.apply_dynamic_quantization()

    def apply_dynamic_quantization(self) -> None:
        """Applies PyTorch dynamic INT8 quantization to linear layers for fast CPU inference."""
        logger.info("Applying dynamic INT8 quantization to CrossEncoder model...")
        self.model.model = torch.quantization.quantize_dynamic(
            self.model.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        self.is_quantized = True
        logger.info("Dynamic INT8 quantization applied successfully.")

    def rerank_query(
        self,
        query: str,
        candidate_ids: List[str],
        corpus: Dict[str, Dict[str, str]],
        top_k: int = 10
    ) -> Dict[str, float]:
        """
        Re-scores a candidate list for a single query.

        Args:
            query: Query text
            candidate_ids: Ordered or unordered list of candidate document IDs
            corpus: Document dictionary
            top_k: Number of final re-ranked documents to return
        """
        if not candidate_ids:
            return {}

        pairs = [
            (query, get_document_text(corpus[doc_id]))
            for doc_id in candidate_ids
            if doc_id in corpus
        ]

        if not pairs:
            return {}

        scores = self.model.predict(pairs, show_progress_bar=False)

        # Sort descending by cross-encoder score
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return {
            candidate_ids[idx]: float(scores[idx])
            for idx in ranked_indices
        }

    def rerank_run(
        self,
        initial_run: Dict[str, Dict[str, float]],
        queries: Dict[str, str],
        corpus: Dict[str, Dict[str, str]],
        rerank_depth: int = 50,
        top_k: int = 10
    ) -> Tuple[Dict[str, Dict[str, float]], float]:
        """
        Re-ranks an entire initial candidate run (e.g. from BM25, Dense, or Hybrid).

        Args:
            initial_run: First-stage run mapping query_id -> {doc_id: score}
            queries: Mapping of query_id -> query_text
            corpus: Document collection
            rerank_depth: Number of top candidates per query to pass into cross-encoder
            top_k: Number of final ranked candidates to retain

        Returns:
            reranked_run: Dict of query_id -> {doc_id: cross_encoder_score}
            avg_latency_ms: Average re-ranking latency in milliseconds
        """
        reranked_run: Dict[str, Dict[str, float]] = {}
        latencies: List[float] = []

        logger.info(f"Re-ranking {len(queries):,} queries (depth={rerank_depth}, top_k={top_k})...")
        for q_id in tqdm(queries.keys(), desc="Cross-Encoder Re-ranking"):
            if q_id not in initial_run:
                continue

            q_text = queries[q_id]
            # Take top candidates from initial run
            sorted_candidates = sorted(
                initial_run[q_id].items(), key=lambda x: x[1], reverse=True
            )[:rerank_depth]
            candidate_ids = [doc_id for doc_id, _ in sorted_candidates]

            t0 = time.perf_counter()
            ranked = self.rerank_query(q_text, candidate_ids, corpus, top_k=top_k)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0)
            reranked_run[q_id] = ranked

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        logger.info(f"Cross-encoder re-ranking complete. Avg latency: {avg_latency:.2f} ms/query.")
        return reranked_run, avg_latency
