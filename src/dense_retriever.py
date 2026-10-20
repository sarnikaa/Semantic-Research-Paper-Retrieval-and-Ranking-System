"""
Dense Semantic Retriever implementation using Sentence-Transformers and Vector Search.
Supports computing embeddings locally or loading pre-computed .npy matrices (e.g. from Colab),
with FAISS index support and pure PyTorch/NumPy exact cosine fallback.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import time
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from src.data_loader import get_document_text

logger = logging.getLogger(__name__)


MODEL_ALIASES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "specter2": "allenai/specter2_base",
    "scibert": "allenai/scibert_scivocab_uncased",
}


class DenseRetriever:
    """Dense vector retriever using transformer embeddings and FAISS / cosine vector search."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 64,
        use_faiss: bool = True
    ):
        resolved_name = MODEL_ALIASES.get(model_name.lower().strip(), model_name)
        self.model_name = resolved_name
        self.batch_size = batch_size
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing DenseRetriever (model='{self.model_name}', device='{self.device}', faiss={self.use_faiss})...")
        self.encoder = SentenceTransformer(self.model_name, device=self.device)
        self.doc_ids: List[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.faiss_index = None

    def encode_corpus(
        self,
        corpus: Dict[str, Dict[str, str]],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """Encodes all corpus documents into normalized dense vectors."""
        self.doc_ids = list(corpus.keys())
        texts = [get_document_text(corpus[doc_id]) for doc_id in self.doc_ids]

        logger.info(f"Encoding {len(texts):,} documents using {self.model_name}...")
        start_time = time.time()
        embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        elapsed = time.time() - start_time
        logger.info(f"Encoded {len(texts):,} documents in {elapsed:.2f}s ({len(texts)/elapsed:.1f} docs/sec).")

        self.doc_embeddings = embeddings
        if self.use_faiss:
            self.build_faiss_index()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, {"doc_ids": self.doc_ids, "embeddings": self.doc_embeddings})
            logger.info(f"Saved document embeddings to {save_path}")

        return embeddings

    def build_faiss_index(self) -> None:
        """Constructs an in-memory FAISS IndexFlatIP for sub-linear search."""
        if not FAISS_AVAILABLE or self.doc_embeddings is None:
            return
        d = self.doc_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(self.doc_embeddings.astype(np.float32))
        logger.info(f"Built FAISS IndexFlatIP index with {self.faiss_index.ntotal} vectors of dimension {d}.")

    def load_embeddings(self, load_path: str) -> None:
        """Loads precomputed embeddings from disk (e.g., generated on Colab)."""
        logger.info(f"Loading precomputed embeddings from {load_path}...")
        data = np.load(load_path, allow_pickle=True).item()
        self.doc_ids = data["doc_ids"]
        self.doc_embeddings = data["embeddings"]
        logger.info(f"Loaded {len(self.doc_ids):,} embeddings of dimension {self.doc_embeddings.shape[1]}.")
        if self.use_faiss:
            self.build_faiss_index()

    def search(self, query: str, top_k: int = 100) -> Dict[str, float]:
        """Encodes a single query and returns top_k cosine-similar documents."""
        if self.doc_embeddings is None:
            raise ValueError("No corpus embeddings found. Call encode_corpus() or load_embeddings() first.")

        query_vec = self.encoder.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # Use FAISS index if initialized
        if self.faiss_index is not None:
            q_mat = query_vec.reshape(1, -1).astype(np.float32)
            scores, indices = self.faiss_index.search(q_mat, top_k)
            return {
                self.doc_ids[idx]: float(scores[0][i])
                for i, idx in enumerate(indices[0])
                if idx >= 0
            }

        # Fallback to pure matrix dot product
        scores = np.dot(self.doc_embeddings, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {
            self.doc_ids[idx]: float(scores[idx])
            for idx in top_indices
        }

    def retrieve_all(
        self,
        queries: Dict[str, str],
        top_k: int = 100
    ) -> Tuple[Dict[str, Dict[str, float]], float]:
        """Encodes all queries and computes batch ranking scores."""
        query_ids = list(queries.keys())
        query_texts = [queries[q_id] for q_id in query_ids]

        logger.info(f"Encoding and searching {len(query_texts):,} queries...")
        t0 = time.perf_counter()
        query_embeddings = self.encoder.encode(
            query_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        # Batch dot product: [N_queries, D] x [D, N_docs] = [N_queries, N_docs]
        similarity_matrix = np.dot(query_embeddings, self.doc_embeddings.T)
        t1 = time.perf_counter()
        total_search_time_ms = (t1 - t0) * 1000.0

        run: Dict[str, Dict[str, float]] = {}
        for i, q_id in enumerate(query_ids):
            scores = similarity_matrix[i]
            top_indices = np.argsort(scores)[::-1][:top_k]
            run[q_id] = {self.doc_ids[idx]: float(scores[idx]) for idx in top_indices}

        avg_latency_ms = total_search_time_ms / len(query_ids)
        logger.info(f"Dense search completed. Avg latency: {avg_latency_ms:.2f} ms/query.")
        return run, avg_latency_ms
