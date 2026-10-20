"""
Late-Interaction Token Retrieval Prototype (ColBERT-style MaxSim).
Computes token-level representation alignments:
Score(Q, D) = sum_{q in Q} max_{d in D} (E_q . E_d^T)

Bridges the paradigm gap between Bi-Encoders and Cross-Encoders,
enabling fine-grained token-level alignment with precomputable document tokens.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
import time
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class LateInteractionMaxSimRetriever:
    """Late-interaction token retriever using contextualized token embeddings and MaxSim."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing LateInteractionMaxSimRetriever on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.doc_token_embeddings: Dict[str, torch.Tensor] = {}

    def _encode_tokens(self, text: str, max_length: int = 128) -> torch.Tensor:
        """Encodes text into normalized token-level embedding vectors [seq_len, dim]."""
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # outputs.last_hidden_state: [1, seq_len, dim]
            token_embeddings = outputs.last_hidden_state.squeeze(0)  # [seq_len, dim]
            # Normalize token embeddings
            norm = torch.norm(token_embeddings, p=2, dim=1, keepdim=True).clamp(min=1e-9)
            normalized_tokens = token_embeddings / norm

        return normalized_tokens

    def index_documents(self, corpus: Dict[str, Dict[str, str]], max_docs: int = 200) -> None:
        """Indexes token embeddings for candidate documents."""
        logger.info(f"Indexing token embeddings for {min(len(corpus), max_docs)} documents...")
        t0 = time.time()
        for doc_id, doc in list(corpus.items())[:max_docs]:
            title = doc.get("title", "")
            text = doc.get("text", "")
            combined = f"{title} {text}".strip()
            self.doc_token_embeddings[doc_id] = self._encode_tokens(combined, max_length=128)
        logger.info(f"Indexed {len(self.doc_token_embeddings)} documents in {time.time() - t0:.2f}s.")

    def compute_maxsim(self, query_tokens: torch.Tensor, doc_tokens: torch.Tensor) -> float:
        """
        Computes MaxSim: sum_{q in Q} max_{d in D} (E_q . E_d^T)
        Args:
            query_tokens: [len_q, dim]
            doc_tokens: [len_d, dim]
        """
        # Similarity matrix: [len_q, len_d]
        sim_matrix = torch.matmul(query_tokens, doc_tokens.t())
        # Max along doc dimension: [len_q]
        max_sims, _ = torch.max(sim_matrix, dim=1)
        # Sum over query tokens
        score = torch.sum(max_sims).item()
        return float(score)

    def search(self, query: str, top_k: int = 10) -> Dict[str, float]:
        """Performs Late-Interaction MaxSim search over indexed documents."""
        q_tokens = self._encode_tokens(query, max_length=64)
        scores: Dict[str, float] = {}

        for doc_id, d_tokens in self.doc_token_embeddings.items():
            scores[doc_id] = self.compute_maxsim(q_tokens, d_tokens)

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return {doc_id: score for doc_id, score in sorted_results}
