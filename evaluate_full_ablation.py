"""
Phase 4: Full Multi-Stage Ablation Evaluation Script.
Evaluates the entire pipeline:
1. Okapi BM25
2. Dense (all-MiniLM-L6-v2)
3. Hybrid RRF (BM25 + Dense)
4. Hybrid + Cross-Encoder Re-Ranking (ms-marco-MiniLM-L-6-v2)

Computes nDCG@10, MAP, MRR@10, Recall@100, latency statistics,
and generates the complete publication-ready comparative ablation table.
"""

import os
import sys
import argparse
import logging
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data_loader import load_beir_dataset
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.hybrid_fusion import reciprocal_rank_fusion
from src.cross_encoder_reranker import NeuralReRanker
from src.evaluator import IREvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("full_ablation")


def run_full_ablation(
    dataset_name: str = "scifact",
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 100,
    rerank_depth: int = 50,
    data_dir: str = "data",
    results_dir: str = "results"
):
    print("=" * 85)
    print(f"  FULL MULTI-STAGE ABLATION: BM25 -> DENSE -> HYBRID -> CROSS-ENCODER")
    print(f"  Dataset: {dataset_name.upper()} | Eval Metric Set: nDCG@10, MAP, MRR@10, Recall@100")
    print("=" * 85)

    # 1. Load Dataset
    corpus, queries, qrels = load_beir_dataset(dataset_name=dataset_name, data_dir=data_dir, split="test")
    evaluator = IREvaluator(qrels)
    metrics = ["ndcg@10", "map", "mrr@10", "recall@100"]

    # 2. Stage 1: BM25
    print("\n>>> [1/4] Okapi BM25 Lexical Retrieval...")
    bm25 = BM25Retriever()
    bm25.index(corpus)
    bm25_run, bm25_latency = bm25.retrieve_all(queries, top_k=top_k)
    bm25_scores = evaluator.evaluate_run(bm25_run, model_name="BM25", metrics=metrics)
    bm25_scores["latency_ms"] = round(bm25_latency, 2)

    # 3. Stage 2: Dense Retrieval
    print("\n>>> [2/4] Dense Semantic Vector Retrieval...")
    emb_cache = os.path.join(data_dir, f"{dataset_name}_embeddings_minilm.npy")
    dense = DenseRetriever(model_name=dense_model_name)
    if os.path.exists(emb_cache):
        logger.info(f"Loading precomputed embeddings from {emb_cache}")
        dense.load_embeddings(emb_cache)
    else:
        logger.info(f"Computing embeddings and caching to {emb_cache}...")
        dense.encode_corpus(corpus, save_path=emb_cache)

    dense_run, dense_latency = dense.retrieve_all(queries, top_k=top_k)
    dense_scores = evaluator.evaluate_run(dense_run, model_name="Dense_MiniLM", metrics=metrics)
    dense_scores["latency_ms"] = round(dense_latency, 2)

    # 4. Stage 3: Hybrid Fusion (RRF)
    print("\n>>> [3/4] Hybrid Fusion (Reciprocal Rank Fusion k=60)...")
    hybrid_run = reciprocal_rank_fusion([bm25_run, dense_run], k=60, top_k=top_k)
    hybrid_latency = bm25_latency + dense_latency
    hybrid_scores = evaluator.evaluate_run(hybrid_run, model_name="Hybrid_RRF", metrics=metrics)
    hybrid_scores["latency_ms"] = round(hybrid_latency, 2)

    # 5. Stage 4: Cross-Encoder Re-ranking
    print(f"\n>>> [4/4] Neural Cross-Encoder Re-ranking (Top-{rerank_depth})...")
    reranker = NeuralReRanker(model_name=cross_encoder_name)
    ce_run, ce_latency = reranker.rerank_run(
        initial_run=hybrid_run,
        queries=queries,
        corpus=corpus,
        rerank_depth=rerank_depth,
        top_k=top_k
    )
    total_pipeline_latency = hybrid_latency + ce_latency
    ce_scores = evaluator.evaluate_run(ce_run, model_name="Hybrid_CrossEncoder", metrics=metrics)
    ce_scores["latency_ms"] = round(total_pipeline_latency, 2)

    # 6. Statistical Significance Comparison
    print("\n>>> Running Statistical Significance Testing (Wilcoxon)...")
    runs_to_compare = {
        "BM25": bm25_run,
        "Dense": dense_run,
        "Hybrid_RRF": hybrid_run,
        "Hybrid_CrossEncoder": ce_run
    }
    comparison_table = evaluator.compare_runs(runs_to_compare, metrics=metrics, stat_test="student")

    # 7. Formatted Table
    table_data = [
        [
            "1. BM25 (Lexical Only)",
            f"{bm25_scores['ndcg@10']:.4f}",
            f"{bm25_scores['map']:.4f}",
            f"{bm25_scores['mrr@10']:.4f}",
            f"{bm25_scores['recall@100']:.4f}",
            f"{bm25_scores['latency_ms']:.2f} ms"
        ],
        [
            "2. Dense (all-MiniLM-L6-v2)",
            f"{dense_scores['ndcg@10']:.4f}",
            f"{dense_scores['map']:.4f}",
            f"{dense_scores['mrr@10']:.4f}",
            f"{dense_scores['recall@100']:.4f}",
            f"{dense_scores['latency_ms']:.2f} ms"
        ],
        [
            "3. Hybrid (BM25 + Dense RRF)",
            f"{hybrid_scores['ndcg@10']:.4f}",
            f"{hybrid_scores['map']:.4f}",
            f"{hybrid_scores['mrr@10']:.4f}",
            f"{hybrid_scores['recall@100']:.4f}",
            f"{hybrid_scores['latency_ms']:.2f} ms"
        ],
        [
            "4. Hybrid + Cross-Encoder Re-ranker",
            f"{ce_scores['ndcg@10']:.4f}",
            f"{ce_scores['map']:.4f}",
            f"{ce_scores['mrr@10']:.4f}",
            f"{ce_scores['recall@100']:.4f}",
            f"{ce_scores['latency_ms']:.2f} ms"
        ]
    ]

    headers = ["Pipeline Stage", "nDCG@10", "MAP", "MRR@10", "Recall@100", "Latency"]
    formatted_table = tabulate(table_data, headers=headers, tablefmt="github")

    print("\n" + "=" * 85)
    print(f"  COMPLETE ABLATION STUDY RESULTS [{dataset_name.upper()}]")
    print("=" * 85)
    print(formatted_table)
    print("\n--- Statistical Significance (ranx compare) ---")
    print(comparison_table)
    print("=" * 85)

    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, f"{dataset_name}_full_ablation.json")
    IREvaluator.save_results({
        "bm25": bm25_scores,
        "dense": dense_scores,
        "hybrid_rrf": hybrid_scores,
        "hybrid_cross_encoder": ce_scores
    }, out_json)

    return formatted_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="scifact")
    parser.add_argument("--rerank_depth", type=int, default=50)
    args = parser.parse_args()

    run_full_ablation(dataset_name=args.dataset, rerank_depth=args.rerank_depth)
