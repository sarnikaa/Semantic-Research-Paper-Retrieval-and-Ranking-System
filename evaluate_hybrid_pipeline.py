"""
Phase 2 & 3: Dense Retrieval & Hybrid RRF Fusion Evaluation Script.
Evaluates BM25 vs Dense (all-MiniLM-L6-v2) vs Hybrid (RRF),
generates the comparative ablation table, and runs statistical significance tests.
"""

import os
import sys
import argparse
import logging
from tabulate import tabulate

# Ensure project root is on PYTHONPATH and UTF-8 console output
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data_loader import load_beir_dataset
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.hybrid_fusion import reciprocal_rank_fusion
from src.evaluator import IREvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_hybrid")


def run_hybrid_evaluation(
    dataset_name: str = "scifact",
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 100,
    rrf_k: int = 60,
    data_dir: str = "data",
    results_dir: str = "results"
):
    print("=" * 80)
    print(f"  RUNNING ABLATION EVALUATION: BM25 vs DENSE vs HYBRID ({dataset_name.upper()})")
    print("=" * 80)

    # 1. Load Dataset
    corpus, queries, qrels = load_beir_dataset(dataset_name=dataset_name, data_dir=data_dir, split="test")
    evaluator = IREvaluator(qrels)
    metrics = ["ndcg@10", "map", "mrr@10", "recall@100"]

    # 2. Stage 1: BM25 Baseline
    print("\n>>> [1/3] Running Sparse BM25 Retrieval...")
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.index(corpus)
    bm25_run, bm25_latency = bm25.retrieve_all(queries, top_k=top_k)
    bm25_scores = evaluator.evaluate_run(bm25_run, model_name="BM25", metrics=metrics)
    bm25_scores["latency_ms"] = round(bm25_latency, 2)

    # 3. Stage 2: Dense Retrieval
    print("\n>>> [2/3] Running Dense Vector Retrieval...")
    emb_cache = os.path.join(data_dir, f"{dataset_name}_embeddings_minilm.npy")
    dense = DenseRetriever(model_name=dense_model_name)
    if os.path.exists(emb_cache):
        logger.info(f"Loading cached embeddings from {emb_cache}")
        dense.load_embeddings(emb_cache)
    else:
        logger.info(f"Computing embeddings and caching to {emb_cache}...")
        dense.encode_corpus(corpus, save_path=emb_cache)

    dense_run, dense_latency = dense.retrieve_all(queries, top_k=top_k)
    dense_scores = evaluator.evaluate_run(dense_run, model_name="Dense_MiniLM", metrics=metrics)
    dense_scores["latency_ms"] = round(dense_latency, 2)

    # 4. Stage 3: Hybrid Fusion (RRF)
    print("\n>>> [3/3] Running Reciprocal Rank Fusion (RRF)...")
    hybrid_run = reciprocal_rank_fusion([bm25_run, dense_run], k=rrf_k, top_k=top_k)
    hybrid_latency = bm25_latency + dense_latency  # combined retrieval latency
    hybrid_scores = evaluator.evaluate_run(hybrid_run, model_name="Hybrid_RRF", metrics=metrics)
    hybrid_scores["latency_ms"] = round(hybrid_latency, 2)

    # 5. Statistical Significance Comparison using ranx
    print("\n>>> Running Statistical Significance Comparison...")
    runs_to_compare = {
        "BM25": bm25_run,
        "Dense": dense_run,
        "Hybrid_RRF": hybrid_run
    }
    comparison_table = evaluator.compare_runs(runs_to_compare, metrics=metrics, stat_test="student")

    # 6. Build Consolidated Ablation Table
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
        ]
    ]

    headers = ["Pipeline Stage", "nDCG@10", "MAP", "MRR@10", "Recall@100", "Latency"]
    formatted_table = tabulate(table_data, headers=headers, tablefmt="github")

    # 7. Save ablation to JSON
    os.makedirs(results_dir, exist_ok=True)
    summary_file = os.path.join(results_dir, f"{dataset_name}_ablation_summary.json")
    IREvaluator.save_results({
        "dataset": dataset_name,
        "bm25": bm25_scores,
        "dense": dense_scores,
        "hybrid_rrf": hybrid_scores
    }, summary_file)

    print("\n" + "=" * 80)
    print(f"  FINAL ABLATION TABLE [{dataset_name.upper()}]")
    print("=" * 80)
    print(formatted_table)
    print("\n--- Statistical Significance (ranx report) ---")
    print(comparison_table)
    print("=" * 80)

    return formatted_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BM25 vs Dense vs Hybrid")
    parser.add_argument("--dataset", type=str, default="scifact")
    parser.add_argument("--dense_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--rrf_k", type=int, default=60)
    args = parser.parse_args()

    run_hybrid_evaluation(
        dataset_name=args.dataset,
        dense_model_name=args.dense_model,
        top_k=args.top_k,
        rrf_k=args.rrf_k
    )
