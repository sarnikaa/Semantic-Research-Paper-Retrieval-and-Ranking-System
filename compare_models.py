"""
Multi-Model Benchmark Comparison:
Compares:
1. all-MiniLM-L6-v2 (Lightweight standard dense retriever)
2. BAAI/bge-small-en-v1.5 (Top-performing MTEB dense retriever)
Evaluates retrieval accuracy, latency, and rank fusion performance.
"""

import os
import sys
import json
import logging
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from src.data_loader import load_beir_dataset
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.hybrid_fusion import reciprocal_rank_fusion
from src.evaluator import IREvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_model_comparison(dataset_name: str = "scifact", results_dir: str = "results"):
    print("=" * 85)
    print("  MULTI-MODEL COMPARISON: all-MiniLM-L6-v2 vs BAAI/bge-small-en-v1.5")
    print("=" * 85)

    os.makedirs(results_dir, exist_ok=True)
    corpus, queries, qrels = load_beir_dataset(dataset_name, split="test")
    evaluator = IREvaluator(qrels)
    metrics = ["ndcg@10", "map", "mrr@10", "recall@100"]

    # 1. BM25 Baseline
    bm25 = BM25Retriever()
    bm25.index(corpus)
    bm25_run, bm25_lat = bm25.retrieve_all(queries, top_k=100)
    bm25_scores = evaluator.evaluate_run(bm25_run, model_name="BM25", metrics=metrics)

    # 2. Model A: all-MiniLM-L6-v2
    print("\n>>> Evaluating Dense Model A: all-MiniLM-L6-v2...")
    dense_a = DenseRetriever("sentence-transformers/all-MiniLM-L6-v2")
    cache_a = os.path.join("data", f"{dataset_name}_embeddings_minilm.npy")
    if os.path.exists(cache_a):
        dense_a.load_embeddings(cache_a)
    else:
        dense_a.encode_corpus(corpus, save_path=cache_a)
    run_a, lat_a = dense_a.retrieve_all(queries, top_k=100)
    scores_a = evaluator.evaluate_run(run_a, model_name="all-MiniLM-L6-v2", metrics=metrics)

    # Hybrid with Model A
    hybrid_a = reciprocal_rank_fusion([bm25_run, run_a], k=10, top_k=100)
    scores_hyb_a = evaluator.evaluate_run(hybrid_a, model_name="Hybrid_MiniLM", metrics=metrics)

    # 3. Model B: BAAI/bge-small-en-v1.5
    print("\n>>> Evaluating Dense Model B: BAAI/bge-small-en-v1.5...")
    dense_b = DenseRetriever("BAAI/bge-small-en-v1.5")
    cache_b = os.path.join("data", f"{dataset_name}_embeddings_bge_small.npy")
    if os.path.exists(cache_b):
        dense_b.load_embeddings(cache_b)
    else:
        dense_b.encode_corpus(corpus, save_path=cache_b)
    run_b, lat_b = dense_b.retrieve_all(queries, top_k=100)
    scores_b = evaluator.evaluate_run(run_b, model_name="bge-small-en-v1.5", metrics=metrics)

    # Hybrid with Model B
    hybrid_b = reciprocal_rank_fusion([bm25_run, run_b], k=10, top_k=100)
    scores_hyb_b = evaluator.evaluate_run(hybrid_b, model_name="Hybrid_BGE", metrics=metrics)

    # Statistical Significance
    comparison = evaluator.compare_runs(
        {
            "BM25": bm25_run,
            "Dense_MiniLM": run_a,
            "Hybrid_MiniLM": hybrid_a,
            "Dense_BGE": run_b,
            "Hybrid_BGE": hybrid_b
        },
        metrics=metrics,
        stat_test="student"
    )

    table_data = [
        ["BM25 (Sparse Baseline)", f"{bm25_scores['ndcg@10']:.4f}", f"{bm25_scores['map']:.4f}", f"{bm25_scores['mrr@10']:.4f}", f"{bm25_scores['recall@100']:.4f}", f"{bm25_lat:.2f} ms"],
        ["Dense: all-MiniLM-L6-v2", f"{scores_a['ndcg@10']:.4f}", f"{scores_a['map']:.4f}", f"{scores_a['mrr@10']:.4f}", f"{scores_a['recall@100']:.4f}", f"{lat_a:.2f} ms"],
        ["Hybrid: BM25 + all-MiniLM", f"{scores_hyb_a['ndcg@10']:.4f}", f"{scores_hyb_a['map']:.4f}", f"{scores_hyb_a['mrr@10']:.4f}", f"{scores_hyb_a['recall@100']:.4f}", f"{bm25_lat+lat_a:.2f} ms"],
        ["Dense: BAAI/bge-small-en-v1.5", f"{scores_b['ndcg@10']:.4f}", f"{scores_b['map']:.4f}", f"{scores_b['mrr@10']:.4f}", f"{scores_b['recall@100']:.4f}", f"{lat_b:.2f} ms"],
        ["Hybrid: BM25 + BGE-small", f"{scores_hyb_b['ndcg@10']:.4f}", f"{scores_hyb_b['map']:.4f}", f"{scores_hyb_b['mrr@10']:.4f}", f"{scores_hyb_b['recall@100']:.4f}", f"{bm25_lat+lat_b:.2f} ms"],
    ]

    headers = ["Model / Configuration", "nDCG@10", "MAP", "MRR@10", "Recall@100", "Latency"]
    formatted = tabulate(table_data, headers=headers, tablefmt="github")

    print("\n" + "=" * 85)
    print("  MODEL COMPARISON BENCHMARK TABLE")
    print("=" * 85)
    print(formatted)
    print("\n--- Significance Report ---")
    print(comparison)
    print("=" * 85)

    out_file = os.path.join(results_dir, "model_comparison.json")
    with open(out_file, "w") as f:
        json.dump({
            "bm25": bm25_scores,
            "minilm_dense": scores_a,
            "minilm_hybrid": scores_hyb_a,
            "bge_dense": scores_b,
            "bge_hybrid": scores_hyb_b
        }, f, indent=2)

    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    run_model_comparison()
