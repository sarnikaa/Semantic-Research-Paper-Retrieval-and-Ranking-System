"""
Phase 1: Baseline Evaluation Script.
Downloads SciFact from BEIR, builds an Okapi BM25 index,
evaluates on test queries using ranx, and outputs standard IR metrics.
"""

import os
import sys
import argparse
import logging

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_beir_dataset
from src.bm25_retriever import BM25Retriever
from src.evaluator import IREvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_bm25")


def run_bm25_baseline(
    dataset_name: str = "scifact",
    data_dir: str = "data",
    results_dir: str = "results",
    top_k: int = 100,
    k1: float = 1.5,
    b: float = 0.75
):
    print("=" * 70)
    print(f"  PHASE 1: OKAPI BM25 BASELINE EVALUATION ({dataset_name.upper()})")
    print("=" * 70)

    # 1. Load dataset
    corpus, queries, qrels = load_beir_dataset(dataset_name=dataset_name, data_dir=data_dir, split="test")

    # 2. Build BM25 index
    retriever = BM25Retriever(k1=k1, b=b)
    retriever.index(corpus)

    # 3. Retrieve candidates
    run, avg_latency_ms = retriever.retrieve_all(queries, top_k=top_k)

    # 4. Evaluate using ranx
    evaluator = IREvaluator(qrels)
    metrics = ["ndcg@10", "map", "mrr@10", "recall@100"]
    scores = evaluator.evaluate_run(run, model_name=f"BM25_{dataset_name}", metrics=metrics)
    scores["avg_latency_ms"] = round(avg_latency_ms, 2)
    scores["corpus_size"] = len(corpus)
    scores["num_queries"] = len(queries)

    # 5. Save results
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, f"{dataset_name}_bm25_baseline.json")
    evaluator.save_results(scores, out_file)

    print("\n" + "=" * 70)
    print(f"  FINAL BM25 BASELINE RESULTS [{dataset_name.upper()}]")
    print("=" * 70)
    for m in metrics:
        print(f"  {m:<14}: {scores[m]:.4f}")
    print(f"  {'Avg Latency':<14}: {scores['avg_latency_ms']:.2f} ms / query")
    print(f"  Results saved to : {out_file}")
    print("=" * 70)

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BM25 baseline on BEIR datasets")
    parser.add_argument("--dataset", type=str, default="scifact", help="Dataset name ('scifact' or 'scidocs')")
    parser.add_argument("--top_k", type=int, default=100, help="Number of documents to retrieve per query")
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter")
    args = parser.parse_args()

    run_bm25_baseline(
        dataset_name=args.dataset,
        top_k=args.top_k,
        k1=args.k1,
        b=args.b
    )
