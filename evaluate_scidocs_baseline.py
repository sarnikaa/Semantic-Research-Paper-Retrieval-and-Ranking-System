"""
SCIDOCS Benchmark Lexical Baseline Evaluation Script.
Evaluates Okapi BM25 on the SCIDOCS dataset (BEIR benchmark)
and saves results to 'results/scidocs_bm25_baseline.json'.
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from src.data_loader import load_beir_dataset
from src.bm25_retriever import BM25Retriever
from src.evaluator import IREvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def evaluate_scidocs(results_dir: str = "results", top_k: int = 100):
    print("=" * 80)
    print("  SCIDOCS BENCHMARK EVALUATION (OKAPI BM25 BASELINE)")
    print("=" * 80)

    corpus, queries, qrels = load_beir_dataset("scidocs", split="test")
    evaluator = IREvaluator(qrels)
    metrics = ["ndcg@10", "map", "mrr@10", "recall@100"]

    bm25 = BM25Retriever()
    bm25.index(corpus)
    run, avg_latency_ms = bm25.retrieve_all(queries, top_k=top_k)

    scores = evaluator.evaluate_run(run, model_name="BM25_scidocs", metrics=metrics)
    scores["avg_latency_ms"] = round(avg_latency_ms, 2)
    scores["corpus_size"] = len(corpus)
    scores["num_queries"] = len(queries)

    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "scidocs_bm25_baseline.json")
    with open(out_file, "w") as f:
        json.dump(scores, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  SCIDOCS BM25 BASELINE RESULTS:")
    print(f"  nDCG@10    : {scores['ndcg@10']:.4f}")
    print(f"  MAP        : {scores['map']:.4f}")
    print(f"  MRR@10     : {scores['mrr@10']:.4f}")
    print(f"  Recall@100 : {scores['recall@100']:.4f}")
    print(f"  Latency    : {scores['avg_latency_ms']:.2f} ms")
    print(f"  Saved to   : {out_file}")
    print("=" * 80)
    return scores


if __name__ == "__main__":
    evaluate_scidocs()
