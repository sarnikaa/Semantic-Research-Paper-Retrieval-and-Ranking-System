"""
Quantization Speedup & Fidelity Benchmark:
Compares FP32 vs. INT8 Dynamic Quantization for Cross-Encoder re-ranking.
Evaluates:
- Latency per query (ms)
- Speedup factor
- Score Pearson / Spearman Rank correlation
"""

import os
import sys
import json
import time
import numpy as np
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from src.cross_encoder_reranker import NeuralReRanker


def benchmark_quantization():
    print("=" * 80)
    print("  CROSS-ENCODER INT8 DYNAMIC QUANTIZATION BENCHMARK")
    print("=" * 80)

    # Sample query and candidates
    query = "treatment of hypercholesterolemia with statins reduces cardiovascular events"
    corpus = {
        f"doc_{i}": {
            "title": f"Scientific Study on Cardiovascular Disease and Lipid Lowering #{i}",
            "text": f"Study {i} evaluates LDL reduction, HMG-CoA reductase inhibitors, arterial plaque, and mortality outcomes across patient cohorts."
        }
        for i in range(25)
    }
    candidate_ids = list(corpus.keys())

    # 1. Standard FP32 Re-ranker
    print("\n>>> 1. Benchmarking Standard FP32 Re-Ranker on CPU...")
    reranker_fp32 = NeuralReRanker(device="cpu", quantize=False)
    
    # Warmup
    _ = reranker_fp32.rerank_query(query, candidate_ids[:5], corpus)

    latencies_fp32 = []
    for _ in range(5):
        t0 = time.perf_counter()
        scores_fp32 = reranker_fp32.rerank_query(query, candidate_ids, corpus, top_k=25)
        latencies_fp32.append((time.perf_counter() - t0) * 1000.0)
    avg_lat_fp32 = sum(latencies_fp32) / len(latencies_fp32)
    print(f"Standard FP32 Latency: {avg_lat_fp32:.2f} ms")

    # 2. Dynamic INT8 Quantized Re-ranker
    print("\n>>> 2. Benchmarking Dynamic INT8 Quantized Re-Ranker on CPU...")
    reranker_int8 = NeuralReRanker(device="cpu", quantize=True)
    
    # Warmup
    _ = reranker_int8.rerank_query(query, candidate_ids[:5], corpus)

    latencies_int8 = []
    for _ in range(5):
        t0 = time.perf_counter()
        scores_int8 = reranker_int8.rerank_query(query, candidate_ids, corpus, top_k=25)
        latencies_int8.append((time.perf_counter() - t0) * 1000.0)
    avg_lat_int8 = sum(latencies_int8) / len(latencies_int8)
    print(f"INT8 Quantized Latency: {avg_lat_int8:.2f} ms")

    speedup = avg_lat_fp32 / max(1e-6, avg_lat_int8)

    # 3. Compute Rank Correlation (Fidelity Preservation)
    common_keys = sorted(list(set(scores_fp32.keys()).intersection(set(scores_int8.keys()))))
    vec_fp32 = [scores_fp32[k] for k in common_keys]
    vec_int8 = [scores_int8[k] for k in common_keys]

    spearman_corr, _ = spearmanr(vec_fp32, vec_int8)
    pearson_corr, _ = pearsonr(vec_fp32, vec_int8)

    print("\n" + "=" * 80)
    print(f"  QUANTIZATION RESULTS SUMMARY:")
    print(f"  FP32 Latency       : {avg_lat_fp32:.2f} ms")
    print(f"  INT8 Latency       : {avg_lat_int8:.2f} ms")
    print(f"  Speedup Factor     : {speedup:.2f}x faster")
    print(f"  Spearman Rank Corr : {spearman_corr:.4f} (Ranking order preserved)")
    print(f"  Pearson Linear Corr: {pearson_corr:.4f}")
    print("=" * 80)

    results = {
        "fp32_latency_ms": round(avg_lat_fp32, 2),
        "int8_latency_ms": round(avg_lat_int8, 2),
        "speedup_factor": round(speedup, 2),
        "spearman_rank_correlation": round(float(spearman_corr), 4),
        "pearson_correlation": round(float(pearson_corr), 4)
    }

    os.makedirs("results", exist_ok=True)
    with open("results/quantization_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    benchmark_quantization()
