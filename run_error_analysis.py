"""
Automated Error Analysis & Case Study Extraction Script.
Finds:
1. 'Semantic Wins': Queries where Dense/Hybrid excels over BM25 (Vocabulary Mismatch cases).
2. 'Lexical Wins': Queries where BM25 excels over Dense (Exact Acronym/Nomenclature cases).

Outputs structured case studies and saves 'results/error_analysis_cases.json'.
"""

import os
import sys
import json
import logging
from ranx import Qrels, Run, evaluate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from src.data_loader import load_beir_dataset
from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.hybrid_fusion import reciprocal_rank_fusion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_error_analysis(dataset_name: str = "scifact", results_dir: str = "results"):
    print("=" * 80)
    print("  QUALITATIVE ERROR ANALYSIS & CASE STUDY EXTRACTION")
    print("=" * 80)

    corpus, queries, qrels_dict = load_beir_dataset(dataset_name, split="test")
    qrels = Qrels(qrels_dict)

    # 1. Run BM25
    bm25 = BM25Retriever()
    bm25.index(corpus)
    bm25_run, _ = bm25.retrieve_all(queries, top_k=50)

    # 2. Run Dense
    dense = DenseRetriever()
    dense.load_embeddings(os.path.join("data", f"{dataset_name}_embeddings_minilm.npy"))
    dense_run, _ = dense.retrieve_all(queries, top_k=50)

    # 3. Hybrid Run
    hybrid_run = reciprocal_rank_fusion([bm25_run, dense_run], k=60, top_k=50)

    # 4. Compute per-query nDCG@10
    q_ids = list(queries.keys())
    bm25_scores_arr = evaluate(qrels, Run(bm25_run), metrics=["ndcg@10"], return_mean=False)
    dense_scores_arr = evaluate(qrels, Run(dense_run), metrics=["ndcg@10"], return_mean=False)
    hybrid_scores_arr = evaluate(qrels, Run(hybrid_run), metrics=["ndcg@10"], return_mean=False)

    # Delta analysis
    deltas = []
    for idx, q_id in enumerate(q_ids):
        q_text = queries[q_id]
        s_bm25 = float(bm25_scores_arr[idx])
        s_dense = float(dense_scores_arr[idx])
        s_hybrid = float(hybrid_scores_arr[idx])

        # Relevant doc titles
        rel_doc_ids = list(qrels_dict.get(q_id, {}).keys())
        rel_titles = [corpus[d]["title"] for d in rel_doc_ids if d in corpus]

        deltas.append({
            "query_id": q_id,
            "query_text": q_text,
            "bm25_ndcg": round(s_bm25, 4),
            "dense_ndcg": round(s_dense, 4),
            "hybrid_ndcg": round(s_hybrid, 4),
            "diff_hybrid_bm25": round(s_hybrid - s_bm25, 4),
            "diff_bm25_dense": round(s_bm25 - s_dense, 4),
            "relevant_titles": rel_titles
        })

    # Sort for Semantic Wins (Hybrid/Dense >> BM25)
    semantic_wins = sorted(deltas, key=lambda x: x["diff_hybrid_bm25"], reverse=True)[:3]

    # Sort for Lexical Wins (BM25 >> Dense)
    lexical_wins = sorted(deltas, key=lambda x: x["diff_bm25_dense"], reverse=True)[:2]

    print("\n" + "=" * 80)
    print("  CASE STUDY A: SEMANTIC WINS (Hybrid & Dense >> BM25)")
    print("  Cause: Vocabulary Mismatch / Synonymous Paraphrasing")
    print("=" * 80)
    for i, case in enumerate(semantic_wins, 1):
        print(f"\n[{i}] Query: \"{case['query_text']}\"")
        print(f"    BM25 nDCG@10  : {case['bm25_ndcg']:.4f}")
        print(f"    Dense nDCG@10 : {case['dense_ndcg']:.4f}")
        print(f"    Hybrid nDCG@10: {case['hybrid_ndcg']:.4f}  (Gain: +{case['diff_hybrid_bm25']:.4f})")
        if case['relevant_titles']:
            print(f"    Target Paper  : \"{case['relevant_titles'][0]}\"")

    print("\n" + "=" * 80)
    print("  CASE STUDY B: LEXICAL WINS (BM25 >> Dense)")
    print("  Cause: Exact Acronyms, Specific Genes, or Model Names")
    print("=" * 80)
    for i, case in enumerate(lexical_wins, 1):
        print(f"\n[{i}] Query: \"{case['query_text']}\"")
        print(f"    BM25 nDCG@10  : {case['bm25_ndcg']:.4f}")
        print(f"    Dense nDCG@10 : {case['dense_ndcg']:.4f}")
        print(f"    Hybrid nDCG@10: {case['hybrid_ndcg']:.4f}  (BM25 Advantage: +{case['diff_bm25_dense']:.4f})")
        if case['relevant_titles']:
            print(f"    Target Paper  : \"{case['relevant_titles'][0]}\"")

    # Save to JSON
    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "error_analysis_cases.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "semantic_wins": semantic_wins,
            "lexical_wins": lexical_wins
        }, f, indent=2)

    print(f"\n[SUCCESS] Error analysis cases saved to: {out_file}")


if __name__ == "__main__":
    run_error_analysis()
