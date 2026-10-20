"""
Standard IR Evaluation Harness using ranx.
Computes nDCG@10, MAP, MRR, Recall@100, latency statistics,
and supports statistical significance testing between models.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from ranx import Qrels, Run, evaluate, compare

logger = logging.getLogger(__name__)

DEFAULT_METRICS = ["ndcg@10", "map", "mrr@10", "recall@100"]


class IREvaluator:
    """Evaluates IR runs against ground-truth qrels with standardized metrics."""

    def __init__(self, qrels_dict: Dict[str, Dict[str, int]]):
        """
        Args:
            qrels_dict: Dict of {query_id: {doc_id: relevance_score}}
        """
        self.raw_qrels = qrels_dict
        self.qrels = Qrels(qrels_dict)

    def evaluate_run(
        self,
        run_dict: Dict[str, Dict[str, float]],
        model_name: str = "model",
        metrics: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Evaluates a model run against the qrels.

        Args:
            run_dict: Dict of {query_id: {doc_id: score}}
            model_name: Name of the model/pipeline evaluated
            metrics: List of metrics to compute (default: ndcg@10, map, mrr@10, recall@100)

        Returns:
            Dict of {metric_name: score}
        """
        if metrics is None:
            metrics = DEFAULT_METRICS

        run = Run(run_dict, name=model_name)
        results = evaluate(self.qrels, run, metrics=metrics, return_mean=True)

        logger.info(f"--- Evaluation Results for [{model_name}] ---")
        for metric, score in results.items():
            logger.info(f"  {metric:<12}: {score:.4f}")

        return results

    def compare_runs(
        self,
        runs_dict: Dict[str, Dict[str, Dict[str, float]]],
        metrics: Optional[list] = None,
        stat_test: str = "student"
    ) -> Any:
        """
        Runs statistical significance comparisons across multiple models.

        Args:
            runs_dict: Dict of {model_name: run_dict}
            metrics: Metrics to compare
            stat_test: Statistical test to use ('wilcoxon' or 'student')
        """
        if metrics is None:
            metrics = DEFAULT_METRICS

        ranx_runs = [Run(r, name=name) for name, r in runs_dict.items()]
        comparison = compare(
            qrels=self.qrels,
            runs=ranx_runs,
            metrics=metrics,
            stat_test=stat_test
        )
        return comparison

    @staticmethod
    def save_results(
        results: Dict[str, Any],
        output_path: str
    ) -> None:
        """Saves evaluation results to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")
