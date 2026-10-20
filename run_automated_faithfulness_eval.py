"""
Automated Faithfulness Evaluation Benchmark:
Runs automated hallucination and fact containment evaluation across scientific abstracts
and generated summaries, outputting quantitative metrics and validation verdicts.
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.faithfulness_scorer import compute_faithfulness_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_benchmark():
    print("=" * 80)
    print("  AUTOMATED FAITHFULNESS & HALLUCINATION BENCHMARK EVALUATION")
    print("=" * 80)

    # Test cases: Grounded summaries vs Hallucinatory synthetic summaries
    test_cases = [
        {
            "id": "scifact_grounded_1",
            "type": "Grounded LLM Summary",
            "source": (
                "Hypercholesterolemia is a major risk factor for cardiovascular disease. "
                "In a randomized trial of 4,444 patients with coronary heart disease, simvastatin "
                "reduced LDL cholesterol by 35% and total mortality by 30% (p < 0.00001)."
            ),
            "summary": (
                "Simvastatin treatment lowered LDL cholesterol by 35% and overall mortality by 30% "
                "in a clinical trial of 4,444 coronary heart disease patients, counteracting hypercholesterolemia."
            )
        },
        {
            "id": "scifact_grounded_2",
            "type": "Grounded LLM Summary",
            "source": (
                "CRISPR-Cas9 enables targeted genome editing in eukaryotic cells. "
                "Using human codon-optimized Cas9 and single guide RNAs (sgRNAs), we achieved "
                "endogenous gene disruption frequencies of up to 25% in human 293T cells."
            ),
            "summary": (
                "The paper demonstrates CRISPR-Cas9 genome editing using single guide RNAs (sgRNAs) "
                "and optimized Cas9, reaching disruption frequencies of 25% in human 293T cells."
            )
        },
        {
            "id": "synthetic_hallucinated_1",
            "type": "Adversarial Hallucinated Summary",
            "source": (
                "Hypercholesterolemia is a major risk factor for cardiovascular disease. "
                "In a randomized trial of 4,444 patients with coronary heart disease, simvastatin "
                "reduced LDL cholesterol by 35% and total mortality by 30% (p < 0.00001)."
            ),
            "summary": (
                "Aspirin was administered to 10,000 cancer patients in Boston, curing leukemia "
                "with an 85% success rate and eliminating tumor necrosis factor."
            )
        }
    ]

    results = []
    for tc in test_cases:
        metrics = compute_faithfulness_metrics(tc["source"], tc["summary"])
        res_entry = {
            "id": tc["id"],
            "type": tc["type"],
            "metrics": metrics
        }
        results.append(res_entry)

        print(f"\n--- Case [{tc['id']}] ({tc['type']}) ---")
        print(f"Grounding Score     : {metrics['grounding_score']:.4f}")
        print(f"Entity Containment  : {metrics['entity_containment'] * 100:.1f}%")
        print(f"Hallucination Rate  : {metrics['hallucination_rate'] * 100:.1f}%")
        print(f"Token F1 Overlap    : {metrics['token_f1']:.4f}")
        print(f"Verdict             : {metrics['verdict']}")
        if metrics["sample_unsupported_facts"]:
            print(f"Unsupported Entities: {metrics['sample_unsupported_facts']}")

    # Save to results
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "automated_faithfulness_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ Automated faithfulness benchmark complete! Saved to {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
