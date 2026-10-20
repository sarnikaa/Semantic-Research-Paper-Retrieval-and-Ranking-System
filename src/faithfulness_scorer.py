"""
Automated Factual Faithfulness and Hallucination Scorer.
Quantitatively assesses factual grounding of LLM-generated summaries and rationales
against source scientific paper abstracts.

Metrics:
1. Named Entity & Numerical Fact Containment (% of entities in summary grounded in source).
2. Unsupported Fact Rate (% of novel entities/numbers not present in source -> hallucination metric).
3. Token Overlap Precision, Recall, and F1.
4. Composite Grounding Score [0.0 - 1.0].
"""

import re
import string
from typing import Dict, List, Set, Any


def extract_key_facts(text: str) -> Set[str]:
    """
    Extracts key numerical expressions, capitalized multi-word scientific terms,
    and acronyms from text.
    """
    # 1. Numbers, percentages, p-values, metrics (e.g. 95%, 0.6989, p < 0.01)
    numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
    
    # 2. Acronyms (e.g. BM25, RRF, FAISS, MAP, MRR, LLM, BERT)
    acronyms = set(re.findall(r"\b[A-Z0-9]{2,}\b", text))
    
    # 3. Capitalized scientific terms
    words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    capitalized_terms = {w.lower() for w in words if len(w) > 3}
    
    # 4. Normalized tokens
    clean_text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = {w for w in clean_text.split() if len(w) > 3}
    
    return numbers.union(acronyms).union(capitalized_terms).union(tokens)


def compute_faithfulness_metrics(source_abstract: str, generated_summary: str) -> Dict[str, Any]:
    """
    Computes automated quantitative faithfulness metrics between source abstract and LLM summary.

    Returns:
        Dict containing containment, hallucination rate, token overlap, and composite grounding score.
    """
    if not generated_summary.strip():
        return {
            "grounding_score": 0.0,
            "entity_containment": 0.0,
            "hallucination_rate": 0.0,
            "token_precision": 0.0,
            "token_recall": 0.0,
            "token_f1": 0.0,
            "unsupported_facts": [],
            "verdict": "FAIL - Empty Summary"
        }

    source_facts = extract_key_facts(source_abstract)
    summary_facts = extract_key_facts(generated_summary)

    # Clean words for token metrics
    clean_source = set(source_abstract.lower().translate(str.maketrans("", "", string.punctuation)).split())
    clean_summary = set(generated_summary.lower().translate(str.maketrans("", "", string.punctuation)).split())

    # Stopwords to filter out trivial matches
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "of", "from", "as", "is", "are", "was", "were", "this", "that", "these",
        "those", "paper", "presents", "we", "study", "results", "show", "our", "which"
    }
    meaningful_summary_facts = {f for f in summary_facts if f not in stopwords and len(f) > 2}
    meaningful_source_facts = {f for f in source_facts if f not in stopwords and len(f) > 2}

    if not meaningful_summary_facts:
        grounded_facts = set()
        unsupported_facts = set()
        containment = 1.0
        hallucination_rate = 0.0
    else:
        grounded_facts = meaningful_summary_facts.intersection(meaningful_source_facts)
        unsupported_facts = meaningful_summary_facts.difference(meaningful_source_facts)
        containment = len(grounded_facts) / len(meaningful_summary_facts)
        hallucination_rate = len(unsupported_facts) / len(meaningful_summary_facts)

    # Token overlap precision and recall
    overlap = clean_summary.intersection(clean_source) - stopwords
    clean_summary_filtered = clean_summary - stopwords
    clean_source_filtered = clean_source - stopwords

    precision = len(overlap) / max(1, len(clean_summary_filtered))
    recall = len(overlap) / max(1, len(clean_source_filtered))
    f1 = (2 * precision * recall) / max(1e-6, (precision + recall))

    # Composite Grounding Score (weighted harmonic blend)
    composite = round((0.6 * containment) + (0.4 * precision), 4)

    if composite >= 0.75 and hallucination_rate < 0.20:
        verdict = "FAITHFUL (High Grounding)"
    elif composite >= 0.50:
        verdict = "ACCEPTABLE (Moderate Grounding)"
    else:
        verdict = "LOW GROUNDING (Potential Hallucination)"

    return {
        "grounding_score": composite,
        "entity_containment": round(containment, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "token_precision": round(precision, 4),
        "token_recall": round(recall, 4),
        "token_f1": round(f1, 4),
        "supported_facts_count": len(grounded_facts),
        "unsupported_facts_count": len(unsupported_facts),
        "sample_unsupported_facts": sorted(list(unsupported_facts))[:5],
        "verdict": verdict
    }
