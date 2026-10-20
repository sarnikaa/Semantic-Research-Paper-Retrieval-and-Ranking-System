"""
Unit & Integration Tests for the Semantic Paper Retrieval System.
"""

import pytest
from src.bm25_retriever import BM25Retriever
from src.hybrid_fusion import reciprocal_rank_fusion, min_max_normalize


def test_bm25_retrieval():
    corpus = {
        "doc1": {"title": "Attention Is All You Need", "text": "We introduce the Transformer architecture based on attention."},
        "doc2": {"title": "Convolutional Neural Networks", "text": "Deep CNN models for computer vision and image classification."},
        "doc3": {"title": "BERT Pretraining", "text": "Bidirectional encoder representations from transformers."}
    }
    retriever = BM25Retriever()
    retriever.index(corpus)
    
    results = retriever.search("transformer attention", top_k=2)
    assert len(results) > 0
    top_doc = list(results.keys())[0]
    assert top_doc in ["doc1", "doc3"]


def test_rrf_fusion():
    run_bm25 = {
        "q1": {"doc1": 10.5, "doc2": 8.2, "doc3": 1.0}
    }
    run_dense = {
        "q1": {"doc3": 0.88, "doc1": 0.75, "doc2": 0.40}
    }

    fused = reciprocal_rank_fusion([run_bm25, run_dense], k=60, top_k=3)
    assert "q1" in fused
    assert len(fused["q1"]) == 3
    # doc1 is rank 1 in BM25 and rank 2 in Dense -> should have highest or second highest combined score
    scores = fused["q1"]
    assert scores["doc1"] > scores["doc2"]


def test_score_normalization():
    raw_scores = {"docA": 20.0, "docB": 10.0, "docC": 0.0}
    norm_scores = min_max_normalize(raw_scores)
    assert norm_scores["docA"] == 1.0
    assert norm_scores["docC"] == 0.0
    assert norm_scores["docB"] == 0.5


def test_bibtex_formatting():
    from src.bibtex_formatter import format_bibtex
    doc = {"title": "Attention Is All You Need", "year": "2017", "authors": ["Vaswani", "Shazeer"]}
    bib = format_bibtex("12345", doc)
    assert "@article{paper_12345," in bib
    assert "Attention Is All You Need" in bib
    assert "Vaswani and Shazeer" in bib


def test_pseudo_relevance_feedback():
    from src.query_expansion import pseudo_relevance_feedback
    corpus = {
        "d1": {"title": "Colorectal tumors", "text": "Dietary fiber and gut microbiome impact colon carcinogenesis."}
    }
    expanded = pseudo_relevance_feedback("colon cancer", ["d1"], corpus, expansion_terms=2)
    assert len(expanded.split()) > len("colon cancer".split())


def test_dense_retriever_with_faiss():
    from src.dense_retriever import DenseRetriever
    dense = DenseRetriever(use_faiss=True)
    corpus = {
        "p1": {"title": "Deep Neural Networks", "text": "Supervised deep learning for classification."},
        "p2": {"title": "Cardiovascular Disease", "text": "Heart attacks and arterial hypertension."}
    }
    dense.encode_corpus(corpus)
    res = dense.search("neural network learning", top_k=1)
    assert len(res) == 1
    assert "p1" in res


def test_arxiv_subset_loading():
    from src.data_loader import load_arxiv_subset
    papers = load_arxiv_subset(categories=("cs.IR",), max_papers=5)
    assert len(papers) > 0
    first_paper = list(papers.values())[0]
    assert "title" in first_paper
    assert "text" in first_paper
    assert "categories" in first_paper


def test_semantic_scholar_citation_fetch():
    from src.authority_scorer import fetch_semantic_scholar_citations
    res = fetch_semantic_scholar_citations("Attention Is All You Need", timeout=2)
    assert isinstance(res, dict)
    assert "citation_count" in res
    assert "reference_count" in res


def test_automated_faithfulness_scorer():
    from src.faithfulness_scorer import compute_faithfulness_metrics
    source = "Simvastatin lowered LDL cholesterol by 35% in coronary disease patients."
    good_summary = "Simvastatin reduced LDL cholesterol by 35% in patients with coronary disease."
    bad_summary = "Aspirin cured leukemia in 10000 patients with a 90% success rate."
    
    m_good = compute_faithfulness_metrics(source, good_summary)
    m_bad = compute_faithfulness_metrics(source, bad_summary)

    assert m_good["grounding_score"] > m_bad["grounding_score"]
    assert m_bad["hallucination_rate"] > m_good["hallucination_rate"]


def test_multi_format_citations():
    from src.bibtex_formatter import format_all_citations
    doc = {
        "title": "Language Models are Few-Shot Learners",
        "authors": ["Tom Brown", "Benjamin Mann"],
        "year": "2020"
    }
    cites = format_all_citations("2005.14165", doc)
    assert "BibTeX" in cites and "@article{" in cites["BibTeX"]
    assert "APA 7th" in cites and "(2020)" in cites["APA 7th"] and "Language Models" in cites["APA 7th"]
    assert "IEEE" in cites and "Tom Brown" in cites["IEEE"]
    assert "RIS" in cites and "TY  - JOUR" in cites["RIS"]


def test_late_interaction_maxsim():
    from src.late_interaction_retriever import LateInteractionMaxSimRetriever
    retriever = LateInteractionMaxSimRetriever()
    corpus = {
        "docA": {"title": "Transformer Attention", "text": "Self-attention layers in transformer models."},
        "docB": {"title": "Cardiovascular Health", "text": "Heart disease risk and blood pressure."}
    }
    retriever.index_documents(corpus, max_docs=2)
    results = retriever.search("transformer attention", top_k=1)
    assert len(results) == 1
    assert "docA" in results


def test_visualizer_charts():
    import numpy as np
    from src.visualizer import (
        plot_semantic_embedding_space,
        plot_multi_signal_radar,
        plot_rank_transition_flow,
        plot_interactive_pareto_frontier
    )
    # 1. Radar chart
    signals = {"BM25": 0.8, "Dense": 0.9, "Cross-Encoder": 0.95, "PageRank": 0.4, "Recency": 0.7}
    fig_radar = plot_multi_signal_radar("Sample Paper", signals)
    assert fig_radar is not None

    # 2. PCA space
    q_vec = np.random.randn(384)
    doc_vecs = np.random.randn(5, 384)
    doc_ids = [f"d_{i}" for i in range(5)]
    corpus = {d: {"title": f"Paper {d}"} for d in doc_ids}
    fig_pca = plot_semantic_embedding_space("test query", q_vec, doc_vecs, doc_ids, corpus)
    assert fig_pca is not None

    # 3. Bump chart
    stage_runs = {
        "BM25": {"d_0": 10.0, "d_1": 5.0},
        "Dense": {"d_1": 0.9, "d_0": 0.5}
    }
    fig_bump = plot_rank_transition_flow(["d_0", "d_1"], stage_runs, corpus)
    assert fig_bump is not None

    # 4. Pareto frontier
    fig_pareto = plot_interactive_pareto_frontier({})
    assert fig_pareto is not None


