"""
Query Expansion & Literature Synthesis Module.
Implements:
1. Classical Pseudo-Relevance Feedback (PRF / Rocchio term expansion) for BM25
2. HyDE (Hypothetical Document Embeddings via Ollama)
3. Multi-Paper Literature Synthesis & Comparative Overview
"""

import os
import re
from typing import Dict, List, Optional, Any
from collections import Counter
import requests

from src.data_loader import get_document_text

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Standard English IR stopwords
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "study", "paper", "results", "using", "used", "shows", "showed", "demonstrate"
}


def pseudo_relevance_feedback(
    query: str,
    top_doc_ids: List[str],
    corpus: Dict[str, Dict[str, str]],
    expansion_terms: int = 4
) -> str:
    """
    Expands the original query with salient terms from top pseudo-relevant documents.

    Args:
        query: Original user query
        top_doc_ids: Top retrieved document IDs from first-pass retrieval
        corpus: Document collection
        expansion_terms: Number of expansion terms to add

    Returns:
        Expanded query string
    """
    query_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
    candidate_tokens = []

    for doc_id in top_doc_ids:
        if doc_id not in corpus:
            continue
        text = get_document_text(corpus[doc_id])
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        for token in tokens:
            if token not in STOPWORDS and token not in query_tokens:
                candidate_tokens.append(token)

    if not candidate_tokens:
        return query

    most_common = [term for term, _ in Counter(candidate_tokens).most_common(expansion_terms)]
    expanded_query = f"{query} {' '.join(most_common)}"
    return expanded_query


def generate_hyde_document(query: str, model: str = "llama3.2:3b") -> str:
    """
    Hypothetical Document Embeddings (HyDE).
    Generates a synthetic scientific abstract answering the query.
    """
    prompt = (
        f"You are an expert scientific researcher.\n"
        f"Write a short, hypothetical academic abstract (4-5 sentences) that directly answers "
        f"or addresses the following hypothesis/query:\n"
        f"Query: \"{query}\"\n\n"
        f"Write only the abstract text with objective scientific terminology:"
    )
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        }
        res = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=45)
        if res.status_code == 200:
            return res.json().get("response", "").strip()
        return query
    except Exception:
        return query


def synthesize_literature_overview(
    query: str,
    top_papers: List[Dict[str, str]],
    model: str = "llama3.2:3b"
) -> str:
    """
    Multi-paper comparative synthesis (mini-RAG).
    Compares and contrasts the top 3 retrieved papers regarding the query.
    """
    papers_context = ""
    for idx, paper in enumerate(top_papers, start=1):
        papers_context += (
            f"\n[Paper {idx}]: \"{paper.get('title', 'Untitled')}\"\n"
            f"Abstract: {paper.get('text', '')}\n"
        )

    prompt = (
        f"You are an objective scientific literature synthesis assistant.\n"
        f"User Query / Research Topic: \"{query}\"\n\n"
        f"Synthesize the following 3 retrieved papers:\n{papers_context}\n\n"
        f"In 3 structured sections:\n"
        f"1. Executive Summary (1-2 sentences on what the collective literature says)\n"
        f"2. Points of Agreement / Shared Insights (citing [Paper 1], [Paper 2], etc.)\n"
        f"3. Key Distinctions or Disagreements\n"
        f"Base your analysis strictly on the abstracts provided above with zero extrapolation."
    )

    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }
        res = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=45)
        if res.status_code == 200:
            return res.json().get("response", "").strip()
    except Exception:
        pass

    # Extractive Fallback synthesis
    lines = [
        "### 📑 Collective Literature Synthesis (Extractive Grounded Overview)",
        f"**Research Topic:** *{query}*\n",
        "**1. Executive Summary:**",
        "The retrieved literature collectively investigates key mechanisms, clinical/experimental findings, and theoretical frameworks directly addressing this inquiry.\n",
        "**2. Individual Findings:**"
    ]
    for idx, p in enumerate(top_papers[:3], 1):
        abstract_snippet = p.get('text', '')[:220].strip() + "..."
        lines.append(f"- **[Paper {idx}] \"{p.get('title', 'Untitled')}\":** {abstract_snippet}")
    lines.append("\n*(Note: For generative LLM synthesis, ensure Ollama is connected on localhost or enter an external API key in the sidebar).*")
    return "\n".join(lines)



def generate_literature_matrix(
    top_papers: List[Dict[str, Any]],
    query: str,
    model: str = "llama3.2:3b"
) -> str:
    """
    Generates a structured Literature Review Comparative Matrix in Markdown format.
    Columns: | Paper Title | Core Problem Addressed | Methodology | Dataset Evaluated | Key Findings |
    """
    papers_context = ""
    for idx, paper in enumerate(top_papers[:4], start=1):
        papers_context += (
            f"\n[Paper {idx}]: \"{paper.get('title', 'Untitled')}\"\n"
            f"Abstract: {paper.get('text', '')}\n"
        )

    prompt = (
        f"You are a scientific literature analysis assistant.\n"
        f"Target Query: \"{query}\"\n\n"
        f"Papers to analyze:\n{papers_context}\n\n"
        f"Produce a structured Markdown table comparing these papers with the EXACT columns:\n"
        f"| Paper | Core Problem Addressed | Methodology / Architecture | Dataset / Experimental Setup | Key Findings |\n"
        f"Keep entries concise (1-2 lines per cell). Only output the Markdown table without conversational filler."
    )

    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }
        res = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=45)
        if res.status_code == 200:
            return res.json().get("response", "").strip()
    except Exception:
        pass

    # Extractive Fallback Markdown Table
    rows = [
        "| Paper | Core Problem Addressed | Methodology / Setup | Key Findings |",
        "| :--- | :--- | :--- | :--- |"
    ]
    for p in top_papers[:4]:
        title = p.get('title', 'Untitled')[:45]
        abstract = p.get('text', '')
        sents = [s.strip() for s in abstract.split(".") if len(s.strip()) > 15]
        problem = sents[0][:70] + "..." if sents else "Scientific hypothesis"
        findings = sents[-1][:70] + "..." if len(sents) > 1 else abstract[:70]
        rows.append(f"| **{title}** | {problem} | Empirical Study | {findings} |")
    return "\n".join(rows)


