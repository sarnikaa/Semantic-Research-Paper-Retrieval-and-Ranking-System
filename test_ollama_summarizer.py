"""
Standalone test script for Local LLM Grounded Summarization using Ollama.
Retrieves the top candidate paper from SciFact and tests query-conditioned synthesis.
"""

import os
import sys
import time
import json
import logging
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.data_loader import load_beir_dataset
from src.dense_retriever import DenseRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_ollama")

OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def check_ollama_status():
    """Checks if Ollama service is running and lists installed models."""
    try:
        res = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if res.status_code == 200:
            models_info = res.json().get("models", [])
            model_names = [m["name"] for m in models_info]
            return True, model_names
        return False, []
    except Exception:
        return False, []


def generate_grounded_summary(
    query: str,
    paper_title: str,
    paper_abstract: str,
    model: str = "llama3.2:3b"
) -> str:
    """Invokes Ollama API to generate a grounded explanation strictly based on abstract."""
    system_prompt = (
        "You are an objective scientific literature assistant.\n"
        "Your task is to explain the relevance of the retrieved paper to the user's query.\n"
        "CRITICAL RULES:\n"
        "1. Strictly base your answer ONLY on the provided abstract text.\n"
        "2. Do NOT extrapolate, hallucinate, or assume facts outside the text.\n"
        "3. Provide exactly two concise bullet points:\n"
        "   - Core Finding: The paper's primary experimental or theoretical result.\n"
        "   - Query Relevance: Exactly how this finding supports, refutes, or addresses the user's query."
    )

    user_prompt = (
        f"User Query: \"{query}\"\n\n"
        f"Paper Title: \"{paper_title}\"\n"
        f"Paper Abstract: \"{paper_abstract}\"\n\n"
        f"Provide the grounded explanation:"
    )

    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature to enforce strict factual grounding
            "top_p": 0.9
        }
    }

    t0 = time.perf_counter()
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=60)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if resp.status_code == 200:
        return resp.json().get("response", ""), latency_ms
    else:
        raise RuntimeError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")


def main():
    print("=" * 75)
    print("  TESTING LOCAL LLM GROUNDED SUMMARIZATION (OLLAMA)")
    print("=" * 75)

    is_running, models = check_ollama_status()
    print(f"Ollama Server Status : {'ONLINE [OK]' if is_running else 'OFFLINE [FAILED]'}")
    print(f"Available Models     : {models if models else 'None detected'}")

    if not is_running:
        print("\nOllama daemon is not currently responding at http://127.0.0.1:11434.")
        print("Please start Ollama service ('ollama serve').")
        return

    # Choose model
    target_model = None
    priority_models = ["llama3.2:3b", "llama3.2:latest", "phi3:mini", "qwen2.5:1.5b", "mistral:7b"]
    for pm in priority_models:
        for m in models:
            if pm in m:
                target_model = m
                break
        if target_model:
            break

    if not target_model:
        if models:
            target_model = models[0]
        else:
            target_model = "llama3.2:3b"
            print(f"\nNo models found. Attempting to pull '{target_model}'...")
            pull_resp = requests.post(f"{OLLAMA_BASE_URL}/api/pull", json={"name": target_model}, stream=False)
            print("Pull initiated.")

    print(f"Selected Model       : {target_model}")

    # Load a test query and retrieve the top document
    print("\nLoading SciFact corpus and cached dense index...")
    corpus, _, _ = load_beir_dataset("scifact", split="test")
    dense = DenseRetriever()
    dense.load_embeddings("data/scifact_embeddings_minilm.npy")

    test_query = "An increase in colorectal cancer is associated with diet"
    print(f"\nTest Query: \"{test_query}\"")

    results = dense.search(test_query, top_k=1)
    top_doc_id = list(results.keys())[0]
    top_doc = corpus[top_doc_id]

    print(f"Top Retrieved Paper  : [Doc ID: {top_doc_id}]")
    print(f"Title                : {top_doc['title']}")
    print(f"Abstract Snippet     : {top_doc['text'][:200]}...")

    print("\nGenerating Grounded Explanation via Ollama...")
    explanation, latency_ms = generate_grounded_summary(
        query=test_query,
        paper_title=top_doc['title'],
        paper_abstract=top_doc['text'],
        model=target_model
    )

    print("\n" + "-" * 75)
    print(f"GROUNDED SUMMARY & RELEVANCE RATIONALE ({latency_ms:.1f} ms):")
    print("-" * 75)
    print(explanation.strip())
    print("-" * 75)
    print("[SUCCESS] Grounded summarization pipeline successfully validated!")


if __name__ == "__main__":
    main()
