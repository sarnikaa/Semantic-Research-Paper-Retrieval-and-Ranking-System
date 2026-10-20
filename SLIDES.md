# Presentation Slides: Semantic Research Paper Retrieval and Ranking System

---

## Slide 1: Problem Statement & Objectives

### Title: Semantic Research Paper Retrieval and Ranking System
**Subtitle:** Multi-Stage Hybrid Search, Neural Re-ranking, and Grounded Local LLM Synthesis

### Problem Statement
* **Lexical Matching Breakdown:** Traditional academic search (Okapi BM25 / TF-IDF) relies strictly on surface keyword overlap, failing when queries and relevant literature express identical concepts using different terminology.
* **Vocabulary Mismatch & High Overhead:** Conceptual and interdisciplinary literature is missed, while researchers face significant cognitive overhead manually sifting through dense, technical abstracts.
* **Score Calibration Gap:** Conventional engines fail to calibrate and balance semantic similarity, citation graph authority, and temporal recency in a unified ranking function.

### Objectives
* **Dual-Index Candidate Generation:** Build parallel Okapi BM25 sparse lexical and scientific dense vector indexes (`all-MiniLM-L6-v2`, FAISS) for high-recall first-stage candidate retrieval.
* **Hybrid Rank Fusion:** Combine lexical precision and semantic recall using parameter-tuned Reciprocal Rank Fusion (RRF, $k=10$).
* **Neural Re-Ranking & Calibrated Scoring:** Maximize top-tier precision with joint cross-attention (`ms-marco-MiniLM-L-6-v2`) and normalize multi-signal weights (PageRank + Recency).
* **Grounded Local LLM Layer:** Integrate Ollama (`llama3.2:3b`) for on-demand query-conditioned explanations, literature synthesis, and HyDE query expansion with zero external API dependencies.
* **Empirical Benchmarking:** Quantitatively evaluate on the BEIR SciFact and SCIDOCS benchmarks using nDCG@10, MAP, MRR, Recall@100, and query latency with paired statistical significance tests.

---

## Slide 2: Dataset, Algorithms & Architecture

### Corpus & Evaluation Framework (Dual-Track)
* **Interactive Live Search Corpus:** **arXiv Metadata Dataset** filtered to Computer Science domains (`cs.IR`, `cs.CL`, `cs.LG`; ~200k papers) with Semantic Scholar citation edges.
* **Quantitative Benchmark:** **BEIR SciFact & SCIDOCS** benchmarks providing real scientific queries and expert-annotated relevance judgments (`qrels`).
* **Preprocessing:** Token normalization and stopword removal for BM25; raw natural language preserved for transformer encoders.

### Algorithms & Multi-Stage Architecture
* **Stage 1 — Dual Candidate Generation:**
  * *Lexical Retriever:* **Okapi BM25** (`rank-bm25`) for exact keyword, gene, and acronym matching.
  * *Dense Vector Retriever:* **Sentence-Transformers** (`all-MiniLM-L6-v2`) with **FAISS (IndexFlatIP / HNSW)** for fast cosine similarity search.
* **Stage 2 — Hybrid Fusion & Neural Re-Ranking:**
  * *Fusion:* Parameter-tuned **Reciprocal Rank Fusion (RRF, $k=10$)** yielding **nDCG@10 = 0.6989** (+4.7% over BM25, $p < 0.01$).
  * *Neural Re-Ranker:* **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) scoring joint `(query, abstract)` pairs.
  * *Calibration Layer:* Min-Max normalized linear blend of topical score, citation **PageRank** centrality, and exponential **temporal recency decay**.
* **Stage 3 — Grounded Synthesis & Query Expansion:**
  * **Ollama Runtime:** Locally hosted, quantized LLM (`llama3.2:3b`) executing offline.
  * *Capabilities:* On-demand grounded relevance rationale, multi-paper literature synthesis (mini-RAG), and HyDE hypothetical abstract generation.
* **Evaluation & Rigor:**
  * Statistical significance via Student's paired $t$-test ($p < 0.01$).
  * Automated qualitative error analysis contrasting semantic wins vs. lexical wins.
