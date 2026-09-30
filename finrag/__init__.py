"""
FinRAG — Financial Retrieval-Augmented Generation pipeline.

Modules
-------
config      Dataset configurations and project paths.
data        JSONL / TSV loaders.
chunking    Dataset-aware document splitting.
models      Embedding model and LLM (lazy singletons, MPS-aware).
vectorstore ChromaDB build / load helpers.
retrieval   ChromaDB + BM25 retrieval with RRF fusion.
evaluation  NDCG@10 and coverage metrics.
guardrails  Query and answer validation to keep outputs grounded and safe.
pipeline    End-to-end orchestrator.
"""
