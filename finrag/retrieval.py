"""
Hybrid retrieval with ChromaDB dense search, BM25, and Reciprocal Rank Fusion.
"""

import logging

from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)

_RETRIEVER_TOP_K = 10


# ── BM25 indexes ─────────────────────────────────────────────────────────────

def build_bm25_retriever(lc_docs) -> BM25Retriever:
    """Build an in-memory BM25 retriever that returns its top 10 chunks."""
    return BM25Retriever.from_documents(lc_docs, k=_RETRIEVER_TOP_K)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def rrf_fuse(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked doc-ID lists with Reciprocal Rank Fusion.

    RRF score for document d:
        score(d) = Σ  1 / (k + rank_r(d))   over all retrievers r

    k=60 is the standard constant — robust across datasets with no tuning.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        seen: set[str] = set()
        rank = 0
        for doc_id in ranked:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            rank += 1
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Optional: MultiQuery expansion ───────────────────────────────────────────

def expand_query(query: str, llm) -> list[str]:
    """
    Use an LLM to generate 3 alternative phrasings of the query.
    Returns [original] + [variants] so retrieval always includes the original.
    Falls back gracefully if the LLM fails.
    """
    prompt = (
        "Generate 3 alternative search queries for the following financial question. "
        "Output only the queries, one per line, no numbering or explanation.\n\n"
        f"Original: {query}"
    )
    try:
        response = llm.invoke(prompt)
        variants = [
            line.strip()
            for line in response.content.strip().split("\n")
            if line.strip() and line.strip() != query
        ][:3]
        queries = [query] + variants
        logger.debug("MultiQuery expanded to %d variants.", len(queries))
        return queries
    except Exception as exc:
        logger.warning("MultiQuery expansion failed (%s) — using original query.", exc)
        return [query]


# ── Core retrieval function ───────────────────────────────────────────────────

def retrieve_hybrid(
    vectorstore,
    bm25_retriever: BM25Retriever,
    query: str,
    dataset_type: str,
    llm=None,
    k: int = _RETRIEVER_TOP_K,
) -> list[tuple[str, float]]:
    """
    Retrieve top-10 chunks from dense search and BM25, then fuse their rankings.

    Parameters
    ----------
    vectorstore   : ChromaDB store for this dataset
    bm25_retriever: LangChain BM25Retriever
    query         : raw query string
    dataset_type  : "passage" | "tabular"
    llm           : optional LLM for MultiQuery expansion (None = disabled)
    k             : final top-k to return after fusion

    Returns
    -------
    List of (corpus_id, RRF score) sorted descending.
    """
    # ── Stage 1 — Optional MultiQuery expansion ───────────────────────────
    queries = expand_query(query, llm) if llm is not None else [query]

    # ── Stage 2 — Dense retrieval (top 10 per query variant) ──────────────
    dense_ranked: list[list[str]] = []
    for q in queries:
        if dataset_type == "passage":
            docs = vectorstore.max_marginal_relevance_search(
                q,
                k=_RETRIEVER_TOP_K,
                fetch_k=_RETRIEVER_TOP_K * 3,
                lambda_mult=0.7,
            )
        else:
            docs = vectorstore.similarity_search(q, k=_RETRIEVER_TOP_K)
        dense_ranked.append([d.metadata["id"] for d in docs])

    # ── Stage 3 — BM25 retrieval ──────────────────────────────────────────
    bm25_docs   = bm25_retriever.invoke(query)          # always use original query
    bm25_ranked = [d.metadata["id"] for d in bm25_docs]

    # Pool query variants into the top 10 dense results, then fuse with BM25.
    dense_fused = rrf_fuse(dense_ranked, k=60)[:_RETRIEVER_TOP_K]
    fused = rrf_fuse(
        [[doc_id for doc_id, _ in dense_fused], bm25_ranked],
        k=60,
    )
    return fused[:k]
