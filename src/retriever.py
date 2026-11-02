"""Dense, sparse, and reciprocal-rank-fused retrieval."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.indexer import DEFAULT_CHROMA_PATH, get_collection

BM25_PATH = Path("./bm25_index.pkl")
TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize text using a simple word regex."""
    return TOKEN_PATTERN.findall(text.lower())


def build_bm25_index(chunks: list[dict[str, Any]], path: str | Path = BM25_PATH) -> None:
    """Persist a BM25 index over chunk text and corresponding chunk records."""
    tokenized = [tokenize(chunk["text"]) for chunk in chunks]
    index = BM25Okapi(tokenized) if tokenized else None
    payload = {"index": index, "chunks": chunks}
    with Path(path).open("wb") as file:
        pickle.dump(payload, file)


def load_bm25_index(path: str | Path = BM25_PATH) -> dict[str, Any] | None:
    """Load persisted BM25 data, returning None when no index exists."""
    index_path = Path(path)
    if not index_path.exists():
        return None
    with index_path.open("rb") as file:
        return pickle.load(file)


def dense_search(
    query: str,
    k: int = 10,
    persist_dir: str | Path = DEFAULT_CHROMA_PATH,
    model: Any = None,
) -> list[tuple[str, str, dict[str, Any], float]]:
    """Return ranked (chunk_id, text, metadata, distance) dense matches."""
    collection = get_collection(persist_dir, model)
    if not collection.count():
        return []
    result = collection.query(query_texts=[query], n_results=min(k, collection.count()))
    rows: list[tuple[str, str, dict[str, Any], float]] = []
    for chunk_id, text, metadata, score in zip(
        result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        rows.append((chunk_id, text, metadata, float(score)))
    return rows


def bm25_search(query: str, k: int = 10, path: str | Path = BM25_PATH) -> list[dict[str, Any]]:
    """Return the top keyword matches from the persisted BM25 index."""
    payload = load_bm25_index(path)
    if not payload or payload["index"] is None:
        return []
    chunks = payload["chunks"]
    scores = payload["index"].get_scores(tokenize(query))
    ranked_indices = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), i))[:k]
    return [{**chunks[i], "score": float(scores[i])} for i in ranked_indices]


def rrf_fuse(
    rankings: list[list[dict[str, Any]]], k: int = 60, top_n: int = 5
) -> list[dict[str, Any]]:
    """Merge ranked chunk lists by reciprocal rank and return top_n results."""
    fused: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, chunk in enumerate(ranking, start=1):
            chunk_id = str(chunk["chunk_id"])
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            item = fused.setdefault(chunk_id, {**chunk, "score": 0.0})
            item["score"] += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda item: (-item["score"], item["chunk_id"]))[:top_n]


def hybrid_search(
    query: str,
    persist_dir: str | Path = DEFAULT_CHROMA_PATH,
    bm25_path: str | Path = BM25_PATH,
    model: Any = None,
) -> list[dict[str, Any]]:
    """Fuse the top ten dense and BM25 results into five candidates."""
    dense_results = [
        {"chunk_id": chunk_id, "text": text, "metadata": metadata, "score": score}
        for chunk_id, text, metadata, score in dense_search(query, 10, persist_dir, model)
    ]
    return rrf_fuse(
        [dense_results, bm25_search(query, 10, bm25_path)],
        k=60,
        top_n=5,
    )
