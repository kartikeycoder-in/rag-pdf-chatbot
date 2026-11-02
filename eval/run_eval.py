"""Evaluate dense, BM25, and hybrid retrieval with hit-rate@5."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.indexer import build_dense_index, load_embedding_model
from src.ingest import load_and_chunk_directory
from src.retriever import bm25_search, build_bm25_index, dense_search, rrf_fuse

EVAL_PATH = ROOT / "eval" / "eval_set.json"


def _metadata(result: Any) -> dict[str, Any]:
    """Return metadata from a dense tuple or dictionary result."""
    return result["metadata"] if isinstance(result, dict) else result[2]


def _acceptable_pages(item: dict[str, Any]) -> set[int]:
    """Return the primary and alternate acceptable evidence pages."""
    return {int(item["expected_page"]), *(int(page) for page in item.get("acceptable_pages", []))}


def _hit(results: list[Any], item: dict[str, Any], top_k: int = 5) -> bool:
    """Return whether an expected source page appears within top_k."""
    pages = _acceptable_pages(item)
    return any(
        _metadata(result).get("source") == item["expected_source"]
        and int(_metadata(result).get("page", -1)) in pages
        for result in results[:top_k]
    )


def _rank(results: list[Any], item: dict[str, Any]) -> int | None:
    """Return the rank of any accepted page, or None when absent from top ten."""
    pages = _acceptable_pages(item)
    for rank, result in enumerate(results, start=1):
        metadata = _metadata(result)
        if metadata.get("source") == item["expected_source"] and int(metadata.get("page", -1)) in pages:
            return rank
    return None


def _as_dict(result: tuple[str, str, dict[str, Any], float]) -> dict[str, Any]:
    """Convert a dense search tuple to the record shape used by RRF."""
    chunk_id, text, metadata, score = result
    return {"chunk_id": chunk_id, "text": text, "metadata": metadata, "score": score}


def _miss_reason(ranks: dict[str, int | None]) -> str:
    """Give a cautious retrieval-based diagnostic for a hit@5 miss."""
    if ranks["Hybrid"] is not None and ranks["Hybrid"] > 5:
        return "RRF combined the two rankings and pushed this page below rank 5."
    if ranks["Dense"] is not None and ranks["Dense"] <= 5 and (ranks["BM25"] is None or ranks["BM25"] > 5):
        return "The wording may have weak exact-term overlap for BM25."
    if ranks["BM25"] is not None and ranks["BM25"] <= 5 and (ranks["Dense"] is None or ranks["Dense"] > 5):
        return "The query's wording may differ semantically from the relevant passage."
    return "Nearby chunks or document terminology may be competing for the top ranks."


def main() -> None:
    """Build indexes if needed and print retrieval hit-rate@5 results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), help="Evaluate only this split")
    args = parser.parse_args()
    items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    if items and not all("split" in item for item in items):
        print("WARNING: split labels are absent; evaluation uses all questions and is not held out.")
    elif args.split:
        items = [item for item in items if item.get("split") == args.split]
        print(f"Evaluating split={args.split}: {len(items)} questions")
    if not items:
        print("No questions selected.")
        return
    chunks = load_and_chunk_directory(ROOT / "data")
    model = load_embedding_model() if chunks else None
    if chunks:
        indexed_count = build_dense_index(chunks, ROOT / "chroma_db", model)
        build_bm25_index(chunks, ROOT / "bm25_index.pkl")
    else:
        indexed_count = 0

    if indexed_count < 100 or len(items) < 15:
        gaps = []
        if indexed_count < 100:
            gaps.append(f"only {indexed_count} indexed chunks (need 100+)")
        if len(items) < 15:
            gaps.append(f"only {len(items)} eval questions (need 15+)")
        print("WARNING: evaluation numbers are not reliable: " + "; ".join(gaps) + ".")

    retrieved: list[dict[str, Any]] = []
    for item in items:
        query = item["question"]
        dense = dense_search(query, 10, ROOT / "chroma_db", model)
        sparse = bm25_search(query, 10, ROOT / "bm25_index.pkl")
        hybrid = rrf_fuse([[_as_dict(result) for result in dense], sparse], k=60, top_n=10)
        retrieved.append({"item": item, "Dense": dense, "BM25": sparse, "Hybrid": hybrid})

    results: dict[str, tuple[float, dict[str, float]]] = {}
    for name in ("Dense", "BM25", "Hybrid"):
        hits = sum(_hit(row[name], row["item"]) for row in retrieved)
        per_type = {}
        for kind in ("keyword", "paraphrase"):
            typed_rows = [row for row in retrieved if row["item"].get("type") == kind]
            per_type[kind] = (
                sum(_hit(row[name], row["item"]) for row in typed_rows) / len(typed_rows)
                if typed_rows else float("nan")
            )
        results[name] = (hits / len(items) if items else float("nan"), per_type)

    print("\n| Retrieval | Overall hit-rate@5 | Keyword | Paraphrase |")
    print("|---|---:|---:|---:|")
    for name, (overall, per_type) in results.items():
        overall_text = f"{overall:.1%}" if overall == overall else "N/A"
        keyword = f"{per_type['keyword']:.1%}" if per_type["keyword"] == per_type["keyword"] else "N/A"
        paraphrase = f"{per_type['paraphrase']:.1%}" if per_type["paraphrase"] == per_type["paraphrase"] else "N/A"
        print(f"| {name} | {overall_text} | {keyword} | {paraphrase} |")

    misses = []
    for row in retrieved:
        ranks = {name: _rank(row[name], row["item"]) for name in ("Dense", "BM25", "Hybrid")}
        if any(rank is None or rank > 5 for rank in ranks.values()):
            misses.append((row["item"], ranks))
    if misses:
        print("\n## Questions missed by at least one method at hit-rate@5")
        for item, ranks in misses:
            pretty = {name: (f"rank {rank}" if rank is not None else "not in top 10") for name, rank in ranks.items()}
            print(f"- **{item['question']}** — Dense: {pretty['Dense']}; BM25: {pretty['BM25']}; Hybrid: {pretty['Hybrid']}. {_miss_reason(ranks)}")
    else:
        print("\nAll expected pages appeared in the top five for all three methods.")


if __name__ == "__main__":
    main()
