"""Compare retrieval settings without modifying the app's persistent indexes."""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.indexer import build_dense_index, load_embedding_model
from src.ingest import load_and_chunk_directory
from src.retriever import bm25_search, build_bm25_index, dense_search, rrf_fuse
from eval.run_eval import _as_dict, _hit

EVAL_PATH = ROOT / "eval" / "eval_set.json"


def _metrics(rows: list[tuple[dict[str, Any], dict[str, list[Any]]]], method: str) -> tuple[float, float, float]:
    """Return overall, keyword, and paraphrase hit-rate@5."""
    def rate(selected: list[tuple[dict[str, Any], dict[str, list[Any]]]]) -> float:
        return sum(_hit(ranks[method], item) for item, ranks in selected) / len(selected) if selected else float("nan")
    return rate(rows), rate([r for r in rows if r[0].get("type") == "keyword"]), rate([r for r in rows if r[0].get("type") == "paraphrase"])


def _retrieve(items: list[dict[str, Any]], dense_dir: Path, bm25_path: Path, model: Any,
              depth: int, rrf_k: int, weights: tuple[float, float], cross_encoder: Any = None) -> tuple[list[tuple[dict[str, Any], dict[str, list[Any]]]], list[float]]:
    """Retrieve rankings for items and optionally rerank hybrid candidates."""
    rows = []
    latencies = []
    for item in items:
        query = item["question"]
        dense = dense_search(query, depth, dense_dir, model)
        sparse = bm25_search(query, depth, bm25_path)
        dense_dicts = [_as_dict(result) for result in dense]
        hybrid = _weighted_rrf(dense_dicts, sparse, rrf_k, weights, 20 if cross_encoder else 5)
        if cross_encoder:
            start = time.perf_counter()
            scores = cross_encoder.predict([(query, result["text"]) for result in hybrid])
            latencies.append(time.perf_counter() - start)
            hybrid = [result for _, result in sorted(zip(scores, hybrid), key=lambda pair: float(pair[0]), reverse=True)[:5]]
        rows.append((item, {"Dense": dense, "BM25": sparse, "Hybrid": hybrid}))
    return rows, latencies


def _weighted_rrf(dense: list[dict[str, Any]], sparse: list[dict[str, Any]], k: int,
                  weights: tuple[float, float], top_n: int) -> list[dict[str, Any]]:
    """Fuse two ranked lists with weighted reciprocal-rank scores."""
    fused: dict[str, dict[str, Any]] = {}
    for ranking, weight in ((dense, weights[0]), (sparse, weights[1])):
        seen: set[str] = set()
        for rank, chunk in enumerate(ranking, 1):
            cid = str(chunk["chunk_id"])
            if cid in seen:
                continue
            seen.add(cid)
            item = fused.setdefault(cid, {**chunk, "score": 0.0})
            item["score"] += weight / (k + rank)
    return sorted(fused.values(), key=lambda result: (-result["score"], str(result["chunk_id"])))[:top_n]


def _report_table(rows: list[tuple[str, list[tuple[dict[str, Any], dict[str, list[Any]]]], str]], title: str) -> None:
    """Print a markdown table for all method metrics."""
    print(f"\n### {title}\n\n| Configuration | Method | Overall | Keyword | Paraphrase | Mean rerank latency/query |")
    print("|---|---|---:|---:|---:|---:|")
    for config, records, latency_text in rows:
        for method in ("Dense", "BM25", "Hybrid"):
            overall, keyword, paraphrase = _metrics(records, method)
            print(f"| {config} | {method} | {overall:.1%} | {keyword:.1%} | {paraphrase:.1%} | {latency_text if method == 'Hybrid' else '—'} |")


def main() -> None:
    """Run separate parameter sweeps on dev and evaluate its winner once on test."""
    all_items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    has_splits = bool(all_items) and all("split" in item for item in all_items)
    if not has_splits:
        print("WARNING: no complete split labels found; running all questions. Results are not held out.")
        dev = all_items
        test: list[dict[str, Any]] = []
    else:
        dev = [item for item in all_items if item.get("split") == "dev"]
        test = [item for item in all_items if item.get("split") == "test"]
        if not dev:
            print("WARNING: no dev questions; tuning on all questions. Results are not held out.")
            dev, test = all_items, []
    chunks = load_and_chunk_directory(ROOT / "data")
    model = load_embedding_model()
    configs = [(f"depth={d}, RRF=60, weights=1:1, chunk=800/150", d, 60, (1., 1.), (800,150)) for d in (10,20,30)]
    configs += [(f"depth=10, RRF={k}, weights=1:1, chunk=800/150",10,k,(1.,1.),(800,150)) for k in (20,100)]
    configs += [(f"depth=10, RRF=60, weights={a:g}:{b:g}, chunk=800/150",10,60,(a,b),(800,150)) for a,b in ((1.5,1.),(1.,1.5))]
    configs += [(f"depth=10, RRF=60, weights=1:1, chunk={size}/{overlap}",10,60,(1.,1.),(size,overlap)) for size,overlap in ((500,100),(1200,200))]
    dev_table = []
    retained: list[tuple[tuple[str,int,int,tuple[float,float],tuple[int,int]], Path, Path]] = []
    temp_root = tempfile.TemporaryDirectory(prefix="rag-tune-")
    for idx, (label, depth, rrf_k, weights, chunk_params) in enumerate(configs):
        size, overlap = chunk_params
        local_chunks = chunks if (size,overlap)==(800,150) else load_and_chunk_directory_with_splitter(ROOT / "data", size, overlap)
        dense_dir = Path(temp_root.name) / f"dense-{idx}"
        bm25_path = Path(temp_root.name) / f"bm25-{idx}.pkl"
        build_dense_index(local_chunks, dense_dir, model)
        build_bm25_index(local_chunks, bm25_path)
        result, _ = _retrieve(dev, dense_dir, bm25_path, model, depth, rrf_k, weights)
        dev_table.append((label, result, "—"))
        retained.append(((label,depth,rrf_k,weights,chunk_params),dense_dir,bm25_path))
    # Reranker is an independent configuration using the app's default candidate source.
    cross_record = None
    try:
        from sentence_transformers import CrossEncoder
        cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        default = retained[0]
        label = "cross-encoder rerank top20 (default hybrid candidates)"
        rec, latencies = _retrieve(dev, default[1], default[2], model, 10, 60, (1.,1.), cross)
        cross_record = (label, rec, f"{sum(latencies)/len(latencies):.3f}s" if latencies else "N/A")
        dev_table.append(cross_record)
    except Exception as exc:
        print(f"WARNING: cross-encoder experiment skipped: {type(exc).__name__}: {exc}")
    _report_table(dev_table, f"Dev tuning results ({len(dev)} questions)")
    scores = [(sum(_hit(r[1]["Hybrid"], r[0]) for r in records)/len(records) if records else -1., i)
              for i, (_, records, _) in enumerate(dev_table)]
    best_score, best_i = max(scores, default=(-1.,0))
    best_label, best_records, _ = dev_table[best_i]
    print(f"\nDev recommendation: {best_label} — hybrid hit-rate@5 {best_score:.1%}.")
    if test:
        if best_label.startswith("cross-encoder") and cross_record:
            # Test only the chosen setting, with no further selection.
            chosen = retained[0]
            cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            test_rows, lat = _retrieve(test, chosen[1], chosen[2], model, 10, 60, (1.,1.), cross)
            test_lat = f"{sum(lat)/len(lat):.3f}s" if lat else "N/A"
        else:
            chosen = retained[best_i]
            _, depth, rrf_k, weights, _ = chosen[0]
            test_rows, lat = _retrieve(test, chosen[1], chosen[2], model, depth, rrf_k, weights)
            test_lat = "—"
        _report_table([(best_label, test_rows, test_lat)], f"Held-out test results for dev-selected winner ({len(test)} questions)")
    temp_root.cleanup()


def load_and_chunk_directory_with_splitter(data_dir: Path, chunk_size: int, chunk_overlap: int) -> list[dict[str, Any]]:
    """Chunk all PDFs with requested temporary chunking parameters."""
    from src.ingest import chunk_pdf
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[dict[str, Any]] = []
    for pdf in sorted(data_dir.glob("*.pdf")):
        chunks.extend(chunk_pdf(pdf, splitter))
    return chunks


if __name__ == "__main__":
    main()
