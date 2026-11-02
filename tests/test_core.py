"""Core regression tests for chunk metadata, fusion, and the no-context path."""

import pymupdf

from src.ingest import chunk_pdf
from src.llm import NOT_FOUND
from src.pipeline import answer
from src.retriever import rrf_fuse
from eval.run_eval import _hit


def test_chunk_pdf_metadata(tmp_path):
    """Extracted chunks keep source filename and one-based page metadata."""
    pdf_path = tmp_path / "metadata.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "PDF page metadata sample " * 40)
    document.save(pdf_path)
    document.close()

    chunks = chunk_pdf(pdf_path)

    assert chunks
    assert all(chunk["metadata"]["source"] == "metadata.pdf" for chunk in chunks)
    assert all(chunk["metadata"]["page"] == 1 for chunk in chunks)
    assert all(chunk["metadata"]["chunk_id"] == chunk["chunk_id"] for chunk in chunks)


def test_rrf_fuse_combines_duplicate_chunks():
    """RRF combines scores when a chunk appears in both rankings."""
    first = {"chunk_id": "a", "text": "a", "metadata": {}}
    second = {"chunk_id": "b", "text": "b", "metadata": {}}
    fused = rrf_fuse([[first, second], [second]], k=60, top_n=5)

    assert [item["chunk_id"] for item in fused] == ["b", "a"]
    assert fused[0]["score"] == (1 / 62 + 1 / 61)


def test_answer_returns_exact_not_found_without_context():
    """No retrieved evidence returns the exact fallback without an LLM call."""
    result = answer("unanswerable", search_fn=lambda _: [], llm_fn=lambda *_: "must not run")

    assert result == {"answer": NOT_FOUND, "sources": []}


def test_answer_hides_sources_for_not_found_response():
    """Fallback answers do not expose irrelevant retrieved chunks as sources."""
    chunk = {"chunk_id": "x", "text": "irrelevant", "metadata": {"source": "paper.pdf", "page": 1}}
    result = answer("unanswerable", search_fn=lambda _: [chunk], llm_fn=lambda *_: NOT_FOUND)

    assert result == {"answer": NOT_FOUND, "sources": []}


def test_eval_hit_accepts_an_alternate_page():
    """A retrieved source on an explicitly acceptable page counts as a hit."""
    results = [{"metadata": {"source": "paper.pdf", "page": 7}}]
    item = {"expected_source": "paper.pdf", "expected_page": 3, "acceptable_pages": [7]}

    assert _hit(results, item)
