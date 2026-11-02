"""Question answering orchestration over hybrid retrieval and generation."""

from __future__ import annotations

from typing import Any, Callable

from src.llm import NOT_FOUND, generate_answer
from src.retriever import hybrid_search


def answer(
    question: str,
    *,
    search_fn: Callable[[str], list[dict[str, Any]]] = hybrid_search,
    llm_fn: Callable[[str, list[dict[str, Any]]], str] = generate_answer,
) -> dict[str, Any]:
    """Retrieve evidence and return an answer with its supporting chunks."""
    sources = search_fn(question)
    if not sources:
        return {"answer": NOT_FOUND, "sources": []}
    response = llm_fn(question, sources)
    if response == NOT_FOUND:
        return {"answer": NOT_FOUND, "sources": []}
    return {"answer": response, "sources": sources}
