"""Prompt construction and OpenAI-compatible Groq generation."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, AuthenticationError, OpenAI, RateLimitError

NOT_FOUND = "I couldn't find this in the uploaded documents."
SYSTEM_PROMPT = (
    "Answer ONLY from the numbered context. Cite every factual claim as [filename, p.X], "
    "using the exact PDF filename and page of the chunk that actually contains the supporting statement. "
    "When several chunks support a claim, prefer the most detailed chunk over an abstract or summary. "
    "Do not cite a context number. "
    f'If the context lacks the answer, reply exactly: "{NOT_FOUND}"'
)
CITATION_PATTERN = re.compile(r"\[[^\]\n,]+\.pdf,\s*p\.\d+\]", re.IGNORECASE)
NUMBERED_CITATION_PATTERN = re.compile(r"\[(\d+),\s*p\.\d+\]")


class LLMError(RuntimeError):
    """Friendly, user-facing provider error safe to display in the app."""


def build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Format numbered context chunks with source labels and the question."""
    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        context_parts.append(
            f"[{index}] Source: {metadata['source']}, page {metadata['page']}\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts) if context_parts else "(No context provided.)"
    return f"Context:\n{context}\n\nQuestion: {question}"


def _citation_sources(chunks: list[dict[str, Any]]) -> str:
    """Format the top three retrieved chunks as a compact source line."""
    citations = []
    for chunk in chunks[:3]:
        metadata = chunk["metadata"]
        citation = f"[{metadata['source']}, p.{metadata['page']}]"
        if citation not in citations:
            citations.append(citation)
    return "Sources: " + ", ".join(citations)


def _friendly_error(error: Exception) -> LLMError:
    """Translate provider exceptions into concise UI-safe messages."""
    if isinstance(error, AuthenticationError):
        return LLMError("Groq authentication failed. Check LLM_API_KEY in your .env file.")
    if isinstance(error, RateLimitError):
        return LLMError("Groq rate limit reached after retry. Please wait briefly and try again.")
    if isinstance(error, APIConnectionError):
        return LLMError("Could not connect to Groq. Check your connection and LLM_BASE_URL.")
    return LLMError("The language model request failed. Check your Groq settings and try again.")


def generate_answer(question: str, chunks: list[dict[str, Any]]) -> str:
    """Generate a cited answer through Groq's OpenAI-compatible chat endpoint."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    missing = [name for name, value in (
        ("LLM_BASE_URL", base_url), ("LLM_API_KEY", api_key), ("LLM_MODEL", model)
    ) if not value]
    if missing:
        raise LLMError("Missing required .env setting(s): " + ", ".join(missing))

    client = OpenAI(base_url=base_url, api_key=api_key)
    request = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
    }
    try:
        response = client.chat.completions.create(**request)
    except RateLimitError:
        time.sleep(1)
        try:
            response = client.chat.completions.create(**request)
        except (AuthenticationError, RateLimitError, APIConnectionError) as error:
            raise _friendly_error(error) from None
    except (AuthenticationError, APIConnectionError) as error:
        raise _friendly_error(error) from None

    answer = (response.choices[0].message.content or "").strip()
    if answer == NOT_FOUND:
        return NOT_FOUND
    answer = answer.replace("【", "[").replace("】", "]")

    def replace_numbered_citation(match: re.Match[str]) -> str:
        """Convert a model's [n, p.X] reference to the matching PDF citation."""
        index = int(match.group(1)) - 1
        if 0 <= index < len(chunks):
            metadata = chunks[index]["metadata"]
            return f"[{metadata['source']}, p.{metadata['page']}]"
        return match.group(0)

    answer = NUMBERED_CITATION_PATTERN.sub(replace_numbered_citation, answer)
    if not CITATION_PATTERN.search(answer):
        answer = f"{answer}\n\n{_citation_sources(chunks)}".strip()
    return answer
