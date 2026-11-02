"""Unit tests for answer post-processing and friendly Groq errors."""

from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError, RateLimitError

import src.llm as llm


CHUNKS = [
    {"chunk_id": "a", "text": "Solar panels use sunlight.", "metadata": {"source": "solar.pdf", "page": 2}},
    {"chunk_id": "b", "text": "An inverter changes current.", "metadata": {"source": "power.pdf", "page": 4}},
    {"chunk_id": "c", "text": "Storage saves excess power.", "metadata": {"source": "battery.pdf", "page": 1}},
    {"chunk_id": "d", "text": "Fourth source text.", "metadata": {"source": "fourth.pdf", "page": 8}},
]


def test_system_prompt_prefers_detailed_supporting_chunk():
    """The prompt directs citations to detailed evidence pages."""
    assert "chunk that actually contains the supporting statement" in llm.SYSTEM_PROMPT
    assert "prefer the most detailed chunk over an abstract or summary" in llm.SYSTEM_PROMPT
    assert llm.NOT_FOUND in llm.SYSTEM_PROMPT


def _response(text: str):
    """Build the small response shape used by the OpenAI SDK wrapper."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _install_client_mock(monkeypatch, create):
    """Set required settings and replace the external OpenAI client."""
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key-never-print")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")
    completions = SimpleNamespace(create=create)
    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    ))


def _status_error(error_type, status: int):
    """Create an SDK status error with an in-memory request and response."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return error_type("provider error", response=response, body=None)


def test_exact_fallback_is_returned_unchanged(monkeypatch):
    """The required fallback bypasses all citation post-processing."""
    _install_client_mock(monkeypatch, lambda **_: _response(llm.NOT_FOUND))

    assert llm.generate_answer("off topic", CHUNKS) == llm.NOT_FOUND


def test_sources_are_appended_when_model_omits_citation(monkeypatch):
    """Uncited output receives source labels for at most three chunks."""
    _install_client_mock(monkeypatch, lambda **_: _response("Panels convert sunlight to electricity."))

    answer = llm.generate_answer("What do panels do?", CHUNKS)

    assert "Panels convert sunlight to electricity." in answer
    assert "Sources: [solar.pdf, p.2], [power.pdf, p.4], [battery.pdf, p.1]" in answer
    assert "fourth.pdf" not in answer


def test_numbered_context_citation_is_normalized(monkeypatch):
    """Numbered model citations map to the corresponding PDF filename/page."""
    _install_client_mock(monkeypatch, lambda **_: _response("An inverter changes current [2, p.4]."))

    assert llm.generate_answer("What does an inverter do?", CHUNKS) == (
        "An inverter changes current [power.pdf, p.4]."
    )


def test_fullwidth_model_citation_brackets_are_normalized(monkeypatch):
    """Groq's full-width citation brackets are converted to the required syntax."""
    _install_client_mock(monkeypatch, lambda **_: _response("A panel uses sunlight【solar.pdf, p.2】."))

    assert llm.generate_answer("What do panels use?", CHUNKS) == (
        "A panel uses sunlight[solar.pdf, p.2]."
    )


def test_rate_limit_retries_once_and_then_succeeds(monkeypatch):
    """A 429 is retried once with a short backoff."""
    calls = 0

    def create(**_):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _status_error(RateLimitError, 429)
        return _response("Panel output depends on sunlight.")

    _install_client_mock(monkeypatch, create)
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    answer = llm.generate_answer("What affects output?", CHUNKS)

    assert calls == 2
    assert sleeps == [1]
    assert "Sources:" in answer


def test_repeated_rate_limit_error_is_friendly(monkeypatch):
    """A second 429 returns a friendly message after the single retry."""
    calls = 0

    def create(**_):
        nonlocal calls
        calls += 1
        raise _status_error(RateLimitError, 429)

    _install_client_mock(monkeypatch, create)
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)

    with pytest.raises(llm.LLMError, match="Groq rate limit reached after retry"):
        llm.generate_answer("question", CHUNKS)

    assert calls == 2


def test_authentication_error_is_friendly_and_does_not_expose_key(monkeypatch):
    """Authentication failures become safe, concise UI messages."""
    _install_client_mock(
        monkeypatch,
        lambda **_: (_ for _ in ()).throw(_status_error(AuthenticationError, 401)),
    )

    with pytest.raises(llm.LLMError, match="Groq authentication failed") as error:
        llm.generate_answer("question", CHUNKS)

    assert "test-key-never-print" not in str(error.value)
