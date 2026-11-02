"""Persistent dense index construction and loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_chunks"


class SentenceTransformerEmbedder:
    """Small adapter exposing SentenceTransformer embeddings to Chroma."""

    def __init__(self, model: SentenceTransformer) -> None:
        self.model = model

    def name(self) -> str:
        """Identify this custom Chroma embedding function."""
        return "sentence-transformers-all-MiniLM-L6-v2"

    def get_config(self) -> dict[str, str]:
        """Return the small configuration Chroma persists for this embedder."""
        return {"model_name": DEFAULT_MODEL}

    @classmethod
    def build_from_config(cls, config: dict[str, str]) -> "SentenceTransformerEmbedder":
        """Reconstruct the embedding function from Chroma configuration."""
        return cls(load_embedding_model(config.get("model_name", DEFAULT_MODEL)))

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed input texts as normalized vectors."""
        vectors = self.model.encode(input, normalize_embeddings=True)
        return vectors.tolist() if hasattr(vectors, "tolist") else vectors

    def embed_query(self, input: list[str]) -> list[list[float]]:
        """Embed query texts using the same normalized model vectors."""
        return self(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """Embed document texts using the same normalized model vectors."""
        return self(input)


def load_embedding_model(model_name: str = DEFAULT_MODEL) -> Any:
    """Load the configured sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def get_collection(
    persist_dir: str | Path = DEFAULT_CHROMA_PATH,
    model: Any = None,
) -> Any:
    """Open the persistent Chroma collection using the selected embedder."""
    model = model or load_embedding_model()
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=SentenceTransformerEmbedder(model),
        metadata={"hnsw:space": "cosine"},
    )


def build_dense_index(
    chunks: list[dict[str, Any]],
    persist_dir: str | Path = DEFAULT_CHROMA_PATH,
    model: Any = None,
) -> int:
    """Add missing chunks to the persistent dense index and return its count."""
    if not chunks:
        return 0
    collection = get_collection(persist_dir, model)
    current = collection.get(include=["documents", "metadatas"])
    existing = {
        chunk_id: (text, metadata)
        for chunk_id, text, metadata in zip(current["ids"], current["documents"], current["metadatas"])
    }
    requested_ids = {chunk["chunk_id"] for chunk in chunks}
    stale_ids = set(existing) - requested_ids
    if stale_ids:
        collection.delete(ids=sorted(stale_ids))
    new_chunks = [
        chunk for chunk in chunks
        if existing.get(chunk["chunk_id"]) != (chunk["text"], chunk["metadata"])
    ]
    if new_chunks:
        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in new_chunks],
            documents=[chunk["text"] for chunk in new_chunks],
            metadatas=[chunk["metadata"] for chunk in new_chunks],
        )
    return int(collection.count())

