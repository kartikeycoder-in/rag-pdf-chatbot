"""PDF text extraction and deterministic page-aware chunking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pymupdf as fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def load_pdf_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract non-empty page text, retaining the source filename and 1-based page."""
    path = Path(pdf_path)
    pages: list[dict[str, Any]] = []
    try:
        with fitz.open(path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append({"text": text, "metadata": {"source": path.name, "page": page_number}})
    except Exception as exc:
        logger.warning("Skipping unreadable PDF %s: %s", path, exc)
    if not pages:
        logger.warning("No extractable text found in %s; skipping", path)
    return pages


def chunk_pdf(pdf_path: str | Path, splitter: RecursiveCharacterTextSplitter | None = None) -> list[dict[str, Any]]:
    """Extract and split one PDF into chunks with stable source/page/chunk metadata."""
    splitter = splitter or RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks: list[dict[str, Any]] = []
    for page in load_pdf_pages(pdf_path):
        page_chunks = splitter.split_text(page["text"])
        for local_index, text in enumerate(page_chunks):
            metadata = {**page["metadata"], "chunk_id": f'{page["metadata"]["source"]}:p{page["metadata"]["page"]}:c{local_index}'}
            chunks.append({"chunk_id": metadata["chunk_id"], "text": text, "metadata": metadata})
    return chunks


def load_and_chunk_directory(data_dir: str | Path = "data") -> list[dict[str, Any]]:
    """Chunk every PDF in a directory in filename order."""
    directory = Path(data_dir)
    all_chunks: list[dict[str, Any]] = []
    for pdf_path in sorted(directory.glob("*.pdf")) if directory.exists() else []:
        all_chunks.extend(chunk_pdf(pdf_path))
    return all_chunks


def main() -> None:
    """Print a chunk count and up to three representative chunks from data/."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    chunks = load_and_chunk_directory()
    print(f"Loaded {len(chunks)} chunks from data/")
    for chunk in chunks[:3]:
        print(f"\nMetadata: {chunk['metadata']}\nText: {chunk['text'][:500]}")


if __name__ == "__main__":
    main()
