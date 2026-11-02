"""Streamlit interface for multi-PDF question answering."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.indexer import build_dense_index, get_collection, load_embedding_model
from src.ingest import load_and_chunk_directory
from src.llm import LLMError
from src.pipeline import answer
from src.retriever import build_bm25_index, hybrid_search

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
BM25_FILE = ROOT / "bm25_index.pkl"

st.set_page_config(page_title="PDF Q&A", page_icon="📄")
load_dotenv(ROOT / ".env")


@st.cache_resource
def cached_embedding_model():
    """Cache the embedding model for this Streamlit process."""
    return load_embedding_model()


def _index_status() -> tuple[int, int]:
    """Return dense chunk and document counts without creating an index."""
    if not CHROMA_DIR.exists():
        return 0, 0
    try:
        collection = get_collection(CHROMA_DIR, cached_embedding_model())
        count = int(collection.count())
        metadata = collection.get(include=["metadatas"])["metadatas"]
        docs = {item.get("source") for item in metadata if item}
        return count, len(docs)
    except Exception:
        return 0, 0


st.title("📄 RAG-based PDF Q&A")
st.caption("Upload PDFs, build a hybrid search index, and ask questions with page citations.")

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if st.button("Build index", type="primary", disabled=not uploads):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for uploaded in uploads or []:
            (DATA_DIR / Path(uploaded.name).name).write_bytes(uploaded.getvalue())
        progress = st.progress(0, text="Extracting and chunking PDFs…")
        chunks = load_and_chunk_directory(DATA_DIR)
        progress.progress(30, text=f"Prepared {len(chunks)} chunks. Building dense index…")
        if chunks:
            model = cached_embedding_model()
            dense_count = build_dense_index(chunks, CHROMA_DIR, model)
            progress.progress(75, text="Building BM25 keyword index…")
            build_bm25_index(chunks, BM25_FILE)
            progress.progress(100, text="Index ready")
            docs = len({chunk["metadata"]["source"] for chunk in chunks})
            st.success(f"Indexed {docs} documents and {len(chunks)} chunks (dense store total: {dense_count}).")
        else:
            progress.empty()
            st.error("No extractable PDF text found. Scanned PDFs need OCR before indexing.")

    indexed_chunks, indexed_docs = _index_status()
    st.metric("Documents indexed", indexed_docs)
    st.metric("Chunks indexed", indexed_chunks)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    meta = source["metadata"]
                    st.markdown(f"**{meta['source']} — page {meta['page']}**")
                    st.write(source["text"])

question = st.chat_input("Ask a question about your PDFs")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not indexed_chunks or not BM25_FILE.exists():
            st.error("No index exists yet. Upload PDFs and click Build index in the sidebar.")
        elif not os.getenv("LLM_API_KEY"):
            st.error("LLM_API_KEY is missing. Add your Groq API key to .env to enable answers.")
        else:
            try:
                result = answer(
                    question,
                    search_fn=lambda query: hybrid_search(
                        query, CHROMA_DIR, BM25_FILE, cached_embedding_model()
                    ),
                )
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources"):
                        for source in result["sources"]:
                            meta = source["metadata"]
                            st.markdown(f"**{meta['source']} — page {meta['page']}**")
                            st.write(source["text"])
                st.session_state.messages.append({
                    "role": "assistant", "content": result["answer"], "sources": result["sources"]
                })
            except LLMError as exc:
                st.error(str(exc))
            except Exception:
                st.error("The language model request failed. Check your Groq settings and try again.")
