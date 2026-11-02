# RAG-based PDF Q&A Chatbot

Streamlit app for asking questions across multiple PDFs with filename and page citations. It combines MiniLM dense retrieval with BM25 keyword retrieval and Reciprocal Rank Fusion (RRF).

## Architecture

**Indexing:** PyMuPDF extracts text page by page. LangChain's recursive splitter creates 800-character chunks with 150-character overlap and source/page metadata. MiniLM embeddings go to persistent Chroma (`./chroma_db`); lowercase regex tokens and BM25 scores are persisted in `./bm25_index.pkl`.

**Questions:** Dense and BM25 each return ranked matches. RRF adds `1 / (60 + rank)` for each list and returns the top five chunks. The OpenAI-compatible LLM endpoint receives numbered passages with their PDF filename and page, answers only from those passages, and cites them as `[filename, p.X]`.

## Groq setup and run

Requires Python 3.11 or newer. Create a free Groq API key at [console.groq.com](https://console.groq.com/keys), then configure the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set LLM_API_KEY to your Groq key.
streamlit run app.py
```

The default model is `openai/gpt-oss-120b`. Other Groq model choices include `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`; set `LLM_MODEL` in `.env` to switch. Groq free-tier request and token limits vary by model and may change. A 429 response is retried once; if the retry is still rate limited, wait and try again.

In the sidebar, upload one or more PDFs and click **Build index**. Ask questions in chat and expand **Sources** to inspect retrieved passages. `python -m src.ingest` prints an ingestion preview. Scanned PDFs need OCR before they can be indexed.

## Add your own PDFs and evaluation questions

Upload PDFs in the app or place them in `data/`. Run `python scripts/eval_helper.py` to print each PDF's page number and the first 200 extracted characters from that page. Use that evidence to write your own `eval/eval_set.json` entries with `question`, `expected_source`, `expected_page`, and optionally `type` set to `keyword` or `paraphrase`. Do not use a question unless its answer and expected page are supported by the PDF.

Run `python eval/run_eval.py --split dev` and `python eval/run_eval.py --split test` to compare dense, BM25, and hybrid hit-rate@5 overall and by question type. `python eval/tune.py` sweeps retrieval settings on dev and reports its single selected configuration on test. The script warns if there are fewer than 100 indexed chunks or fewer than 15 evaluation questions; results below either threshold are not reliable. This retrieval evaluation makes no LLM calls.

## Evaluation

Run on 2026-11-02 over 747 indexed chunks from 11 PDFs and 40 paper questions (34 dev, 6 test; 21 keyword, 19 paraphrase). A hit means an expected source and accepted page appear within the first five retrieved chunks. Split-specific baseline results:

| Split | Retrieval | Overall hit-rate@5 | Keyword | Paraphrase |
|---|---|---:|---:|---:|
| Dev (n=34) | Dense | 73.5% | 72.2% | 75.0% |
| Dev (n=34) | BM25 | 82.4% | 83.3% | 81.2% |
| Dev (n=34) | Hybrid | 82.4% | 83.3% | 81.2% |
| Test (n=6) | Dense | 100.0% | 100.0% | 100.0% |
| Test (n=6) | BM25 | 83.3% | 100.0% | 66.7% |
| Test (n=6) | Hybrid | 83.3% | 100.0% | 66.7% |

The test set is only six questions, so its apparent 100% dense result is not reliable evidence. Hybrid did not beat the single methods on the held-out split; on dev it tied BM25 and trailed dense. `python eval/tune.py` runs independent dev sweeps and evaluates only its dev-selected winner once on test. In this run, the best dev hybrid hit-rate was 85.3% (cross-encoder reranking, tied with 1200/200 chunking); its held-out result was 83.3%, equal to BM25 and below dense. Mean reranker latency was 0.727 seconds per dev query (0.755 seconds on test). Do not treat these small-sample results as conclusive.

## Limitations

- ResNet PDF page 8 extraction flattens the superscript in “10³-layer” to “103-layer”; the rendered page is readable and its supporting sentence is verified.
- The 40-question set covers eight papers, with only six held-out test questions; results are uncertain and are not a broad benchmark.
- Every evaluation question targets evidence on one page; multi-page synthesis is not measured. The held-out set contains only six questions, so its rates remain highly uncertain.
- Cross-document questions (for example, “which papers use masking?”) may miss relevant papers because top-5 retrieval comes from one query.
- Add OCR for scanned PDFs.
- Add multi-turn memory for follow-up questions.
