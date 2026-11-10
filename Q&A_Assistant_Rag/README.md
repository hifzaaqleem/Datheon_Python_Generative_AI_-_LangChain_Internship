# Document Q&A Assistant (RAG) with page citations

Load PDF/TXT/MD files -> clean -> `RecursiveCharacterTextSplitter` -> embeddings -> ChromaDB (cosine)
-> top-k retrieval -> **relevance gate** -> grounded prompt -> LLM -> answer + `[file, page N]` citations.

## Quick start
```bash
pip install -r requirements.txt
python generate_synthetic_data.py                       # creates ./synthetic_data

# 1) Fully offline smoke test (no model download, no API key)
python rag_pipeline.py --docs synthetic_data --embeddings hash --llm extractive --reset \
       -q "How many days of paid annual leave does a full-time employee get?"

# 2) Real setup: local MiniLM embeddings + local Ollama model
python rag_pipeline.py --docs ./my_pdfs --reset --llm ollama   # interactive prompt

# Tests
pytest test_core.py -v                                    # pure logic, no heavy deps
pytest test_pipeline.py -v                                # full Chroma pipeline (offline mode)
GOOGLE_API_KEY=... pytest test_pipeline.py -v             # + injection / near-miss LLM tests (Gemini)
```

## Web UI (Streamlit)
```bash
streamlit run app.py          # or: uv run streamlit run app.py
```
Upload PDFs in the sidebar, click **Process**, then chat. Pick the LLM/embeddings in the sidebar
(`extractive` and `ollama` need no key). Answers show citations and an expandable "Retrieved passages" panel.
Not executed in the build environment (Streamlit not installed there): syntax-checked only.

## Free setup
Embeddings run locally for free. For the answer step pick one:
```bash
python rag_pipeline.py --docs ./pdfs --reset --llm ollama --llm-model llama3.2   # local, free
GOOGLE_API_KEY=... python rag_pipeline.py --docs ./pdfs --reset --llm gemini     # free API key
python rag_pipeline.py --docs ./pdfs --reset --llm extractive                     # no model at all
```
(The Gemini provider is untested here; model names and free limits change, use `--llm-model` to override.)

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit web UI (dark, gradient theme; styling in `.streamlit/config.toml`) |
| `rag_pipeline.py` | LangChain + Chroma pipeline and CLI |
| `rag_core.py` | Pure-Python logic (validation, ids, offline embeddings, gating, prompt, citations) |
| `generate_synthetic_data.py` | Builds the synthetic test data below |
| `synthetic_data/` | 5-page PDF, 2 text files, `test_questions.json`, `edge_cases/` |
| `test_core.py`, `test_pipeline.py` | Tests |

## Synthetic test data
* `nimbus_handbook.pdf`: fictional company handbook, 5 pages, one topic per page (page-citation test)
* `support_faq.txt`, `office_notes.txt`: non-paged docs (citation has no page); the second contains a **prompt-injection** line
* `test_questions.json`: 15 answerable (expected keyword + source + page), 4 unanswerable, 2 LLM-only (near-miss "Atlas-9", injection)
* `edge_cases/`: empty, whitespace-only, text-less PDF, corrupt PDF, `.docx`, Latin-1 text, UTF-8 BOM + emoji/CJK, NUL bytes, 20 000-char line

## Bug / risk audit (what is handled)
| # | Risk | Handling |
|---|---|---|
| 1 | PyPDFLoader pages are 0-indexed -> wrong citations | stored/displayed as `page + 1` |
| 2 | Text files have no page | page `0` -> cited as just the file name |
| 3 | Scanned / blank PDF yields no text | skipped with an "needs OCR" reason, not silently ignored |
| 4 | Corrupt / encrypted PDF aborts whole batch | per-file try/except, reported in `IngestReport.skipped` |
| 5 | `TextLoader` raises `RuntimeError` (not `UnicodeDecodeError`) on bad encoding | caught; falls back utf-8-sig -> latin-1 |
| 6 | NUL bytes break SQLite/Chroma | stripped in `clean_text` |
| 7 | `chunk_overlap >= chunk_size` crash deep in splitter | validated up front with clear message |
| 8 | Re-running ingestion duplicates chunks | deterministic SHA-1 ids + upsert |
| 9 | Edited file leaves stale chunks | old chunks for that path deleted before re-add |
| 10 | Chroma distance mistaken for similarity | collection uses cosine; `similarity = 1 - distance` |
| 11 | Answering from irrelevant chunks (hallucination) | score threshold; refusal sentence, LLM not called |
| 12 | Model uses outside knowledge | system prompt + refusal sentence; refusal strips citations |
| 13 | Prompt injection inside documents | context is delimited, declared untrusted, `</context>` stripped from chunks |
| 14 | Citing sources the answer didn't use | model must tag `[Source n]`; only those are cited (fallback: all shown) |
| 15 | Different embedding models share a DB -> dimension crash | collection name includes the model |
| 16 | Python `hash()` is salted per process | offline embeddings use md5 |
| 17 | All-zero vector gives NaN cosine | reserved dimension for empty text |
| 18 | Chroma batch-size limit on large PDFs | batched adds (500) |
| 19 | Empty / huge question, missing API key, LLM timeout, content-as-list | validated / clear `RAGError` / timeout+retries / flattened |
| 20 | `RetrievalQA` removed from `langchain` in 1.0 and can't threshold | explicit retrieve -> gate -> LLM flow |

## Known limitations (be aware)
* **Threshold is model-specific.** Defaults: 0.30 (MiniLM), 0.05 (hash). Tune with a few in-domain questions; log `ans.chunks[*].score`.
* **Offline/hash + extractive mode is lexical only** (no paraphrase, no reasoning). Measured: scores of weak answerable questions overlap unanswerable ones, so extractive mode adds a word-overlap check. It exists for tests/CI; use real embeddings + LLM in production.
* "Near-miss" questions (Atlas-9 vs Atlas-7) need a real LLM to refuse; extractive mode would answer from Atlas-7 text.
* Chunks never span pages (page numbers stay exact, but a sentence cut by a page break is split). Tables/columns in PDFs may extract in odd order. Scanned PDFs need OCR first.
* Citations use the file *name*; two files with the same name in different folders are ambiguous. Page = physical index, not printed page label.
* Only Chroma is implemented (FAISS can be swapped in at `_open_store`, but lacks metadata delete/upsert).

## Verification status
In the build environment (no network, no LangChain/Chroma installed) I ran `test_core.py` (36 checks pass, including a full simulated retrieve -> gate -> answer -> cite run over all 19 synthetic questions) and syntax-checked the rest. **`rag_pipeline.py` itself and `test_pipeline.py` have not been executed**; run `pytest test_pipeline.py` after installing requirements and report any failure, as Chroma/LangChain API details vary between versions.