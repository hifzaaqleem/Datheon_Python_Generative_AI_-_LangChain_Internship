"""Document Q&A assistant: RAG over custom PDF/text files with page citations.

Pipeline:  load (PyPDFLoader / TextLoader) -> clean -> RecursiveCharacterTextSplitter
           -> embeddings -> ChromaDB (persistent, cosine) -> retrieve top-k
           -> relevance gate -> grounded prompt -> LLM -> answer + [file, page] citations

Why not RetrievalQA? RetrievalQA cannot apply a relevance-score cut-off (so it answers from
irrelevant chunks), does not give per-chunk page citations reliably, and was moved out of
`langchain` into `langchain-classic` in LangChain 1.0. The retrieve->gate->prompt->LLM flow
below is the same idea, explicit and version-stable.

Usage:
  python rag_pipeline.py --docs synthetic_data --embeddings hash --llm extractive --reset \
         --question "How many days of annual leave do employees get?"
  python rag_pipeline.py --docs ./my_pdfs --embeddings huggingface --llm ollama      # interactive
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:  # load API keys from a .env file next to this script (optional dependency: python-dotenv)
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import rag_core as core
from rag_core import NOT_FOUND, Answer, Chunk

log = logging.getLogger("rag")

DEFAULT_THRESHOLDS = {"hash": 0.05, "huggingface": 0.30}


class RAGError(RuntimeError):
    """User-facing pipeline error (bad config, missing key, LLM failure...)."""


# ----------------------------------------------------------------------------- config
@dataclass
class RAGConfig:
    db_dir: str = "./chroma_db"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    score_threshold: float | None = None      # None -> default for the embedding provider
    embeddings: str = "huggingface"           # huggingface | hash (offline test only)
    embedding_model: str | None = None
    llm: str = "ollama"                       # ollama | gemini | extractive (no LLM)
    llm_model: str | None = None
    min_chunk_chars: int = 15
    batch_size: int = 500                     # Chroma rejects very large single add() calls

    def __post_init__(self):
        if self.embeddings not in DEFAULT_THRESHOLDS:
            raise ValueError(f"Unknown embeddings provider '{self.embeddings}'")
        if self.llm not in ("ollama", "gemini", "extractive"):
            raise ValueError(f"Unknown llm provider '{self.llm}'")
        if self.score_threshold is None:
            self.score_threshold = DEFAULT_THRESHOLDS[self.embeddings]
        core.validate_params(self.chunk_size, self.chunk_overlap, self.top_k, self.score_threshold)


@dataclass
class IngestReport:
    files_loaded: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    chunks_added: int = 0


# ----------------------------------------------------------------------------- factories
class HashEmbeddings(Embeddings):
    """Offline deterministic embeddings for tests/CI (lexical, NOT semantic)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [core.hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return core.hash_embed(text)


def build_embeddings(cfg: RAGConfig) -> tuple[Embeddings, str]:
    if cfg.embeddings == "hash":
        return HashEmbeddings(), "hash"
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        raise RAGError("Install langchain-huggingface and sentence-transformers.") from e
    model = cfg.embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
    try:
        emb = HuggingFaceEmbeddings(model_name=model, encode_kwargs={"normalize_embeddings": True})
    except Exception as e:  # typically: no network on first run -> model cannot be downloaded
        raise RAGError(f"Could not load embedding model '{model}': {e}. "
                       "First run needs internet access (or use --embeddings hash for offline tests).") from e
    return emb, model


def build_llm(cfg: RAGConfig):
    """Returns a callable (system, user) -> str, or None for extractive mode."""
    if cfg.llm == "extractive":
        return None
    if cfg.llm == "gemini":
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            raise RAGError("GOOGLE_API_KEY is not set (free key from aistudio.google.com).")
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Gemini 3.5+ rejects/ignores custom sampling params, so temperature is deliberately not set.
        model = ChatGoogleGenerativeAI(model=cfg.llm_model or "gemini-3.5-flash-lite",
                                       google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    else:
        from langchain_ollama import ChatOllama
        model = ChatOllama(model=cfg.llm_model or "llama3.2", temperature=0)

    def call(system: str, user: str) -> str:
        content = model.invoke([("system", system), ("human", user)]).content
        if isinstance(content, list):  # some providers return a list of content blocks
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        return str(content).strip()

    return call


# ----------------------------------------------------------------------------- pipeline
class RAGPipeline:
    def __init__(self, cfg: RAGConfig | None = None):
        self.cfg = cfg or RAGConfig()
        self.embeddings, model_name = build_embeddings(self.cfg)
        # One collection per embedding model: mixing vector sizes in a collection crashes Chroma.
        self.collection = core.safe_collection_name("rag", model_name)
        self._llm = build_llm(self.cfg)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.cfg.chunk_size, chunk_overlap=self.cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""])
        self._open_store()

    # -- store ---------------------------------------------------------------------------
    def _open_store(self) -> None:
        self.store = Chroma(
            collection_name=self.collection,
            embedding_function=self.embeddings,
            persist_directory=self.cfg.db_dir,
            collection_metadata={"hnsw:space": "cosine"},  # makes distance = 1 - cosine similarity
        )

    def reset(self) -> None:
        self.store.delete_collection()
        self._open_store()

    def count(self) -> int:
        return len(self.store.get(include=[])["ids"])

    # -- loading -------------------------------------------------------------------------
    @staticmethod
    def _load_file(path: Path) -> list[Document]:
        """Returns one Document per PDF page (page is 1-based) or one per text file (page 0)."""
        if path.suffix.lower() == ".pdf":
            pages = PyPDFLoader(str(path)).load()
            return [Document(page_content=d.page_content, metadata={"page": int(d.metadata.get("page", i)) + 1})
                    for i, d in enumerate(pages)]
        for enc in ("utf-8-sig", "latin-1"):  # TextLoader raises RuntimeError (not UnicodeDecodeError)
            try:
                docs = TextLoader(str(path), encoding=enc).load()
                break
            except (UnicodeDecodeError, RuntimeError):
                continue
        else:
            raise ValueError("could not decode file")
        return [Document(page_content=d.page_content, metadata={"page": 0}) for d in docs]

    # -- ingestion -----------------------------------------------------------------------
    def ingest(self, paths: list[str | Path], reset: bool = False) -> IngestReport:
        if reset:
            self.reset()
        files, skipped = core.collect_files(paths)
        report = IngestReport(skipped=list(skipped))
        for f in files:
            try:
                docs = self._load_file(f)
            except Exception as e:  # corrupt / encrypted PDF, unreadable file, ...
                msg = f"failed to load: {type(e).__name__}: {e}"
                if "cryptography" in str(e).lower():
                    msg += " -> fix: pip install cryptography"
                elif "password" in str(e).lower() or "encrypted" in str(e).lower():
                    msg += " -> the PDF is password-protected; remove the password first"
                report.skipped.append((str(f), msg))
                continue
            pages = []
            for d in docs:
                text = core.clean_text(d.page_content)
                if text:
                    pages.append(Document(page_content=text, metadata=d.metadata))
            if not pages:
                report.skipped.append((str(f), "no extractable text (empty or scanned file - needs OCR)"))
                continue
            # split page by page so each chunk keeps the right page number
            chunks = [c for c in self._splitter.split_documents(pages)
                      if len(c.page_content.strip()) >= self.cfg.min_chunk_chars]
            if not chunks:
                report.skipped.append((str(f), "no usable chunks after splitting"))
                continue
            src_path = str(f.resolve())
            ids, out = [], []
            for i, c in enumerate(chunks):
                page = int(c.metadata.get("page", 0))
                ids.append(core.make_chunk_id(src_path, page, i, c.page_content))
                out.append(Document(page_content=c.page_content,
                                    metadata={"source": f.name, "source_path": src_path, "page": page, "chunk": i}))
            try:  # remove stale chunks of an earlier version of this same file
                self.store.delete(where={"source_path": src_path})
            except Exception as e:
                log.debug("no stale chunks removed for %s: %s", f, e)
            for s in range(0, len(out), self.cfg.batch_size):
                self.store.add_documents(out[s:s + self.cfg.batch_size], ids=ids[s:s + self.cfg.batch_size])
            report.files_loaded.append(f.name)
            report.chunks_added += len(out)
        return report

    # -- question answering ----------------------------------------------------------------
    def retrieve(self, question: str) -> list[Chunk]:
        raw = self.store.similarity_search_with_score(question, k=self.cfg.top_k)
        return [Chunk(text=d.page_content, source=d.metadata.get("source", "unknown"),
                      page=int(d.metadata.get("page", 0)), chunk=int(d.metadata.get("chunk", 0)),
                      score=core.distance_to_similarity(dist))
                for d, dist in raw]

    def ask(self, question: str) -> Answer:
        q = core.validate_question(question)
        if not self.store.get(limit=1, include=[])["ids"]:
            return Answer(answer="No documents are loaded yet. Ingest documents first.", grounded=False)
        hits = self.retrieve(q)
        kept = core.select_chunks(hits, self.cfg.score_threshold, self.cfg.top_k)
        if not kept:
            return Answer(answer=NOT_FOUND, grounded=False, chunks=hits)

        if self._llm is None:
            text, used = core.extractive_answer(q, kept)
        else:
            try:
                text = self._llm(core.SYSTEM_PROMPT, core.build_user_prompt(q, kept))
            except Exception as e:
                raise RAGError(f"LLM call failed: {type(e).__name__}: {e}") from e
            used = core.parse_used_sources(text, len(kept))
            if not used and not core.is_refusal(text):
                used = list(range(len(kept)))   # model forgot tags: cite everything it was shown

        if core.is_refusal(text):
            return Answer(answer=NOT_FOUND, grounded=False, chunks=kept)
        used_chunks = [kept[i] for i in used]
        return Answer(answer=text, grounded=True, citations=core.dedupe_citations(used_chunks),
                      chunks=kept, used=used_chunks)


# ----------------------------------------------------------------------------- CLI
def _print(ans: Answer) -> None:
    print(f"\nAnswer: {ans.answer}")
    if ans.citations:
        print("Sources:")
        for c in ans.citations:
            print(f"  - {c}")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RAG document Q&A with page citations")
    ap.add_argument("--docs", nargs="*", default=[], help="files and/or directories (.pdf .txt .md)")
    ap.add_argument("--question", "-q", help="ask one question and exit (otherwise interactive)")
    ap.add_argument("--db-dir", default="./chroma_db")
    ap.add_argument("--reset", action="store_true", help="wipe the vector store before ingesting")
    ap.add_argument("--embeddings", default="huggingface", choices=list(DEFAULT_THRESHOLDS))
    ap.add_argument("--llm", default="ollama", choices=["ollama", "gemini", "extractive"])
    ap.add_argument("--llm-model"); ap.add_argument("--embedding-model")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=120)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=None, help="min cosine similarity (default depends on embeddings)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.WARNING)

    try:
        cfg = RAGConfig(db_dir=a.db_dir, chunk_size=a.chunk_size, chunk_overlap=a.chunk_overlap, top_k=a.top_k,
                        score_threshold=a.threshold, embeddings=a.embeddings, embedding_model=a.embedding_model,
                        llm=a.llm, llm_model=a.llm_model)
        rag = RAGPipeline(cfg)
        if a.docs:
            rep = rag.ingest(a.docs, reset=a.reset)
            print(f"Loaded {len(rep.files_loaded)} file(s), {rep.chunks_added} chunk(s).")
            for path, why in rep.skipped:
                print(f"  skipped {path}: {why}")
        elif a.reset:
            rag.reset()
        if a.question:
            _print(rag.ask(a.question))
            return 0
        print("Ask a question (blank line or Ctrl-D to quit).")
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if not q:
                break
            try:
                _print(rag.ask(q))
            except ValueError as e:
                print(f"  {e}")
        return 0
    except (RAGError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())