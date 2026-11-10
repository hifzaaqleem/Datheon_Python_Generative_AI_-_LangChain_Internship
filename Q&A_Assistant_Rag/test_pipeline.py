"""End-to-end tests with real LangChain + ChromaDB (offline: hash embeddings + extractive answers).
Run after `pip install -r requirements.txt`:  pytest test_pipeline.py -v
LLM-dependent tests are skipped unless GOOGLE_API_KEY is set."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("langchain_chroma")
pytest.importorskip("langchain_community")
pytest.importorskip("pypdf")

from rag_core import NOT_FOUND  # noqa: E402
from rag_pipeline import RAGConfig, RAGError, RAGPipeline  # noqa: E402

HERE = Path(__file__).parent
DATA = HERE / "synthetic_data"
Q = None


@pytest.fixture(scope="session", autouse=True)
def _data():
    if not (DATA / "test_questions.json").exists():
        subprocess.check_call([sys.executable, str(HERE / "generate_synthetic_data.py")])
    global Q
    Q = json.loads((DATA / "test_questions.json").read_text())


def make(tmp_path, **kw):
    cfg = dict(db_dir=str(tmp_path / "db"), embeddings="hash", llm="extractive", chunk_size=300, chunk_overlap=50)
    cfg.update(kw)
    return RAGPipeline(RAGConfig(**cfg))


@pytest.fixture()
def rag(tmp_path):
    r = make(tmp_path)
    r.ingest([DATA / "nimbus_handbook.pdf", DATA / "support_faq.txt", DATA / "office_notes.txt"], reset=True)
    return r


def test_answerable_questions_have_correct_answer_and_page_citation(rag):
    for t in Q["answerable"]:
        a = rag.ask(t["question"])
        assert a.grounded, t["id"]
        assert any(k in a.answer.lower() for k in t["expected_keywords_any"]), (t["id"], a.answer)
        exp = t["expected_source"] + (f", page {t['expected_page']}" if t["expected_page"] else "")
        assert exp in a.citations, (t["id"], a.citations)


def test_unanswerable_questions_are_refused_without_citations(rag):
    for t in Q["unanswerable"]:
        a = rag.ask(t["question"])
        assert a.answer == NOT_FOUND and not a.citations and not a.grounded, t["id"]


def test_edge_case_files_do_not_crash_ingestion(tmp_path):
    r = make(tmp_path)
    rep = r.ingest([DATA / "edge_cases"], reset=True)
    loaded = set(rep.files_loaded)
    skipped = {Path(p).name: why for p, why in rep.skipped}
    assert {"latin1_notes.txt", "unicode_bom.txt", "nul_bytes.txt", "long_line.txt"} <= loaded
    for bad in ("empty.txt", "whitespace_only.txt", "blank_page.pdf", "corrupt.pdf", "report.docx"):
        assert bad in skipped, bad
    assert r.count() > 0
    # Latin-1 file must be decoded correctly (accents intact) and be retrievable
    assert "10h" in r.ask("Reunião semanal quartas-feiras").answer


def test_reingest_is_idempotent_and_reset_works(rag):
    n = rag.count()
    rag.ingest([DATA / "nimbus_handbook.pdf"])
    rag.ingest([DATA / "nimbus_handbook.pdf"])
    assert rag.count() == n
    rag.reset()
    assert rag.count() == 0
    assert "No documents" in rag.ask("anything at all?").answer


def test_persistence_across_instances(tmp_path):
    a = make(tmp_path)
    a.ingest([DATA / "support_faq.txt"], reset=True)
    b = make(tmp_path)                       # new object, same db_dir (tests the hash() salt bug too)
    assert "45" in b.ask("Within how many days can products be returned?").answer


def test_modified_file_replaces_old_chunks(tmp_path):
    f = tmp_path / "policy.txt"
    f.write_text("The parking fee is 10 euros per day for visitors.", encoding="utf-8")
    r = make(tmp_path)
    r.ingest([f], reset=True)
    f.write_text("The parking fee is 99 euros per day for visitors.", encoding="utf-8")
    r.ingest([f])
    ans = r.ask("What is the parking fee for visitors?").answer
    assert "99" in ans and "10 euros" not in ans


def test_bad_inputs(tmp_path, rag):
    with pytest.raises(ValueError):
        rag.ask("   ")
    with pytest.raises(ValueError):
        RAGConfig(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        RAGConfig(embeddings="nonsense")
    rep = rag.ingest(["/definitely/not/here.pdf"])
    assert rep.skipped and rep.chunks_added == 0


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="needs GOOGLE_API_KEY")
def test_llm_refuses_near_miss_and_ignores_injection(tmp_path):
    r = make(tmp_path, embeddings="huggingface", llm="gemini", chunk_size=800, chunk_overlap=120)
    r.ingest([DATA / "nimbus_handbook.pdf", DATA / "office_notes.txt"], reset=True)
    for t in Q["llm_only"]:
        a = r.ask(t["question"])
        if t.get("must_equal_refusal"):
            assert a.answer == NOT_FOUND, t["id"]
        for bad in t.get("must_not_contain", []):
            assert bad not in a.answer.lower(), t["id"]
        if "expected_keywords_any" in t:
            assert any(k in a.answer.lower() for k in t["expected_keywords_any"]), t["id"]


def test_missing_api_key_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RAGError, match="GOOGLE_API_KEY"):
        RAGPipeline(RAGConfig(db_dir=str(tmp_path / "db"), embeddings="hash", llm="gemini"))