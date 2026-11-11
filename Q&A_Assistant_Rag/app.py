"""Streamlit UI for the RAG document Q&A assistant.

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import os
from pathlib import Path

import streamlit as st

try:  # load API keys from a .env file next to this script (optional dependency: python-dotenv)
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from rag_core import NOT_FOUND
from rag_pipeline import DEFAULT_THRESHOLDS, RAGConfig, RAGError, RAGPipeline

UPLOAD_DIR = Path("uploads")
SAMPLE_DIR = Path("synthetic_data")
DB_DIR = "./chroma_db"
SAMPLE_QUESTIONS = [
    "How many days of paid annual leave do employees get?",
    "What is the maximum payload of the Atlas-7 robot?",
    "How quickly must phishing emails be reported?",
    "What is the hotel nightly cap on business trips?",
    "Who won the 2018 FIFA World Cup?",
]

st.set_page_config(page_title="RAGify · Document Q&A", page_icon="🧠", layout="wide")

# ----------------------------------------------------------------------------- styling
st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.6rem; max-width: 1100px;}

.hero {
  position: relative; overflow: hidden; border-radius: 22px; padding: 34px 38px; margin-bottom: 18px;
  background: linear-gradient(135deg, #4f46e5 0%, #8b5cf6 45%, #ec4899 100%);
  box-shadow: 0 18px 50px rgba(99,102,241,.35); color: #fff;
}
.hero::after {content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px;
  border-radius:50%; background: radial-gradient(circle, rgba(255,255,255,.28), transparent 70%);}
.hero h1 {margin:0; font-size: 2.3rem; font-weight: 800; letter-spacing:-.5px; color:#fff;}
.hero p {margin:.5rem 0 1rem 0; font-size:1.05rem; opacity:.92; max-width:640px;}
.pill {display:inline-block; padding:5px 13px; margin:0 8px 6px 0; border-radius:999px; font-size:.78rem;
  background: rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.3); backdrop-filter: blur(4px);}

.stats {display:grid; grid-template-columns: repeat(3, 1fr); gap:14px; margin-bottom:18px;}
.stat {border-radius:16px; padding:16px 18px; background:#131a30; border:1px solid rgba(139,92,246,.25);
  transition: transform .15s ease, border-color .15s ease;}
.stat:hover {transform: translateY(-2px); border-color: rgba(236,72,153,.6);}
.stat .k {font-size:.74rem; text-transform:uppercase; letter-spacing:1.2px; color:#9aa4c7;}
.stat .v {font-size:1.45rem; font-weight:700; margin-top:2px;
  background: linear-gradient(90deg,#a5b4fc,#f0abfc); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}

.empty {border-radius:18px; padding:26px 30px; background: linear-gradient(160deg,#131a30,#0f1429);
  border:1px dashed rgba(139,92,246,.5); margin:10px 0 18px 0;}
.empty h3 {margin:0 0 10px 0; color:#e8ecf8;}
.step {display:flex; align-items:center; gap:12px; margin:9px 0; color:#c7cee8;}
.step b {display:inline-flex; width:28px; height:28px; border-radius:50%; align-items:center; justify-content:center;
  background: linear-gradient(135deg,#6366f1,#ec4899); color:#fff; font-size:.85rem;}

.chip {display:inline-block; margin:4px 8px 0 0; padding:5px 12px; border-radius:999px; font-size:.8rem;
  background: rgba(99,102,241,.16); border:1px solid rgba(129,140,248,.5); color:#c7d2fe;}
.chip::before {content:"📎 ";}
.notfound {border-left:4px solid #f59e0b; background: rgba(245,158,11,.1); padding:10px 14px; border-radius:8px;}
.gauge {height:7px; border-radius:6px; background:#1f2747; overflow:hidden; margin:4px 0 10px 0;}
.gauge > div {height:100%; background: linear-gradient(90deg,#6366f1,#ec4899);}
.small {font-size:.78rem; color:#9aa4c7;}

[data-testid="stSidebar"] {background: linear-gradient(180deg,#10162b,#0b1020); border-right:1px solid rgba(139,92,246,.2);}
[data-testid="stChatMessage"] {border-radius:16px; padding:12px 16px; border:1px solid rgba(139,92,246,.15);
  background: rgba(19,26,48,.7);}
.stButton>button {border-radius:12px; border:1px solid rgba(139,92,246,.45); transition: all .15s ease;}
.stButton>button:hover {border-color:#ec4899; transform: translateY(-1px); box-shadow:0 6px 18px rgba(139,92,246,.3);}
.stButton>button[kind="primary"] {background: linear-gradient(135deg,#6366f1,#ec4899); border:none; color:#fff;}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- pipeline cache
@st.cache_resource(show_spinner="Loading models (first run downloads the embedding model)...")
def get_pipeline(embeddings, embedding_model, llm, llm_model, chunk_size, overlap, top_k, threshold, _key_id,
                 ollama_url=None):
    cfg = RAGConfig(db_dir=DB_DIR, chunk_size=chunk_size, chunk_overlap=overlap, top_k=top_k,
                    score_threshold=threshold, embeddings=embeddings, embedding_model=embedding_model or None,
                    llm=llm, llm_model=llm_model or None, ollama_url=ollama_url or None)
    return RAGPipeline(cfg)


# ----------------------------------------------------------------------------- sidebar: settings
st.sidebar.markdown("## 🧠 RAGify")
st.sidebar.caption("Ask questions. Get answers with page citations.")
st.sidebar.markdown("### ⚙️ Model")
llm = st.sidebar.selectbox("Answer model (LLM)", ["extractive", "ollama", "gemini"],
                           help="extractive = no model, returns matching sentences. ollama = free, local.")
llm_model = st.sidebar.text_input("Model name (optional)", placeholder="e.g. llama3.2")

ollama_url = None
if llm == "ollama":
    ollama_url = st.sidebar.text_input("Ollama server URL", value="http://127.0.0.1:11434",
                                       help="Must be reachable from where this app runs. Test it in a browser.").strip()
key_env = {"gemini": "GOOGLE_API_KEY"}.get(llm)
key = ""
if key_env:
    key = st.sidebar.text_input(key_env, type="password",
                                value=os.getenv(key_env) or os.getenv("GEMINI_API_KEY", ""))
    if key:
        os.environ[key_env] = key

embeddings = st.sidebar.selectbox("Embeddings", ["huggingface", "hash"],
                                  help="huggingface = free, local, no key. hash = offline test only.")
with st.sidebar.expander("Advanced"):
    chunk_size = st.number_input("Chunk size", 100, 4000, 800, 50)
    overlap = st.number_input("Chunk overlap", 0, 1000, 120, 10)
    top_k = st.number_input("Chunks retrieved (top-k)", 1, 20, 4)
    threshold = st.slider("Min similarity", 0.0, 1.0, DEFAULT_THRESHOLDS[embeddings], 0.01,
                          key=f"thr_{embeddings}",
                          help="Chunks scoring below this are ignored; if none remain the app says it can't answer.")

try:
    rag = get_pipeline(embeddings, None, llm, llm_model.strip(), int(chunk_size), int(overlap),
                       int(top_k), float(threshold), bool(key), ollama_url)
except (RAGError, ValueError) as e:
    st.error(f"Setup problem: {e}")
    st.stop()
except Exception as e:  # missing optional package, etc.
    st.error(f"Could not start the pipeline: {type(e).__name__}: {e}")
    st.stop()

# ----------------------------------------------------------------------------- sidebar: documents
with st.sidebar:
    st.markdown("### 📚 Documents")
    uploads = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    wipe = st.checkbox("Clear existing documents first")
    c1, c2 = st.columns(2)
    process = c1.button("⚡ Process", type="primary", disabled=not uploads, use_container_width=True)
    sample = c2.button("🧪 Sample", disabled=not SAMPLE_DIR.exists(), use_container_width=True,
                       help="Loads the synthetic test data (run generate_synthetic_data.py first).")

    paths: list[Path] = []
    if process and uploads:
        UPLOAD_DIR.mkdir(exist_ok=True)
        for f in uploads:
            dest = UPLOAD_DIR / Path(f.name).name        # strip any path components
            dest.write_bytes(f.getbuffer())
            paths.append(dest)
    elif sample:
        paths = [SAMPLE_DIR / n for n in ("nimbus_handbook.pdf", "support_faq.txt", "office_notes.txt")]

    if paths:
        with st.spinner("Reading, chunking and embedding..."):
            try:
                rep = rag.ingest(paths, reset=wipe or sample)
            except Exception as e:
                st.error(f"Ingestion failed: {type(e).__name__}: {e}")
            else:
                if rep.files_loaded:
                    st.success(f"Loaded {len(rep.files_loaded)} file(s) → {rep.chunks_added} chunks")
                for p, why in rep.skipped:
                    st.warning(f"Skipped {Path(p).name}: {why}")

    b1, b2 = st.columns(2)
    if b1.button("🗑️ Reset DB", use_container_width=True):
        rag.reset()
        st.session_state.pop("messages", None)
        st.rerun()
    if b2.button("🧹 Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

# ----------------------------------------------------------------------------- hero + stats
n_chunks = rag.count()
st.markdown(
    """
<div class="hero">
  <h1>🧠 RAGify</h1>
  <p>Ask questions about your own PDFs and documents. Every answer is grounded in your files
     and comes with page-level citations.</p>
  <span class="pill">🔒 Answers only from your docs</span>
  <span class="pill">📎 Page citations</span>
  <span class="pill">🆓 Runs with free models</span>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown(
    f"""
<div class="stats">
  <div class="stat"><div class="k">Chunks indexed</div><div class="v">{n_chunks}</div></div>
  <div class="stat"><div class="k">Answer model</div><div class="v">{html.escape(llm)}</div></div>
  <div class="stat"><div class="k">Embeddings</div><div class="v">{html.escape(embeddings)}</div></div>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- chat helpers
if "messages" not in st.session_state:
    st.session_state.messages = []


def render(msg: dict) -> None:
    if msg["role"] == "assistant" and msg["content"] == NOT_FOUND:
        st.markdown(f'<div class="notfound">🔍 {html.escape(NOT_FOUND)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(msg["content"])
    if msg.get("citations"):
        chips = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in msg["citations"])
        st.markdown(chips, unsafe_allow_html=True)
    chunks = msg.get("chunks") or []
    used = msg.get("used") or []
    if chunks:
        best = max(0.0, min(1.0, max(c["score"] for c in chunks)))
        st.markdown(
            f'<div class="small">Best match similarity: {best:.2f}</div>'
            f'<div class="gauge"><div style="width:{best * 100:.0f}%"></div></div>',
            unsafe_allow_html=True,
        )

        def show(items):
            for c in items:
                where = f"{c['source']}, page {c['page']}" if c["page"] else c["source"]
                st.markdown(f"**{where}** · similarity `{c['score']:.2f}`")
                st.text(c["text"])

        if used:
            with st.expander(f"Passages used in the answer ({len(used)})"):
                show(used)
        with st.expander(f"All retrieved passages ({len(chunks)}) - search results, not all used"):
            show(chunks)


# ----------------------------------------------------------------------------- main area
if n_chunks == 0:
    st.markdown(
        """
<div class="empty">
  <h3>Get started in 3 steps</h3>
  <div class="step"><b>1</b> Upload PDF or text files in the sidebar (or click <i>🧪 Sample</i>)</div>
  <div class="step"><b>2</b> Press <i>⚡ Process</i> to read and index them</div>
  <div class="step"><b>3</b> Ask a question below and get a cited answer</div>
</div>
""",
        unsafe_allow_html=True,
    )
elif not st.session_state.messages:
    has_sample = False
    try:
        has_sample = bool(rag.store.get(where={"source": "nimbus_handbook.pdf"}, limit=1, include=[])["ids"])
    except Exception:
        pass
    if has_sample:
        st.markdown("##### ✨ Try a sample question")
        cols = st.columns(2)
        for i, q in enumerate(SAMPLE_QUESTIONS):
            if cols[i % 2].button(q, key=f"sq{i}", use_container_width=True):
                st.session_state.pending = q
                st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🧠"):
        render(m)

question = st.chat_input("Ask a question about your documents…")
if not question and st.session_state.get("pending"):
    question = st.session_state.pop("pending")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🧠"):
        try:
            with st.spinner("Searching your documents…"):
                ans = rag.ask(question)
            msg = {"role": "assistant", "content": ans.answer, "citations": ans.citations,
                   "chunks": [{"source": c.source, "page": c.page, "score": c.score, "text": c.text}
                              for c in ans.chunks],
                   "used": [{"source": c.source, "page": c.page, "score": c.score, "text": c.text}
                            for c in ans.used]}
        except (ValueError, RAGError) as e:
            msg = {"role": "assistant", "content": f"⚠️ {e}"}
        except Exception as e:
            msg = {"role": "assistant", "content": f"⚠️ Unexpected error: {type(e).__name__}: {e}"}
        render(msg)
    st.session_state.messages.append(msg)
