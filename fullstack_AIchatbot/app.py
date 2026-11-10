"""
✨ Nova - Streamlit + LangChain + Gemini chatbot
- Streaming token output (st.write_stream)
- Conversation memory (st.session_state, windowed)
- Optional RAG over uploaded .txt / .md / .pdf files (local TF-IDF retriever)
- API key is read from your .env file (GOOGLE_API_KEY or GEMINI_API_KEY)

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

try:  # preferred splitter; falls back to a built-in one if the package is missing
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover
    RecursiveCharacterTextSplitter = None

st.set_page_config(page_title="Nova · AI Chat", page_icon="✨", layout="centered")

# --------------------------------------------------------------------------- #
# Config (system prompt is fixed here and never shown in the UI)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are Nova, a friendly, sharp and concise AI assistant. "
    "Use clear formatting (short paragraphs, lists or code blocks when helpful). "
    "If context from the user's documents is provided, prefer it and mention it; "
    "if the context does not contain the answer, say so instead of guessing."
)
MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash"]

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}\n\n{context_block}"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

SUGGESTIONS = [
    ("🧠", "Explain RAG", "Explain Retrieval-Augmented Generation in simple terms with an example."),
    ("🐍", "Write code", "Write a Python function that removes duplicates from a list while keeping order."),
    ("📅", "Plan my week", "Create a balanced 5-day study plan for learning generative AI."),
    ("💡", "Brainstorm", "Give me 5 creative project ideas using LLMs and Streamlit."),
]


# --------------------------------------------------------------------------- #
# .env loading (python-dotenv if installed, tiny built-in parser otherwise)
# --------------------------------------------------------------------------- #
def load_env() -> None:
    candidates = [Path(__file__).resolve().parent / ".env", Path.cwd() / ".env"]
    try:
        from dotenv import load_dotenv

        for p in candidates:
            if p.exists():
                load_dotenv(p, override=False)
        return
    except ImportError:
        pass
    for p in candidates:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def get_api_key() -> str:
    return (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()


load_env()


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap');

html, body, [class*="css"], .stMarkdown, button, input, textarea {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}
.stApp {
    background:
        radial-gradient(900px 500px at 8% -8%, rgba(139, 92, 246, .38), transparent 60%),
        radial-gradient(800px 480px at 100% 0%, rgba(14, 165, 233, .30), transparent 55%),
        radial-gradient(700px 500px at 50% 110%, rgba(244, 114, 182, .18), transparent 60%),
        #0b0d17;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 860px; }

/* ---------- Hero ---------- */
.hero { text-align: center; padding: 1.2rem 0 .4rem 0; }
.hero .logo {
    width: 68px; height: 68px; margin: 0 auto .6rem auto; border-radius: 22px;
    display: flex; align-items: center; justify-content: center; font-size: 2rem;
    background: linear-gradient(135deg, #8b5cf6, #38bdf8 60%, #f472b6);
    box-shadow: 0 10px 40px rgba(139, 92, 246, .55);
    animation: float 4s ease-in-out infinite;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 3rem; margin: 0; padding: 0;
    background: linear-gradient(90deg, #c4b5fd, #38bdf8, #f9a8d4, #c4b5fd);
    background-size: 250% auto; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; animation: shine 7s linear infinite;
}
.hero p { color: #aab1d6; margin: .35rem 0 .9rem 0; font-size: 1.05rem; }
.chips { display: flex; gap: .5rem; justify-content: center; flex-wrap: wrap; }
.chip {
    padding: .28rem .8rem; border-radius: 999px; font-size: .78rem; color: #dfe3ff;
    background: rgba(255, 255, 255, .06); border: 1px solid rgba(255, 255, 255, .12);
    backdrop-filter: blur(6px);
}
.chip.live::before {
    content: ""; display: inline-block; width: 8px; height: 8px; margin-right: 6px; border-radius: 50%;
    background: #34d399; box-shadow: 0 0 10px #34d399; animation: pulse 1.8s infinite;
}
@keyframes shine { to { background-position: 250% center; } }
@keyframes float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-7px); } }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }

/* ---------- Chat bubbles ---------- */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, .045); border: 1px solid rgba(255, 255, 255, .09);
    border-radius: 20px; padding: 1rem 1.2rem; margin-bottom: .8rem;
    backdrop-filter: blur(10px); box-shadow: 0 8px 28px rgba(0, 0, 0, .25);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(139, 92, 246, .30), rgba(56, 189, 248, .20));
    border-color: rgba(167, 139, 250, .45);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid #38bdf8;
}

/* ---------- Chat input ---------- */
[data-testid="stBottom"] > div { background: transparent; }
[data-testid="stChatInput"] {
    border-radius: 20px; border: 1px solid rgba(167, 139, 250, .55);
    background: rgba(20, 23, 51, .85); box-shadow: 0 0 28px rgba(139, 92, 246, .28);
}
[data-testid="stChatInput"] textarea { color: #f1f3ff; }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 14px; border: 1px solid rgba(167, 139, 250, .35);
    background: rgba(255, 255, 255, .05); color: #eef0ff; transition: all .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #a78bfa; transform: translateY(-2px);
    background: linear-gradient(135deg, rgba(139, 92, 246, .35), rgba(56, 189, 248, .25));
    box-shadow: 0 10px 26px rgba(139, 92, 246, .35); color: #fff;
}
[class*="st-key-sugg_"] button { width: 100%; min-height: 84px; text-align: left; white-space: normal; padding: .8rem 1rem; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #171a3c 0%, #0d0f22 100%);
    border-right: 1px solid rgba(255, 255, 255, .08);
}
.brand { display: flex; align-items: center; gap: .7rem; margin-bottom: .4rem; }
.brand .b-logo {
    width: 42px; height: 42px; border-radius: 14px; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #8b5cf6, #38bdf8); font-size: 1.3rem;
}
.brand .b-name { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.3rem; color: #fff; line-height: 1.1; }
.brand .b-sub { font-size: .75rem; color: #98a0cc; }
.stat-row { display: flex; gap: .6rem; margin: .4rem 0 .2rem 0; }
.stat {
    flex: 1; text-align: center; padding: .6rem .3rem; border-radius: 14px;
    background: rgba(255, 255, 255, .05); border: 1px solid rgba(255, 255, 255, .09);
}
.stat b { display: block; font-size: 1.3rem; color: #c4b5fd; font-family: 'Space Grotesk', sans-serif; }
.stat span { font-size: .7rem; color: #98a0cc; text-transform: uppercase; letter-spacing: .06em; }
.side-title { font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: #8e96c6; margin: 1rem 0 .3rem 0; }

/* ---------- Misc ---------- */
[data-testid="stExpander"] { border-radius: 14px; border: 1px solid rgba(255, 255, 255, .09); background: rgba(255, 255, 255, .03); }
[data-testid="stFileUploader"] section { border-radius: 14px; border: 1px dashed rgba(167, 139, 250, .5); background: rgba(139, 92, 246, .06); }
.keybox {
    padding: 1rem 1.2rem; border-radius: 16px; margin: 1rem 0;
    background: rgba(244, 63, 94, .10); border: 1px solid rgba(244, 63, 94, .45); color: #ffd7de;
}
.keybox code { background: rgba(255, 255, 255, .1); padding: 1px 6px; border-radius: 6px; }
</style>
"""


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    defaults = {
        "session_id": uuid.uuid4().hex[:8],
        "messages": [],       # list[dict]: role, content, optional sources
        "kb": None,           # KnowledgeBase | None
        "kb_sig": None,       # signature of uploaded files + chunk settings
        "pending_prompt": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_chat() -> None:
    st.session_state.messages = []
    st.session_state.pending_prompt = None
    st.session_state.session_id = uuid.uuid4().hex[:8]


def set_pending(text: str) -> None:
    st.session_state.pending_prompt = text


# --------------------------------------------------------------------------- #
# RAG: lightweight TF-IDF knowledge base
# --------------------------------------------------------------------------- #
@dataclass
class KnowledgeBase:
    chunks: list[str]
    sources: list[str]
    vectorizer: TfidfVectorizer = field(repr=False)
    matrix: object = field(repr=False)

    def search(self, query: str, k: int = 4, min_score: float = 0.05):
        if not self.chunks:
            return []
        q_vec = self.vectorizer.transform([query])
        scores = linear_kernel(q_vec, self.matrix).ravel()
        top = scores.argsort()[::-1][:k]
        return [(self.chunks[i], self.sources[i], float(scores[i])) for i in top if scores[i] >= min_score]


def read_uploaded_file(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if RecursiveCharacterTextSplitter is not None:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ).split_text(text)
    step = max(1, chunk_size - chunk_overlap)
    return [text[i:i + chunk_size] for i in range(0, len(text), step) if text[i:i + chunk_size].strip()]


def build_knowledge_base(files, chunk_size: int, chunk_overlap: int):
    chunks, sources, warnings = [], [], []
    for f in files:
        try:
            text = read_uploaded_file(f).strip()
        except Exception as exc:
            warnings.append(f"Could not read **{f.name}**: {exc}")
            continue
        if not text:
            warnings.append(f"**{f.name}** contained no extractable text (scanned PDF?).")
            continue
        for piece in split_text(text, chunk_size, chunk_overlap):
            chunks.append(piece)
            sources.append(f.name)

    if not chunks:
        return None, warnings

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform(chunks)
    except ValueError:
        warnings.append("Documents contained no usable vocabulary.")
        return None, warnings
    return KnowledgeBase(chunks, sources, vectorizer, matrix), warnings


# --------------------------------------------------------------------------- #
# LLM + chain
# --------------------------------------------------------------------------- #
def build_llm(model: str, api_key: str, temperature: float):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature)


def to_lc_messages(history: list[dict], window: int) -> list[BaseMessage]:
    trimmed = history[-window:] if window > 0 else []
    while trimmed and trimmed[0]["role"] != "user":
        trimmed = trimmed[1:]
    return [
        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
        for m in trimmed
    ]


def format_context(hits) -> str:
    if not hits:
        return ""
    body = "\n\n".join(f"[{i}] (source: {src})\n{text}" for i, (text, src, _) in enumerate(hits, 1))
    return f"Context from the user's documents:\n{body}"


def stream_answer(chain, payload: dict):
    for token in chain.stream(payload):
        if token:
            yield token


# --------------------------------------------------------------------------- #
# UI pieces
# --------------------------------------------------------------------------- #
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="b-logo">✨</div>'
            '<div><div class="b-name">Nova</div><div class="b-sub">Powered by Gemini</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-title">Model</div>', unsafe_allow_html=True)
        model = st.selectbox("Model", MODELS, label_visibility="collapsed")

        st.markdown('<div class="side-title">📚 Knowledge (RAG)</div>', unsafe_allow_html=True)
        files = st.file_uploader("Upload .txt, .md or .pdf", type=["txt", "md", "pdf"],
                                 accept_multiple_files=True, label_visibility="collapsed")

        with st.expander("Advanced"):
            temperature = st.slider("Creativity", 0.0, 1.0, 0.4, 0.05)
            window = st.slider("Memory (messages)", 2, 40, 12, 2)
            chunk_size = st.number_input("Chunk size", 200, 4000, 800, 100)
            top_k = st.number_input("Top-k passages", 1, 10, 4, 1)
        chunk_overlap = min(150, int(chunk_size) // 4)

        sig = (tuple((f.name, f.size) for f in files), int(chunk_size))
        if sig != st.session_state.kb_sig:
            st.session_state.kb_sig = sig
            if files:
                with st.spinner("Indexing documents…"):
                    kb, warns = build_knowledge_base(files, int(chunk_size), chunk_overlap)
                st.session_state.kb = kb
                for w in warns:
                    st.warning(w)
            else:
                st.session_state.kb = None

        kb = st.session_state.kb
        n_chunks = len(kb.chunks) if kb else 0
        n_msgs = len(st.session_state.messages)
        st.markdown(
            f'<div class="stat-row"><div class="stat"><b>{n_msgs}</b><span>Messages</span></div>'
            f'<div class="stat"><b>{n_chunks}</b><span>Chunks</span></div></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Session `{st.session_state.session_id}`")

        st.button("🗑️  New chat", on_click=reset_chat, use_container_width=True)
        if st.session_state.messages:
            transcript = "\n\n".join(
                f"**{m['role'].title()}**: {m['content']}" for m in st.session_state.messages
            )
            st.download_button(
                "⬇️  Download chat", data=transcript,
                file_name=f"chat_{datetime.now():%Y%m%d_%H%M%S}.md",
                mime="text/markdown", use_container_width=True,
            )

    return dict(model=model, temperature=temperature, window=window, top_k=int(top_k), docs=n_chunks > 0)


def render_hero(settings: dict, key_ok: bool) -> None:
    docs = "Docs: on" if settings["docs"] else "Docs: off"
    status = '<span class="chip live">Online</span>' if key_ok else '<span class="chip">Key missing</span>'
    st.markdown(
        f"""
        <div class="hero">
          <div class="logo">✨</div>
          <h1>Nova</h1>
          <p>Your streaming AI assistant · remembers the conversation · chats with your documents</p>
          <div class="chips">{status}
            <span class="chip">🤖 {settings['model']}</span>
            <span class="chip">🧠 Memory</span>
            <span class="chip">📚 {docs}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_suggestions(disabled: bool) -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (icon, title, text) in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            st.button(f"{icon}  {title}\n\n{text}", key=f"sugg_{i}", on_click=set_pending,
                      args=(text,), disabled=disabled, use_container_width=True)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**[{i}] {s['source']}** · relevance {s['score']:.2f}")
            st.caption(s["text"][:400] + ("…" if len(s["text"]) > 400 else ""))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)
    settings = render_sidebar()

    api_key = get_api_key()
    key_ok = bool(api_key)
    render_hero(settings, key_ok)

    if not key_ok:
        st.markdown(
            '<div class="keybox">🔑 <b>No Gemini API key found.</b> Add <code>GOOGLE_API_KEY=your-key</code> '
            "to the <code>.env</code> file next to <code>app.py</code>, then restart the app.</div>",
            unsafe_allow_html=True,
        )

    # Welcome screen with starter prompts
    if not st.session_state.messages and not st.session_state.pending_prompt:
        render_suggestions(disabled=not key_ok)

    # Replay history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "✨"):
            st.markdown(msg["content"])
            render_sources(msg.get("sources", []))

    prompt = st.chat_input("Message Nova…", disabled=not key_ok)
    if not prompt:
        prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    if not prompt:
        return

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    history = to_lc_messages(st.session_state.messages, settings["window"])
    st.session_state.messages.append({"role": "user", "content": prompt})

    hits = st.session_state.kb.search(prompt, k=settings["top_k"]) if st.session_state.kb else []
    sources = [{"source": src, "text": text, "score": score} for text, src, score in hits]

    with st.chat_message("assistant", avatar="✨"):
        try:
            llm = build_llm(settings["model"], api_key, settings["temperature"])
            chain = PROMPT | llm | StrOutputParser()
            payload = {
                "system_prompt": SYSTEM_PROMPT,
                "context_block": format_context(hits),
                "history": history,
                "question": prompt,
            }
            answer = st.write_stream(stream_answer(chain, payload))
            render_sources(sources)
        except ImportError as exc:
            st.session_state.messages.pop()
            st.error(f"Missing package: {exc}. Run: python -m pip install -r requirements.txt")
            return
        except Exception as exc:
            st.session_state.messages.pop()
            st.error(f"The model call failed: {type(exc).__name__}: {exc}")
            return

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()  # refresh sidebar counters; history is replayed from session_state


main()