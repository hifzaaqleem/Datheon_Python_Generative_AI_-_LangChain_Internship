"""
Multi-Tool Autonomous AI Agent
LangChain Agent + Streamlit Dashboard
Tools: Web Search (DuckDuckGo), Python Calculator, Local File Reader
LLM: Google Gemini (free tier) — key loaded from .env only
"""

import os
import traceback
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

# Load .env before anything else that needs the key
load_dotenv()  # looks for .env in cwd and parents

# ---------------------------------------------------------------------------
# Page config must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Tool Autonomous AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – clean, eye-catching dark dashboard
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 45%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        letter-spacing: -0.3px;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        color: #f8fafc;
        line-height: 1.25;
    }
    .main-title .emoji {
        font-size: 2.4rem;
        line-height: 1;
        display: inline-block;
        flex-shrink: 0;
        /* Use system emoji fonts — Inter has no robot glyph and shows a broken face */
        font-family: "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji",
                     "Android Emoji", "EmojiOne Color", "Twemoji Mozilla", sans-serif !important;
        font-style: normal;
        font-weight: normal;
        color: initial !important;
        -webkit-text-fill-color: initial !important;
        background: none !important;
        filter: none !important;
        text-shadow: none !important;
    }
    .main-title .title-text {
        font-family: 'Inter', sans-serif;
        color: #f8fafc;
        -webkit-text-fill-color: #f8fafc;
        background: none;
        font-weight: 700;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
        margin-left: 0.15rem;
        line-height: 1.45;
    }

    .tool-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s;
    }
    .tool-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
    }
    .tool-name {
        font-weight: 600;
        color: #e2e8f0;
        font-size: 0.95rem;
    }
    .tool-desc {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 0.25rem;
    }

    .user-bubble {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white;
        padding: 0.85rem 1.1rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.6rem 0;
        max-width: 85%;
        margin-left: auto;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
    }
    .assistant-bubble {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.12);
        color: #e2e8f0;
        padding: 0.85rem 1.1rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.6rem 0;
        max-width: 90%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    .step-box {
        background: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #38bdf8;
        padding: 0.7rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #cbd5e1;
    }
    .step-label {
        color: #38bdf8;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Force sidebar visible even if user collapsed it */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.1);
        transform: none !important;
        margin-left: 0 !important;
        width: 21rem !important;
        min-width: 21rem !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        transform: none !important;
        margin-left: 0 !important;
        width: 21rem !important;
        display: block !important;
    }
    /* Make the reopen control obvious if sidebar is collapsed */
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: #3b82f6 !important;
        color: white !important;
        border-radius: 0 8px 8px 0 !important;
        padding: 0.6rem !important;
        z-index: 999999 !important;
    }
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.8rem;
    }

    .status-ready {
        display: inline-block;
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .status-warn {
        display: inline-block;
        background: rgba(251, 191, 36, 0.15);
        color: #fbbf24;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.4rem !important;
        transition: transform 0.15s, box-shadow 0.15s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35) !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Paths & Gemini key (from .env only — never asked in UI)
# ---------------------------------------------------------------------------
# Relative to this script so it works locally AND on Streamlit Cloud
_BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = _BASE_DIR / "agent_workspace"
UPLOAD_DIR = _BASE_DIR / "agent_uploads"
try:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    # Fallback if filesystem is read-only (rare on some hosts)
    import tempfile
    WORKSPACE = Path(tempfile.gettempdir()) / "agent_workspace"
    UPLOAD_DIR = Path(tempfile.gettempdir()) / "agent_uploads"
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Support env vars and Streamlit Cloud secrets
def _load_gemini_key() -> str:
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"):
        val = os.getenv(name)
        if val and val.strip():
            return val.strip()
    try:
        # Streamlit Cloud: Settings → Secrets
        for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"):
            if name in st.secrets and str(st.secrets[name]).strip():
                return str(st.secrets[name]).strip()
    except Exception:
        pass
    return ""


GEMINI_API_KEY = _load_gemini_key()

# Only these two models
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
]

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web in real-time using DuckDuckGo. Use for current events, facts, research, or any information not in your knowledge. Input should be a clear search query."""
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"URL: {r.get('href', 'N/A')}\n"
                    f"Snippet: {r.get('body', 'N/A')}\n"
                )
        if not results:
            return "No results found for that query."
        return "\n---\n".join(results)
    except Exception as e:
        return f"Web search error: {str(e)}"


@tool
def python_calculator(expression: str) -> str:
    """Evaluate a mathematical or simple Python expression safely.
    Supports arithmetic, math functions (math.sqrt, math.sin, etc.), lists, and basic builtins.
    Examples: '2**10 + math.sqrt(16)', 'sum([1,2,3,4,5])', 'round(math.pi * 2, 3)'."""
    import math
    import ast

    allowed_names = {
        k: getattr(math, k) for k in dir(math) if not k.startswith("_")
    }
    allowed_names.update(
        {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "len": len,
            "int": int,
            "float": float,
            "range": range,
            "list": list,
            "True": True,
            "False": False,
            "None": None,
        }
    )

    try:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return "Error: Imports are not allowed."
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return (
            f"Calculator error: {e}. "
            "Use expressions like 'math.sqrt(9) + 2**3' or 'sum(range(1,11))'."
        )


@tool
def read_local_file(filename: str) -> str:
    """Read the content of a local text file from the agent workspace or uploaded files. Provide the exact filename (e.g. 'notes.txt' or 'data.csv'). Returns the file content or an error message."""
    candidates = [
        WORKSPACE / filename,
        UPLOAD_DIR / filename,
        Path(filename),
    ]
    for path in candidates:
        try:
            if path.is_file() and path.stat().st_size < 2_000_000:
                text = path.read_text(encoding="utf-8", errors="replace")
                if len(text) > 15000:
                    return text[:15000] + "\n\n... [truncated, file too long]"
                return text
        except Exception:
            continue
    available = []
    for d in (WORKSPACE, UPLOAD_DIR):
        if d.exists():
            available.extend([p.name for p in d.iterdir() if p.is_file()])
    avail_str = ", ".join(available[:20]) if available else "(none)"
    return f"File '{filename}' not found. Available files: {avail_str}"


TOOLS = [web_search, python_calculator, read_local_file]


# ---------------------------------------------------------------------------
# Agent factory — Google Gemini only
# ---------------------------------------------------------------------------
def build_agent(model_name: str = "gemini-3.5-flash-lite"):
    """Create a LangChain tool-calling agent powered by Gemini."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.agents import create_agent

    if not GEMINI_API_KEY:
        raise ValueError(
            "Gemini API key not found. Put GOOGLE_API_KEY=... (or GEMINI_API_KEY=...) in a .env file "
            "next to this script. Get a free key at https://aistudio.google.com/apikey"
        )

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )

    system_prompt = """You are a helpful, precise Multi-Tool Autonomous AI Agent.
You have access to three tools:
1. web_search – real-time web search via DuckDuckGo
2. python_calculator – safe math & simple Python expression evaluation
3. read_local_file – read text files from the local workspace

When solving compound questions:
- Think step by step.
- Choose the right tool(s) and use them when needed.
- You may call multiple tools in sequence.
- After gathering information, give a clear final answer.
- Cite sources when using web search (titles + URLs).
- Never invent file contents or search results.
"""

    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=system_prompt,
    )
    return agent


def _extract_text(content: Any) -> str:
    """Turn Gemini/LangChain message content into plain readable text.
    Handles: plain str, list of blocks [{'type':'text','text':...}], dicts, etc.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Gemini style: {'type': 'text', 'text': '...'}
                if "text" in block:
                    parts.append(str(block["text"]))
                elif block.get("type") == "text" and "text" in block:
                    parts.append(str(block["text"]))
                else:
                    # skip signatures / extras
                    if block.get("type") in ("text", None) or "text" in block:
                        parts.append(str(block.get("text", "")))
            else:
                # AIMessage chunk or other object with .text / .content
                text = getattr(block, "text", None) or getattr(block, "content", None)
                if text:
                    parts.append(str(text))
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return str(content)
    # Fallback
    return str(content)


def run_agent(agent, user_query: str) -> dict[str, Any]:
    """Invoke the agent and collect messages / steps."""
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    result = agent.invoke({"messages": [HumanMessage(content=user_query)]})

    messages = result.get("messages", [])
    steps = []
    final_answer = ""

    for msg in messages:
        if isinstance(msg, HumanMessage):
            continue
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    steps.append(
                        {
                            "type": "action",
                            "tool": tc.get("name", "unknown"),
                            "input": tc.get("args", {}),
                        }
                    )
            text = _extract_text(msg.content)
            if text.strip():
                steps.append({"type": "thought", "content": text})
                final_answer = text  # last non-empty AI text is the answer
        elif isinstance(msg, ToolMessage):
            obs = _extract_text(msg.content)
            if len(obs) > 2000:
                obs = obs[:2000] + " …"
            steps.append(
                {
                    "type": "observation",
                    "tool": getattr(msg, "name", "tool"),
                    "content": obs,
                }
            )

    return {"steps": steps, "final_answer": final_answer, "raw_messages": messages}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None


# ---------------------------------------------------------------------------
# Sidebar (no API key field — key comes from .env only)
# ---------------------------------------------------------------------------
if "model_name" not in st.session_state:
    st.session_state.model_name = GEMINI_MODELS[0]

with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Configuration</div>', unsafe_allow_html=True)
    st.caption(f"Model: `{st.session_state.model_name}`")
    st.caption("Change model in main page → Settings")

    st.markdown("---")
    st.markdown('<div class="sidebar-header">🛠️ Available Tools</div>', unsafe_allow_html=True)

    tools_info = [
        ("🔍 Web Search", "DuckDuckGo real-time search"),
        ("🧮 Calculator", "Safe Python math expressions"),
        ("📄 File Reader", "Read local / uploaded text files"),
    ]
    for name, desc in tools_info:
        st.markdown(
            f"""
            <div class="tool-card">
                <div class="tool-name">{name}</div>
                <div class="tool-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="sidebar-header">📁 File Workspace</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a text file for the agent to read",
        type=["txt", "md", "csv", "json", "py", "log", "html"],
        help="File will be saved to the agent workspace.",
    )
    if uploaded is not None:
        save_path = UPLOAD_DIR / uploaded.name
        save_path.write_bytes(uploaded.getvalue())
        (WORKSPACE / uploaded.name).write_bytes(uploaded.getvalue())
        st.success(f"Uploaded: `{uploaded.name}`")

    files = list(WORKSPACE.glob("*")) + list(UPLOAD_DIR.glob("*"))
    unique_names = sorted({f.name for f in files if f.is_file()})
    if unique_names:
        st.caption("Files available to agent:")
        for n in unique_names[:15]:
            st.code(n, language=None)
    else:
        st.caption("No files yet. Upload one above.")

    st.markdown("---")
    if GEMINI_API_KEY:
        st.markdown('<span class="status-ready">● Agent Ready (Gemini · .env)</span>', unsafe_allow_html=True)
        st.caption(f"Key loaded · Model: `{st.session_state.model_name}`")
    else:
        st.markdown('<span class="status-warn">○ Missing GOOGLE_API_KEY in .env</span>', unsafe_allow_html=True)
        st.info(
            "Create a `.env` file next to this script with:\n\n"
            "`GOOGLE_API_KEY=your_key_here`\n\n"
            "Get a free key: [Google AI Studio](https://aistudio.google.com/apikey)"
        )

    st.caption("Built with LangChain · Gemini · Streamlit · DuckDuckGo")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.markdown(
    '''
    <h1 class="main-title">
        <span class="emoji">🤖</span>
        <span class="title-text">Multi-Tool Autonomous AI Agent</span>
    </h1>
    ''',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Reasons, picks tools, and solves research &amp; calculation queries. Powered by Gemini.</p>',
    unsafe_allow_html=True,
)

# Fallback controls in main area (always available if sidebar is hidden)
with st.expander("⚙️ Settings (model & files) — open this if sidebar is missing", expanded=False):
    st.selectbox(
        "Gemini Model",
        options=GEMINI_MODELS,
        key="model_name",
        help="Default: gemini-3.5-flash-lite",
    )
    uploaded_main = st.file_uploader(
        "Upload a text file for the agent",
        type=["txt", "md", "csv", "json", "py", "log", "html"],
        key="main_uploader",
    )
    if uploaded_main is not None:
        (UPLOAD_DIR / uploaded_main.name).write_bytes(uploaded_main.getvalue())
        (WORKSPACE / uploaded_main.name).write_bytes(uploaded_main.getvalue())
        st.success(f"Uploaded: `{uploaded_main.name}`")
    if GEMINI_API_KEY:
        st.success(f"API key loaded · model: `{st.session_state.get('model_name', GEMINI_MODELS[0])}`")
    else:
        st.warning("No API key. On Streamlit Cloud add GOOGLE_API_KEY in **Manage app → Settings → Secrets**.")

with st.expander("💡 Example compound queries (click to try)", expanded=False):
    examples = [
        "What is the current population of Tokyo? Then calculate 15% of that number.",
        "Search for the latest Python release version and tell me what 2 to the power of that major version number is.",
        "Read the file notes.txt if it exists, summarize it, and search the web for any related recent news.",
        "What is the square root of 2025? Also search who won the last FIFA World Cup.",
        "Calculate the compound interest on $10,000 at 5% for 8 years (yearly). Then search for current best savings rates.",
    ]
    for i, ex in enumerate(examples):
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_query = ex

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        role = msg["role"]
        if role == "user":
            st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            if msg.get("steps"):
                with st.expander("🧠 Reasoning Chain (Thought → Action → Observation)", expanded=False):
                    for step in msg["steps"]:
                        if step["type"] == "thought":
                            st.markdown(
                                f'<div class="step-box"><span class="step-label">Thought</span><br>{step["content"]}</div>',
                                unsafe_allow_html=True,
                            )
                        elif step["type"] == "action":
                            st.markdown(
                                f'<div class="step-box"><span class="step-label">Action → {step["tool"]}</span><br>'
                                f'Input: <code>{step["input"]}</code></div>',
                                unsafe_allow_html=True,
                            )
                        elif step["type"] == "observation":
                            obs = step["content"]
                            if len(obs) > 600:
                                obs = obs[:600] + " …"
                            st.markdown(
                                f'<div class="step-box"><span class="step-label">Observation</span><br>{obs}</div>',
                                unsafe_allow_html=True,
                            )
            st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

st.markdown("---")
query = st.chat_input("Ask a compound question… the agent will choose tools autonomously")

if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    if not GEMINI_API_KEY:
        st.error(
            "No Gemini API key found. Create a `.env` file in the project folder with:\n\n"
            "`GOOGLE_API_KEY=your_free_key`\n\n"
            "Get one at https://aistudio.google.com/apikey"
        )
    else:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Agent is thinking & using tools (Gemini)…"):
            try:
                agent = build_agent(st.session_state.model_name)
                outcome = run_agent(agent, query)

                answer = outcome["final_answer"] or "I completed the task but produced no textual answer."
                steps = outcome["steps"]

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "steps": steps,
                    }
                )
            except Exception as e:
                err_msg = f"Agent error: {str(e)}\n\n```\n{traceback.format_exc()[-800:]}\n```"
                st.session_state.messages.append(
                    {"role": "assistant", "content": err_msg, "steps": []}
                )

        st.rerun()

st.markdown(
    """
<div style="text-align:center; color:#64748b; font-size:0.8rem; margin-top:2rem;">
    Key is read only from <code>.env</code> (GOOGLE_API_KEY or GEMINI_API_KEY).<br>
    The agent autonomously decides which tools to call and synthesizes a final answer.
</div>
""",
    unsafe_allow_html=True,
)

