"""
Structured LLM JSON Extractor — Gemini free-tier only.
Soft arithmetic warnings, session-persisted results, readable + JSON views.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st
from pydantic import ValidationError

from llm_client import create_client, FREE_MODELS, DEFAULT_MODEL
from registry import get_spec, labels_map, list_doc_types
from synthetic_data import get_synthetic_samples

st.set_page_config(
    page_title="Structured Extractor | Gemini",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None

DOC_TYPE_LABELS = labels_map()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.5rem; max-width: 1200px; }
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        border-radius: 16px; padding: 1.75rem 2rem; margin-bottom: 1.75rem;
        color: #f8fafc; box-shadow: 0 10px 40px -10px rgba(15, 23, 42, 0.4);
    }
    .app-header h1 { margin: 0; font-size: 1.65rem; font-weight: 700; color: #f8fafc !important; }
    .app-header p { margin: 0.4rem 0 0 0; font-size: 0.95rem; color: #94a3b8 !important; }
    .badge-row { margin-top: 0.85rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .badge {
        background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.15);
        color: #e2e8f0; font-size: 0.72rem; font-weight: 500;
        padding: 0.25rem 0.65rem; border-radius: 999px;
    }
    .badge-free { background: rgba(16,185,129,0.2); border-color: rgba(16,185,129,0.4); color: #6ee7b7; }
    .card-title { font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.05em; color: #64748b; margin-bottom: 0.75rem; }
    .status-success {
        background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 10px;
        padding: 0.85rem 1.1rem; color: #065f46; font-weight: 500; margin-bottom: 1rem;
    }
    .status-warn {
        background: #fffbeb; border: 1px solid #fcd34d; border-radius: 10px;
        padding: 0.85rem 1.1rem; color: #92400e; font-weight: 500; margin-bottom: 1rem;
    }
    .status-error {
        background: #fef2f2; border: 1px solid #fecaca; border-radius: 10px;
        padding: 0.85rem 1.1rem; color: #991b1b; font-weight: 500; margin-bottom: 1rem;
    }
    section[data-testid="stSidebar"] { background-color: #0f172a !important; border-right: 1px solid #1e293b !important; }
    section[data-testid="stSidebar"] > div { background-color: #0f172a !important; }
    section[data-testid="stSidebar"] h3 { color: #94a3b8 !important; font-size: 0.75rem !important;
        text-transform: uppercase !important; letter-spacing: 0.06em !important; }
    section[data-testid="stSidebar"] label { color: #e2e8f0 !important; font-weight: 500 !important; }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] small { color: #94a3b8 !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #1e293b !important; border-color: #334155 !important; color: #f1f5f9 !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0f766e, #0d9488) !important;
        border: none !important; color: white !important; font-weight: 600; border-radius: 10px;
    }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <h1>◈ Structured JSON Extractor</h1>
    <p>Extract structured data with soft validation warnings — Gemini free-tier models</p>
    <div class="badge-row">
        <span class="badge badge-free">Free models only</span>
        <span class="badge">Soft arithmetic warnings</span>
        <span class="badge">No invented numbers</span>
        <span class="badge">4 document types</span>
        <span class="badge">PDF + images</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Configuration")
    doc_type = st.radio(
        "Document type",
        options=list_doc_types(),
        format_func=lambda x: DOC_TYPE_LABELS.get(x, x),
        index=0,
    )
    st.markdown("---")
    st.markdown("### Model")
    default_idx = FREE_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in FREE_MODELS else 0
    model = st.selectbox("Gemini model (free tier)", options=FREE_MODELS, index=default_idx)
    temperature = st.slider("Temperature", 0.0, 0.5, 0.0, 0.05)
    show_messy = st.checkbox("Include messy samples", value=True)
    st.markdown("---")
    st.caption("API key from backend only (GOOGLE_API_KEY / .env)")
    st.caption("Free key → aistudio.google.com/apikey")


def render_readable(doc_type: str, data: dict) -> None:
    if doc_type == "invoice":
        st.markdown(f"### Invoice `{data.get('invoice_number') or '—'}`")
        st.markdown(
            f"**Date:** {data.get('invoice_date') or '—'}  \n"
            f"**Due:** {data.get('due_date') or '—'}  \n"
            f"**Currency:** {data.get('currency') or '—'}  \n"
            f"**Status:** {data.get('status') or '—'}  \n"
            f"**Payment terms:** {data.get('payment_terms') or '—'}"
        )
        seller, buyer = data.get("seller") or {}, data.get("buyer") or {}
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Seller:** {seller.get('name') or '—'}  \n{seller.get('address') or ''}")
        with c2:
            st.markdown(f"**Buyer:** {buyer.get('name') or '—'}  \n{buyer.get('address') or ''}")
        st.markdown("**Line items**")
        for i, item in enumerate(data.get("line_items") or [], 1):
            st.markdown(
                f"{i}. **{item.get('description', '—')}** — "
                f"{item.get('quantity', 0)} × {item.get('unit_price', 0)} = **{item.get('total', 0)}**"
            )
        st.markdown(
            f"**Subtotal:** {data.get('subtotal')}  ·  **Tax:** {data.get('tax_amount')}  ·  "
            f"**Discount:** {data.get('discount')}  \n"
            f"### Total: {data.get('currency') or ''} {data.get('total_amount')}"
        )
    elif doc_type == "resume":
        st.markdown(f"### {data.get('full_name') or '—'}")
        st.markdown(f"{data.get('email') or ''} · {data.get('phone') or ''} · {data.get('location') or ''}")
        if data.get("summary"):
            st.markdown(f"**Summary**  \n{data['summary']}")
        if data.get("experience"):
            st.markdown("**Experience**")
            for exp in data["experience"]:
                st.markdown(f"- **{exp.get('title')}** at **{exp.get('company')}**")
        if data.get("education"):
            st.markdown("**Education**")
            for edu in data["education"]:
                gpa = f" GPA {edu.get('gpa')}" if edu.get("gpa") is not None else ""
                st.markdown(f"- **{edu.get('degree')}** — {edu.get('institution')}{gpa}")
        if data.get("skills"):
            st.markdown("**Skills:** " + ", ".join(s.get("name", "") for s in data["skills"]))
    elif doc_type == "purchase_order":
        st.markdown(f"### Purchase Order `{data.get('po_number') or '—'}`")
        st.markdown(
            f"**Date:** {data.get('po_date') or '—'}  \n"
            f"**Delivery:** {data.get('delivery_date') or '—'}  \n"
            f"**Currency:** {data.get('currency') or '—'}"
        )
        buyer, supplier = data.get("buyer") or {}, data.get("supplier") or {}
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Buyer:** {buyer.get('name') or '—'}")
        with c2:
            st.markdown(f"**Supplier:** {supplier.get('name') or '—'}")
        st.markdown("**Line items**")
        for i, item in enumerate(data.get("line_items") or [], 1):
            code = f"`{item['item_code']}` " if item.get("item_code") else ""
            st.markdown(
                f"{i}. {code}**{item.get('description')}** — "
                f"{item.get('quantity')} × {item.get('unit_price')} = **{item.get('total')}**"
            )
        st.markdown(f"### Total: {data.get('currency') or ''} {data.get('total_amount')}")
    elif doc_type == "job_description":
        st.markdown(f"### {data.get('job_title') or '—'}")
        st.markdown(
            f"**Company:** {data.get('company') or '—'}  \n"
            f"**Location:** {data.get('location') or '—'}  \n"
            f"**Type:** {data.get('employment_type') or '—'}  \n"
            f"**Salary:** {data.get('salary_range') or '—'}"
        )
        if data.get("responsibilities"):
            st.markdown("**Responsibilities**")
            for r in data["responsibilities"]:
                st.markdown(f"- {r}")
        if data.get("skills"):
            st.markdown("**Skills:** " + ", ".join(data["skills"]))
    else:
        st.json(data)


col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.markdown('<p class="card-title">Input</p>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Input mode",
        ["My own text", "Load sample", "Upload file (PDF / image / audio / txt)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    default_text = ""
    file_bytes = None
    file_mime = None
    file_name = None

    if input_mode == "Load sample":
        samples = get_synthetic_samples(doc_type, include_messy=show_messy)
        if samples:
            labels = [
                f"{s['id']} · {s['title']}" + (" ⚠ messy" if s.get("messy") else "")
                for s in samples
            ]
            selected = st.selectbox("Choose a sample", labels, index=0)
            default_text = samples[labels.index(selected)]["text"].strip()
            st.caption("Sample loaded — editable before extract.")
        else:
            st.warning("No samples.")
    elif input_mode == "Upload file (PDF / image / audio / txt)":
        uploaded = st.file_uploader(
            "Upload PDF, image, audio, or text",
            type=[
                "pdf", "png", "jpg", "jpeg", "webp", "gif",
                "wav", "mp3", "m4a", "webm", "ogg", "flac",
                "txt", "text", "md",
            ],
            help="PDF/images use vision. Audio is transcribed+extracted by Gemini. TXT is plain text.",
        )
        if uploaded:
            file_name = uploaded.name
            raw = uploaded.read()
            mime = (uploaded.type or "").lower()
            lower = file_name.lower()

            if lower.endswith(".pdf"):
                mime = "application/pdf"
            elif lower.endswith(".png"):
                mime = "image/png"
            elif lower.endswith((".jpg", ".jpeg")):
                mime = "image/jpeg"
            elif lower.endswith(".webp"):
                mime = "image/webp"
            elif lower.endswith(".gif"):
                mime = "image/gif"
            elif lower.endswith(".wav"):
                mime = "audio/wav"
            elif lower.endswith(".mp3"):
                mime = "audio/mpeg"
            elif lower.endswith(".m4a"):
                mime = "audio/mp4"
            elif lower.endswith(".webm"):
                mime = "audio/webm"
            elif lower.endswith(".ogg"):
                mime = "audio/ogg"
            elif lower.endswith(".flac"):
                mime = "audio/flac"

            if mime.startswith("text/") or lower.endswith((".txt", ".text", ".md")):
                default_text = raw.decode("utf-8", errors="replace").strip()
                st.success(f"Loaded text: {file_name}")
            else:
                file_bytes = raw
                file_mime = mime or "application/octet-stream"
                st.success(f"Loaded file: {file_name} ({file_mime})")
                if mime.startswith("image/"):
                    st.image(raw, caption=file_name, use_container_width=True)
                elif mime == "application/pdf":
                    st.info("PDF attached — Gemini will read its pages.")
                elif mime.startswith("audio/"):
                    st.audio(raw, format=mime)
                    st.info("Audio attached — Gemini will transcribe and extract fields.")
    else:
        st.caption("Paste your own document text below.")

    input_text = st.text_area(
        "Document text / notes",
        value=default_text,
        height=280 if file_bytes else 380,
        placeholder="Paste unstructured text. For PDF/image uploads you can leave this empty or add notes.",
        label_visibility="collapsed",
        key=f"input_{input_mode}_{doc_type}",
    )

    can_extract = bool((input_text and input_text.strip()) or file_bytes)
    extract_btn = st.button(
        "Extract structured JSON",
        type="primary",
        use_container_width=True,
        disabled=not can_extract,
    )

with col_out:
    st.markdown('<p class="card-title">Validated output</p>', unsafe_allow_html=True)

    if extract_btn and can_extract:
        with st.spinner(f"Calling {model}…"):
            try:
                client = create_client(model=model)
                soft, raw_json = client.extract(
                    doc_type=doc_type,
                    text=input_text or "",
                    temperature=temperature,
                    file_bytes=file_bytes,
                    file_mime=file_mime,
                    file_name=file_name,
                )
                st.session_state.extraction_result = {
                    "data": soft.data,
                    "raw_json": raw_json,
                    "warnings": soft.warnings,
                    "doc_type": doc_type,
                    "source": file_name or "text",
                }
            except Exception as e:
                st.session_state.extraction_result = None
                st.markdown(
                    f'<div class="status-error">Extraction failed: {type(e).__name__}</div>',
                    unsafe_allow_html=True,
                )
                st.code(str(e), language="text")

    result = st.session_state.extraction_result
    if result is not None:
        warnings = result.get("warnings") or []
        if warnings:
            st.markdown(
                '<div class="status-warn">⚠ Extracted with warnings (values not altered to “fix” math)</div>',
                unsafe_allow_html=True,
            )
            for w in warnings:
                st.warning(w)
        else:
            st.markdown(
                '<div class="status-success">✓ Extraction successful — no soft warnings</div>',
                unsafe_allow_html=True,
            )

        view_mode = st.radio(
            "View as",
            ["Readable summary", "JSON"],
            horizontal=True,
            key="view_mode_radio",
        )
        if view_mode == "JSON":
            st.subheader("JSON output")
            st.code(result["raw_json"], language="json")
        else:
            render_readable(result["doc_type"], result["data"])

        st.download_button(
            "Download JSON",
            data=result["raw_json"],
            file_name=f"{result['doc_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json",
        )
    elif not extract_btn:
        st.info("Provide text and/or a PDF/image, then click **Extract structured JSON**.")
        with st.expander("Expected schema"):
            try:
                st.json(get_spec(doc_type).schema.model_json_schema())
            except Exception as e:
                st.write(str(e))

st.markdown("---")
with st.expander("How validation works"):
    st.markdown("""
**Hard errors** (retry / fail): wrong types, invalid JSON structure.  

**Soft warnings** (data still returned): arithmetic mismatches, missing recommended fields.  
The model is instructed **not** to invent numbers to make totals match.

**Messy samples** exercise OCR noise, credit notes, non-US GPA scales, and incomplete postings.
""")
