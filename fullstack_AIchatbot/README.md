# Nova - Streamlit + LangChain + Gemini Chatbot

Streaming chat UI with conversation memory and optional document Q&A (RAG).

## Setup
```bash
python -m pip install -r requirements.txt
```
Create a `.env` file next to `app.py`:
```
GOOGLE_API_KEY=your-gemini-api-key
```
(`GEMINI_API_KEY` also works.) Then run from this folder:
```bash
python -m streamlit run app.py
```
Opens at http://localhost:8501. Run from this folder so `.streamlit/config.toml` (dark theme) is picked up.

## Features
- Streaming answers (`st.write_stream`), chat bubbles, starter prompt cards
- Memory in `st.session_state` (windowed); "New chat" resets the session
- RAG: upload .txt/.md/.pdf -> chunked -> TF-IDF retrieval; sources shown per answer
- Chat transcript download