# Multi-Tool Autonomous AI Agent

LangChain tool-calling agent + Streamlit dashboard, powered by **free Google Gemini**.

## Tools
| Tool | Description |
|------|-------------|
| `web_search` | Real-time DuckDuckGo search |
| `python_calculator` | Safe math / Python expression evaluator |
| `read_local_file` | Reads text files from the agent workspace |

## Setup (no API key in the UI)

1. Get a **free** Gemini key: https://aistudio.google.com/apikey

2. Create a `.env` file in this folder:

```bash
cp .env.example .env
# then edit .env and set:
GOOGLE_API_KEY=your_actual_key_here
```

3. Install and run:

```bash
pip install -r requirements.txt
streamlit run multi_tool_ai_agent.py
```

The key is loaded only from `.env` (or the environment). Nothing is typed in the dashboard.

## Models (free tier)
Default: `gemini-2.0-flash`  
Also selectable: `gemini-2.0-flash-lite`, `gemini-1.5-flash`, `gemini-1.5-flash-8b`, `gemini-2.5-flash`

## How it works
1. You ask a compound question.
2. The agent decides which tools to call (search / calc / file).
3. Tools run → observations return.
4. Agent writes a final answer.
5. Full reasoning chain is shown in an expandable panel.

## Example queries
- "What is the current population of Tokyo? Then calculate 15% of that number."
- "Search for the latest Python release and raise 2 to the power of its major version."
- "Read notes.txt and search the web for related news."
