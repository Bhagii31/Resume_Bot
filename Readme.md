# 📄 Resume RAG Bot

A **Retrieval-Augmented Generation** chatbot that answers questions about a résumé —
with a clean web UI. Built from scratch (no LangChain) so every piece is easy to follow.

Ask *"What are her machine learning skills?"* and get an answer grounded in the actual
résumé, not made up.

```
PDF → chunk → embed → store → retrieve → Claude → answer
```

## ✨ Features

- **Full RAG pipeline**, written from scratch — readable and well-commented
- **Local embeddings** with `sentence-transformers` (MiniLM) — no API needed to index
- **ChromaDB** vector store with cosine similarity
- **Claude** (`claude-opus-4-8`) for grounded, no-hallucination answers
- **Modern web UI** (FastAPI + vanilla JS) with streaming, typewriter-style replies
- **CLI tools** too, for each stage of the pipeline

## 🧱 How it works

| Stage | File | What it does |
|-------|------|--------------|
| **Add** | [`add_resume.py`](add_resume.py) | Validates a PDF (real text, not corrupted) then indexes it |
| **1. Ingest** | [`ingest.py`](ingest.py) | Read PDF → chunk → embed → store in ChromaDB |
| **2. Retrieve** | [`retrieve.py`](retrieve.py) | Embed a question → find the top-K most similar chunks |
| **3. Chat (CLI)** | [`chat.py`](chat.py) | Retrieve context → ask Claude → stream the answer |
| **4. Web app** | [`app.py`](app.py) + [`static/`](static/) | The same pipeline behind a FastAPI server + chat UI |

The browser never sees your API key — it only talks to your own `/api/chat` endpoint,
which holds the key server-side.

## 🚀 Quickstart

```bash
# 1. Install dependencies (use a virtualenv)
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Add your résumé (validates + indexes it in one step)
python add_resume.py /path/to/your_resume.pdf

# 3. Set your Claude API key  (get one at https://console.anthropic.com)
export ANTHROPIC_API_KEY="sk-ant-..."

# 4a. Run the web app  →  open http://localhost:8000
uvicorn app:app --reload

# 4b. ...or use the CLI
python chat.py "what are her main skills?"
python chat.py                 # interactive chat
```

> **Note:** the first run downloads the MiniLM embedding model (~80 MB), then caches it.

## 🖼️ Web UI

A modern dark chat interface with suggested questions, streaming answers, and a
mobile-friendly layout. Start the server and open <http://localhost:8000>.

*(Add a screenshot here once you've run it — `![screenshot](docs/screenshot.png)`.)*

## 🛠️ Configuration

Tweak these constants if results feel off:

- **Chunking** — `CHUNK_SIZE` / `CHUNK_OVERLAP` in [`ingest.py`](ingest.py)
- **Retrieval depth** — `DEFAULT_K` in [`retrieve.py`](retrieve.py)
- **Model / answer length** — `LLM_MODEL` / `MAX_TOKENS` in [`chat.py`](chat.py)

## 📦 Tech stack

`pdfplumber` · `sentence-transformers` · `chromadb` · `anthropic` · `fastapi` · `uvicorn`

