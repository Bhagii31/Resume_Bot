"""
app.py — Phase 4 of the Resume RAG Bot: the web app
====================================================
A FastAPI server that puts a web UI in front of the same RAG pipeline you
built on the command line. It reuses your existing code — nothing is
duplicated:

  • retrieve.py  → Retriever (embeds the question, queries ChromaDB)
  • chat.py      → SYSTEM_PROMPT + build_user_message (the prompt)

  Browser  ──POST /api/chat──▶  FastAPI  ──▶ retrieve top-K chunks
                                        ──▶ Claude (streamed)
  Browser  ◀──streamed text───  FastAPI  ◀──  answer tokens

WHY A BACKEND (instead of calling Claude from the browser)?
  Your ANTHROPIC_API_KEY must NEVER ship to the browser — anyone could
  read it and run up your bill. The server holds the key; the browser only
  ever talks to your own /api/chat endpoint.

Setup:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY="sk-ant-..."

Run:
    uvicorn app:app --reload
    # then open http://localhost:8000
"""

import os

import anthropic
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Reuse the pieces you already wrote — single source of truth.
from retrieve import Retriever, DEFAULT_K
from chat import SYSTEM_PROMPT, build_user_message, LLM_MODEL, MAX_TOKENS

app = FastAPI(title="Resume RAG Bot")

# Loaded once at startup (model + ChromaDB collection + Claude client).
# These are heavy to create, so we do it a single time and reuse.
retriever: Retriever | None = None
client: anthropic.Anthropic | None = None


@app.on_event("startup")
def _load() -> None:
    global retriever, client
    retriever = Retriever()        # loads embedding model + ChromaDB
    client = anthropic.Anthropic() # reads ANTHROPIC_API_KEY from the env


class ChatRequest(BaseModel):
    """Body of a POST /api/chat request."""
    question: str
    k: int = DEFAULT_K


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    """
    Run the RAG loop and stream Claude's answer back token-by-token.

    Returns a plain-text stream (not JSON) so the browser can render the
    answer as it arrives, giving that live "typing" feel.
    """
    question = req.question.strip()

    def generate():
        if not question:
            yield "Please ask a question."
            return
        if not os.environ.get("ANTHROPIC_API_KEY"):
            yield "Server error: ANTHROPIC_API_KEY is not set."
            return

        # STEP 1: retrieve  →  STEP 2: build prompt
        hits = retriever.search(question, k=req.k)
        user_message = build_user_message(question, hits)

        # STEP 3: generate (streamed)
        try:
            with client.messages.stream(
                model=LLM_MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except anthropic.APIError as e:
            yield f"\n\n[Error talking to Claude: {e}]"

    return StreamingResponse(generate(), media_type="text/plain")


# Serve the frontend (index.html, style.css, app.js) from /static at the root.
# Mounted LAST so it doesn't shadow the /api routes above.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
