

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
retriever: Retriever | None = None
client: anthropic.Anthropic | None = None


@app.on_event("startup")
def _load() -> None:
    global retriever, client
    retriever = Retriever()        # loads embedding model + ChromaDB
    client = anthropic.Anthropic() # reads ANTHROPIC_API_KEY from the env


class ChatRequest(BaseModel):

    question: str
    k: int = DEFAULT_K


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    
    question = req.question.strip()

    def generate():
        if not question:
            yield "Please ask a question."
            return
        if not os.environ.get("ANTHROPIC_API_KEY"):
            yield "Server error: ANTHROPIC_API_KEY is not set."
            return

        
        hits = retriever.search(question, k=req.k)
        user_message = build_user_message(question, hits)


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

app.mount("/", StaticFiles(directory="static", html=True), name="static")
