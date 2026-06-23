"""
chat.py — Phase 3 of the Resume RAG Bot
========================================
This is the final piece: the "G" in RAG — Generation.

ingest.py indexed your resume; retrieve.py finds the most relevant chunks
for a question. chat.py adds the LLM layer on top: it retrieves context,
hands it to Claude, and Claude writes a natural-language answer grounded
ONLY in your resume.

  STEP 1: Retrieve — get the top-K resume chunks for the question
                     (reuses the Retriever you already built)
  STEP 2: Augment  — stuff those chunks into a prompt as context
  STEP 3: Generate — Claude reads the context + question, writes the answer

WHY GROUND THE ANSWER IN RETRIEVED CHUNKS?
  Without context, an LLM would make up plausible-sounding resume details
  (hallucinate). By forcing Claude to answer from the retrieved chunks —
  and to say "I don't know" when the resume doesn't cover it — answers
  stay faithful to what's actually on the resume.

Setup:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY="sk-ant-..."   # get one at console.anthropic.com

Usage:
    python chat.py "what are her python skills?"
    python chat.py                # interactive chat mode
"""

import argparse
import os
import sys

import anthropic

# Reuse the retrieval layer from Phase 2 — no need to duplicate it.
from retrieve import Retriever, DEFAULT_K


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

# claude-opus-4-8 is Anthropic's most capable model. For a high-volume
# or cost-sensitive bot you could switch to "claude-sonnet-4-6".
LLM_MODEL  = "claude-opus-4-8"
MAX_TOKENS = 1024          # plenty for a resume answer; bump up if truncated
N_CONTEXT  = DEFAULT_K     # how many resume chunks to feed Claude as context


# The system prompt sets the rules. The key instruction is "answer ONLY
# from the context" — that's what keeps Claude from inventing details.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a person's resume.

Rules:
- Answer ONLY using the resume excerpts provided in the user's message.
- If the excerpts don't contain the answer, say so plainly — do not guess
  or invent experience, skills, dates, or employers.
- Be concise and specific. Quote concrete details (tools, numbers, roles)
  when they're present.
- Write in the third person (e.g. "She has...") unless asked otherwise."""


# ═══════════════════════════════════════════════════════════════════
# Build the prompt
# ═══════════════════════════════════════════════════════════════════

def build_user_message(question: str, hits: list[dict]) -> str:
    """
    Combine the retrieved chunks + the question into a single user message.

    The format matters: clearly labeled context, then the question. This
    makes it obvious to Claude which text is "the resume" vs. "the ask".
    """
    if not hits:
        context = "(no relevant resume excerpts were found)"
    else:
        blocks = []
        for h in hits:
            idx = h["metadata"].get("chunk_index", "?")
            blocks.append(f"[Excerpt {idx} | similarity {h['similarity']:.2f}]\n{h['text']}")
        context = "\n\n".join(blocks)

    return (
        f"Here are the most relevant excerpts from the resume:\n\n"
        f"{context}\n\n"
        f"---\n"
        f"Question: {question}"
    )


# ═══════════════════════════════════════════════════════════════════
# Generate the answer
# ═══════════════════════════════════════════════════════════════════

def answer(client: anthropic.Anthropic, retriever: Retriever,
           question: str, k: int = N_CONTEXT) -> None:
    """
    Run the full RAG loop for one question and stream the answer to stdout.
    """
    # ── STEP 1: Retrieve ──────────────────────────────────────────
    hits = retriever.search(question, k=k)

    # ── STEP 2: Augment ───────────────────────────────────────────
    user_message = build_user_message(question, hits)

    # ── STEP 3: Generate (streamed for a responsive, typewriter feel) ─
    print("\n  ", end="", flush=True)
    with client.messages.stream(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            # Indent continuation lines so the answer reads as one block
            print(text.replace("\n", "\n  "), end="", flush=True)
    print("\n")


# ═══════════════════════════════════════════════════════════════════
# Interactive chat loop
# ═══════════════════════════════════════════════════════════════════

def interactive_loop(client: anthropic.Anthropic, retriever: Retriever,
                     k: int) -> None:
    """Keep answering questions until the user types 'quit' or hits Ctrl-C."""
    print("  Resume chat — ask anything about the resume. Type 'quit' to exit.\n")
    while True:
        try:
            question = input("  you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("  Bye!")
            break
        answer(client, retriever, question, k=k)


# ═══════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chat with your resume using retrieval + Claude."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Your question. Omit to enter interactive chat mode.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=N_CONTEXT,
        help=f"How many resume chunks to use as context (default: {N_CONTEXT})",
    )
    args = parser.parse_args()

    # Fail early with a friendly message if the API key is missing.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n  ✗ ANTHROPIC_API_KEY is not set.")
        print("  Get a key at https://console.anthropic.com and run:")
        print('      export ANTHROPIC_API_KEY="sk-ant-..."')
        print("  Then try again.\n")
        sys.exit(1)

    # Anthropic() reads the API key from the ANTHROPIC_API_KEY env var.
    client = anthropic.Anthropic()
    retriever = Retriever()   # loads the model + ChromaDB collection once

    if args.question:
        answer(client, retriever, " ".join(args.question), k=args.k)
    else:
        interactive_loop(client, retriever, k=args.k)
