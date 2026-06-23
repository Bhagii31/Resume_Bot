
import argparse
import os
import sys
import anthropic
from retrieve import Retriever, DEFAULT_K

LLM_MODEL  = "claude-opus-4-8"
MAX_TOKENS = 1024          # plenty for a resume answer; bump up if truncated
N_CONTEXT  = DEFAULT_K     # how many resume chunks to feed Claude as context
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a person's resume.

Rules:
- Answer ONLY using the resume excerpts provided in the user's message.
- If the excerpts don't contain the answer, say so plainly — do not guess
  or invent experience, skills, dates, or employers.
- Be concise and specific. Quote concrete details (tools, numbers, roles)
  when they're present.
- Write in the third person (e.g. "She has...") unless asked otherwise."""

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


def answer(client: anthropic.Anthropic, retriever: Retriever,
           question: str, k: int = N_CONTEXT) -> None:
    
    hits = retriever.search(question, k=k)

    user_message = build_user_message(question, hits)

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

def interactive_loop(client: anthropic.Anthropic, retriever: Retriever,
                     k: int) -> None:
   
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

    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n  ✗ ANTHROPIC_API_KEY is not set.")
        print("  Get a key at https://console.anthropic.com and run:")
        print('      export ANTHROPIC_API_KEY="sk-ant-..."')
        print("  Then try again.\n")
        sys.exit(1)


    client = anthropic.Anthropic()
    retriever = Retriever()   # loads the model + ChromaDB collection once

    if args.question:
        answer(client, retriever, " ".join(args.question), k=args.k)
    else:
        interactive_loop(client, retriever, k=args.k)
