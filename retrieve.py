"""
retrieve.py — Phase 2 of the Resume RAG Bot
============================================
ingest.py turned your resume into vectors stored in ChromaDB.
This script does the OTHER half of retrieval: it takes a plain-English
question, turns it into a vector the same way, and asks ChromaDB for the
chunks whose vectors are most similar.

That's the "R" in RAG — Retrieval. Phase 3 (chat.py) will feed these
chunks to an LLM to write the final answer.

  STEP 1: Load  — open the ChromaDB collection ingest.py built
  STEP 2: Embed — turn the question into a 384-dim vector (same model!)
  STEP 3: Query — ask ChromaDB for the top-K nearest chunks
  STEP 4: Show  — print them, ranked by similarity

Usage:
    python retrieve.py "what are my python skills?"
    python retrieve.py "where did I go to school?" --k 5
    python retrieve.py                 # drops into interactive mode

IMPORTANT:
    The embedding model here MUST match the one in ingest.py, or the
    vectors live in a different "space" and similarity is meaningless.
"""

import argparse
import sys
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────────
# Configuration — MUST match ingest.py
# ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "resume"

DEFAULT_K = 3   # how many chunks to retrieve by default


# ═══════════════════════════════════════════════════════════════════
# Retriever — loads everything once, then answers many queries
# ═══════════════════════════════════════════════════════════════════

class Retriever:
    """
    Holds the embedding model + ChromaDB collection so they're loaded
    ONCE. In interactive mode you query many times — you don't want to
    reload an 80MB model on every question.
    """

    def __init__(self):
        # ── STEP 1: Load the collection ingest.py built ──────────────
        if not Path(CHROMA_DB_PATH).exists():
            print(f"\n  ✗ No database found at {CHROMA_DB_PATH}/")
            print(f"  Run ingest.py first:  python ingest.py --pdf <your.pdf>")
            sys.exit(1)

        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        try:
            self.collection = client.get_collection(COLLECTION_NAME)
        except Exception:
            print(f"\n  ✗ No '{COLLECTION_NAME}' collection in {CHROMA_DB_PATH}/")
            print(f"  Run ingest.py first to build it.")
            sys.exit(1)

        count = self.collection.count()
        if count == 0:
            print(f"\n  ✗ Collection '{COLLECTION_NAME}' is empty.")
            print(f"  Re-run ingest.py — something went wrong during indexing.")
            sys.exit(1)

        # ── Load the SAME model ingest.py used ───────────────────────
        print(f"  Loading model {EMBEDDING_MODEL}...")
        t0 = time.time()
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"  Ready — {count} chunks indexed, model loaded in "
              f"{time.time()-t0:.1f}s\n")

    def search(self, question: str, k: int = DEFAULT_K) -> list[dict]:
        """
        Return the top-k chunks most similar to `question`.

        HOW IT WORKS:
          1. Embed the question with the same model as the chunks.
          2. ChromaDB compares that vector to every stored vector using
             cosine distance and returns the k closest.
          3. We convert distance → similarity (1 - distance) so bigger
             = more relevant, which reads more naturally.

        Returns a list of dicts: {rank, similarity, text, metadata}.
        """
        # ── STEP 2: Embed the question (normalise — matches ingest) ──
        q_vec = self.model.encode(
            [question],
            normalize_embeddings=True,   # same setting as ingest.py
        )

        # ── STEP 3: Ask ChromaDB for the nearest chunks ─────────────
        results = self.collection.query(
            query_embeddings=q_vec.tolist(),
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB nests results one level deep (one list per query).
        docs   = results["documents"][0]
        metas  = results["metadatas"][0]
        dists  = results["distances"][0]

        hits = []
        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
            hits.append({
                "rank":       rank,
                "similarity": 1 - dist,   # cosine distance → similarity
                "text":       doc,
                "metadata":   meta,
            })
        return hits


# ═══════════════════════════════════════════════════════════════════
# Pretty-printing
# ═══════════════════════════════════════════════════════════════════

def print_hits(question: str, hits: list[dict]) -> None:
    """Display retrieved chunks, ranked, with a similarity bar."""
    print(f"\n{'='*60}")
    print(f"  Q: {question}")
    print(f"{'='*60}")

    if not hits:
        print("  No results.")
        return

    for h in hits:
        sim   = h["similarity"]
        bar   = "█" * int(sim * 20)            # crude visual similarity meter
        idx   = h["metadata"].get("chunk_index", "?")
        preview = h["text"].strip().replace("\n", " ")
        if len(preview) > 280:
            preview = preview[:280] + "…"

        print(f"\n  #{h['rank']}  similarity={sim:.3f}  {bar}")
        print(f"      [chunk_{idx}]  {preview}")
    print()


# ═══════════════════════════════════════════════════════════════════
# Interactive mode
# ═══════════════════════════════════════════════════════════════════

def interactive_loop(retriever: Retriever, k: int) -> None:
    """Keep asking for questions until the user types 'quit' or Ctrl-C."""
    print("  Interactive mode — type a question, or 'quit' to exit.")
    print(f"  Returning top {k} chunks per query.\n")
    while True:
        try:
            question = input("  ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("  Bye!")
            break
        hits = retriever.search(question, k=k)
        print_hits(question, hits)


# ═══════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the resume vector store for relevant chunks."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Your question. Omit to enter interactive mode.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"How many chunks to retrieve (default: {DEFAULT_K})",
    )
    args = parser.parse_args()

    retriever = Retriever()

    if args.question:
        question = " ".join(args.question)
        hits = retriever.search(question, k=args.k)
        print_hits(question, hits)
    else:
        interactive_loop(retriever, k=args.k)
