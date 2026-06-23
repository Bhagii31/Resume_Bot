"""
ingest.py — Phase 1 of the Resume RAG Bot
==========================================
This script does the complete indexing pipeline in four steps:

  STEP 1: Read  — extract raw text from your resume PDF
  STEP 2: Chunk — split the text into overlapping 300-token windows
  STEP 3: Embed — convert each chunk to a 384-dim vector with MiniLM
  STEP 4: Store — save chunks + vectors + metadata to ChromaDB on disk

Run this ONCE before starting chat.py.
If you update your resume, just run ingest.py again — it wipes the old
collection and rebuilds from scratch.

Usage:
    python ingest.py                        # uses data/resume.pdf by default
    python ingest.py --pdf data/my_cv.pdf  # point at any PDF

Install dependencies first:
    pip install -r requirements.txt
"""

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────
# Configuration — tweak these if your results feel off
# ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH   = "./chroma_db"
COLLECTION_NAME  = "resume"

# Chunking settings (see CHUNKING_GUIDE below for why these numbers)
CHUNK_SIZE    = 300   # target tokens per chunk
CHUNK_OVERLAP = 50    # tokens shared between consecutive chunks

# Token estimate: 1 word ≈ 1.3 tokens for English.
# Using words as a proxy is fine for MiniLM (max 256 tokens).
# Swap in a real tokenizer if you want exact counts (see bottom of file).
WORDS_PER_TOKEN = 0.75   # i.e. chunk_size words ≈ chunk_size / 0.75 tokens


# ─────────────────────────────────────────────────────────────────
# CHUNKING_GUIDE
# ─────────────────────────────────────────────────────────────────
# Why CHUNK_SIZE = 300 tokens?
#   • MiniLM's hard limit is 256 tokens — anything over is truncated.
#   • 300 words ≈ 230 tokens — safely under the limit with room to spare.
#   • For a resume, 300 words ≈ 3–6 bullet points: enough context,
#     not so much that unrelated content bleeds in.
#
# Why CHUNK_OVERLAP = 50 tokens?
#   • Without overlap, a sentence cut at a chunk boundary loses context
#     in BOTH the chunk before and after the cut.
#   • 50-token overlap means each chunk "remembers" the tail of the
#     previous one, so retrieval is more forgiving of borderline splits.
#   • Rule of thumb: overlap ≈ 15–20% of chunk size.
#
# Why fixed-size here instead of section-aware?
#   • This script is your first working ingest — simplest possible.
#   • Once it's running, upgrade to section_aware_chunks() from
#     chunking_strategies.py for better section isolation.


# ─────────────────────────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """One piece of your resume, ready to store in ChromaDB."""
    id: str                        # unique string ID, e.g. "chunk_0"
    text: str                      # the actual text the LLM will read
    metadata: dict = field(default_factory=dict)   # section, page, word offsets

    @property
    def token_estimate(self) -> int:
        # Rough estimate: 4 chars ≈ 1 token for English text
        return len(self.text) // 4

    def __repr__(self):
        preview = self.text[:70].replace("\n", " ")
        return f"Chunk(id={self.id!r}, ~{self.token_estimate}tok, text={preview!r})"


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Read PDF
# ═══════════════════════════════════════════════════════════════════

def read_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a PDF resume using pdfplumber.

    Why pdfplumber over pypdf?
      pdfplumber preserves horizontal spacing, so bullet-point indentation
      and two-column layouts come out much cleaner. Crucial for resumes.

    Returns the full text as a single string with page breaks marked.
    """
    print(f"\n{'='*55}")
    print(f"STEP 1 — Reading PDF: {pdf_path.name}")
    print(f"{'='*55}")

    if not pdf_path.exists():
        print(f"\n  ✗ File not found: {pdf_path}")
        print(f"  Put your resume PDF in the data/ folder and try again.")
        print(f"  Example: python ingest.py --pdf data/your_resume.pdf")
        sys.exit(1)

    pages_text = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"  Pages found: {total_pages}")

        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text(
                x_tolerance=2,    # horizontal gap tolerance (pixels)
                y_tolerance=2,    # vertical gap tolerance (pixels)
            )
            if text:
                # Clean up excessive blank lines while preserving structure
                text = re.sub(r'\n{3,}', '\n\n', text)
                pages_text.append(text.strip())
                print(f"  Page {i}/{total_pages}: {len(text)} chars extracted")
            else:
                print(f"  Page {i}/{total_pages}: no text found "
                      f"(may be a scanned image — see README for OCR option)")

    full_text = "\n\n".join(pages_text)
    print(f"\n  Total extracted: {len(full_text)} chars  "
          f"(~{len(full_text)//4} tokens)")

    if not full_text.strip():
        print("\n  ✗ No text extracted. Your PDF may be image-based (scanned).")
        print("  Run OCR first: pip install pytesseract pdf2image")
        print("  Then use the ocr_pdf() function in ingest_ocr.py")
        sys.exit(1)

    return full_text


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Chunk text
# ═══════════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    chunk_size: int  = CHUNK_SIZE,
    overlap: int     = CHUNK_OVERLAP,
) -> list[Chunk]:
    """
    Split text into overlapping windows of ~chunk_size words.

    HOW IT WORKS:
      1. Split the full text into individual words.
      2. Walk a window of `chunk_size` words across them.
      3. Each step moves forward by (chunk_size - overlap) words.
      4. The overlap means consecutive chunks share their boundary words,
         so a sentence cut in half appears fully in at least one chunk.

    Visual example (chunk_size=6, overlap=2):
      words:   [A B C D E F G H I J]
      chunk 0: [A B C D E F]
      chunk 1:         [E F G H I J]   ← shares E F with chunk 0
      chunk 2:                 [I J ...]

    Metadata stored per chunk:
      word_start / word_end — position in the original word list
      token_estimate        — rough token count for debugging
      chunk_index           — sequential integer ID
    """
    print(f"\n{'='*55}")
    print(f"STEP 2 — Chunking  "
          f"(chunk_size={chunk_size} words, overlap={overlap} words)")
    print(f"{'='*55}")

    words = text.split()
    total_words = len(words)
    print(f"  Total words: {total_words}")

    chunks = []
    start = 0
    step  = chunk_size - overlap    # how far to advance each iteration

    while start < total_words:
        end        = min(start + chunk_size, total_words)
        chunk_text = " ".join(words[start:end])

        # Skip chunks that are too short to be meaningful
        # (e.g. the final fragment if the last chunk is mostly overlap)
        if len(chunk_text.split()) < 10:
            break

        chunk = Chunk(
            id       = f"chunk_{len(chunks)}",
            text     = chunk_text,
            metadata = {
                "chunk_index":     len(chunks),
                "word_start":      start,
                "word_end":        end,
                "token_estimate":  len(chunk_text) // 4,
                "source":          "resume",
            },
        )
        chunks.append(chunk)
        start += step

    print(f"  Chunks created: {len(chunks)}")
    print(f"  Avg chunk size: ~{sum(c.token_estimate for c in chunks) // len(chunks)} tokens")
    print(f"  Overlap:        {overlap} words "
          f"(≈{int(overlap / chunk_size * 100)}% of chunk size)")

    # Show a sample chunk so you can sanity-check the output
    if chunks:
        print(f"\n  Sample (chunk_0):")
        sample = chunks[0].text[:200].replace("\n", " ")
        print(f"  '{sample}...'")

    return chunks


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — Embed chunks
# ═══════════════════════════════════════════════════════════════════

def embed_chunks(chunks: list[Chunk]) -> np.ndarray:
    """
    Convert every chunk's text into a 384-dimensional vector using
    sentence-transformers all-MiniLM-L6-v2.

    WHY THIS MODEL?
      • Free, runs locally, no API key or internet needed after first download
      • ~80MB, loads in ~1s on CPU
      • 384 dimensions — small enough to store thousands of chunks in RAM
      • Max 256 tokens input — our ~300-word chunks are safely under this
      • Trained specifically for semantic similarity tasks

    WHY normalize_embeddings=True?
      Normalising scales every vector to length 1.0 (a "unit vector").
      This means cosine_similarity(A, B) == dot_product(A, B), which is
      faster to compute and what ChromaDB's cosine space expects.

    Returns: numpy array of shape (num_chunks, 384), dtype float32
    """
    print(f"\n{'='*55}")
    print(f"STEP 3 — Embedding  (model: {EMBEDDING_MODEL})")
    print(f"{'='*55}")
    print(f"  Loading model... (downloads ~80MB on first run, then cached)")

    t0    = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Model loaded in {time.time()-t0:.1f}s")
    print(f"  Output dimensions: "
          f"{model.get_sentence_embedding_dimension()}")
    print(f"  Max input tokens:  {model.max_seq_length}")
    print(f"  Embedding {len(chunks)} chunks...")

    texts = [c.text for c in chunks]

    # batch_size=32: process 32 chunks at a time.
    # Lower this (e.g. 8) if you run out of RAM.
    # show_progress_bar=True: prints a live tqdm bar.
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,   # unit vectors — required for cosine sim
        batch_size=32,
        show_progress_bar=True,
    )

    embeddings = np.array(embeddings, dtype=np.float32)

    print(f"\n  Done in {time.time()-t0:.1f}s")
    print(f"  Embedding matrix: {embeddings.shape}  "   # (num_chunks, 384)
          f"({embeddings.nbytes // 1024} KB)")

    # Sanity check: unit vectors should all have magnitude ≈ 1.0
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  Vector norms: min={norms.min():.4f}  "
          f"max={norms.max():.4f}  (should all be ≈ 1.0)")

    return embeddings


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — Store in ChromaDB
# ═══════════════════════════════════════════════════════════════════

def store_in_chromadb(
    chunks: list[Chunk],
    embeddings: np.ndarray,
) -> chromadb.Collection:
    """
    Save chunks + vectors + metadata to a persistent ChromaDB collection.

    WHY CHROMADB?
      • Zero-config local storage — just a folder on disk
      • Automatically builds an HNSW index for fast nearest-neighbour search
      • Supports metadata filtering (e.g. only search Skills section)
      • PersistentClient auto-saves on every write — no manual .save() needed

    hnsw:space="cosine":
      Tells ChromaDB to use cosine distance for similarity search.
      Must match normalize_embeddings=True in the embedding step.

    WHAT GETS STORED PER ROW:
      id         — unique string key (used for updates/deletes)
      document   — the raw text (what the LLM will read)
      embedding  — the 384-float vector (what similarity search uses)
      metadata   — dict for filtering (chunk_index, word offsets, etc.)
    """
    print(f"\n{'='*55}")
    print(f"STEP 4 — Storing in ChromaDB  (path: {CHROMA_DB_PATH})")
    print(f"{'='*55}")

    # PersistentClient: data survives restarts.
    # The chroma_db/ folder is created if it doesn't exist.
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Wipe and recreate the collection so re-running ingest is idempotent.
    # (If you want to ADD to an existing collection, remove these two lines
    #  and use get_or_create_collection instead.)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing '{COLLECTION_NAME}' collection (re-indexing)")
    except Exception:
        pass   # collection didn't exist yet — that's fine

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        # hnsw:space="cosine" — use cosine distance (matches normalised vecs)
        # alternative: "l2" for Euclidean, "ip" for inner product
    )
    print(f"  Created collection '{COLLECTION_NAME}'")

    # Add in batches of 100 to avoid memory spikes on large documents
    batch_size = 100
    total      = len(chunks)

    for batch_start in tqdm(
        range(0, total, batch_size),
        desc="  Storing batches",
        unit="batch",
    ):
        batch_end   = min(batch_start + batch_size, total)
        batch       = chunks[batch_start:batch_end]
        batch_vecs  = embeddings[batch_start:batch_end]

        collection.add(
            ids        = [c.id for c in batch],
            documents  = [c.text for c in batch],
            embeddings = batch_vecs.tolist(),   # numpy → plain Python list
            metadatas  = [c.metadata for c in batch],
        )

    final_count = collection.count()
    print(f"\n  Stored {final_count} chunks  ✓")

    # Quick smoke-test: retrieve the most similar chunk to a known phrase
    print(f"\n  Smoke test — querying for 'skills and experience':")
    test_model = SentenceTransformer(EMBEDDING_MODEL)
    q_vec = test_model.encode(["skills and experience"],
                               normalize_embeddings=True)
    results = collection.query(
        query_embeddings=q_vec.tolist(),
        n_results=2,
        include=["documents", "distances"],
    )
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        sim = 1 - dist
        preview = doc[:70].replace("\n", " ")
        print(f"  similarity={sim:.3f}  '{preview}...'")

    return collection


# ═══════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════

def ingest(pdf_path: Path) -> None:
    """
    Run the complete four-step indexing pipeline.
    Call this once before starting chat.py.
    """
    t_start = time.time()

    print("\n" + "█" * 55)
    print("  RESUME RAG BOT — Indexing Pipeline")
    print("█" * 55)
    print(f"  PDF   : {pdf_path}")
    print(f"  Model : {EMBEDDING_MODEL}")
    print(f"  DB    : {CHROMA_DB_PATH}")
    print(f"  Chunk : {CHUNK_SIZE} words, {CHUNK_OVERLAP} overlap")

    # ── STEP 1: Read ──────────────────────────────────────────────
    raw_text = read_pdf(pdf_path)

    # ── STEP 2: Chunk ─────────────────────────────────────────────
    chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)

    # ── STEP 3: Embed ─────────────────────────────────────────────
    embeddings = embed_chunks(chunks)

    # ── STEP 4: Store ─────────────────────────────────────────────
    store_in_chromadb(chunks, embeddings)

    # ── Done ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'█'*55}")
    print(f"  ✓ Indexing complete in {elapsed:.1f}s")
    print(f"  {len(chunks)} chunks stored in {CHROMA_DB_PATH}/")
    print(f"{'█'*55}")
    print(f"""
  NEXT STEPS:
    1. Run retrieve.py to test retrieval:
         python retrieve.py

    2. Build chat.py to add the LLM layer:
         See Phase 4 of the learning guide

    3. Upgrade to section-aware chunking (better for resumes):
         Replace chunk_text() with section_aware_chunks()
         from chunking_strategies.py
""")


# ═══════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a resume PDF into ChromaDB for RAG."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("Data/Resume/resume.pdf"),
        help="Path to your resume PDF (default: Data/Resume/resume.pdf)",
    )
    args = parser.parse_args()
    ingest(args.pdf)


# ═══════════════════════════════════════════════════════════════════
# APPENDIX — Upgrade: real tokenizer instead of word estimate
# ═══════════════════════════════════════════════════════════════════
#
# The current chunker counts *words* as a proxy for tokens.
# This is fine for learning but slightly inaccurate.
# To count exact tokens, replace chunk_text() with this version:
#
#   from transformers import AutoTokenizer
#   _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
#
#   def count_tokens(text: str) -> int:
#       return len(_tokenizer.encode(text, add_special_tokens=False))
#
#   def chunk_text_exact(text, chunk_size=300, overlap=50):
#       tokens = _tokenizer.encode(text, add_special_tokens=False)
#       chunks = []
#       step   = chunk_size - overlap
#       for start in range(0, len(tokens), step):
#           end        = min(start + chunk_size, len(tokens))
#           chunk_toks = tokens[start:end]
#           chunk_text = _tokenizer.decode(chunk_toks)
#           if len(chunk_toks) < 10:
#               break
#           chunks.append(Chunk(
#               id       = f"chunk_{len(chunks)}",
#               text     = chunk_text,
#               metadata = {"chunk_index": len(chunks),
#                           "token_start": start,
#                           "token_end":   end},
#           ))
#       return chunks
