
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



EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH   = "./chroma_db"
COLLECTION_NAME  = "resume"
CHUNK_SIZE    = 300   # target tokens per chunk
CHUNK_OVERLAP = 50    


WORDS_PER_TOKEN = 0.75   # i.e. chunk_size words ≈ chunk_size / 0.75 toke

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

def read_pdf(pdf_path: Path) -> str:
    
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

def chunk_text(
    text: str,
    chunk_size: int  = CHUNK_SIZE,
    overlap: int     = CHUNK_OVERLAP,
) -> list[Chunk]:
    
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

    
    if chunks:
        print(f"\n  Sample (chunk_0):")
        sample = chunks[0].text[:200].replace("\n", " ")
        print(f"  '{sample}...'")

    return chunks

def embed_chunks(chunks: list[Chunk]) -> np.ndarray:
  
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

def store_in_chromadb(
    chunks: list[Chunk],
    embeddings: np.ndarray,
) -> chromadb.Collection:
    
    print(f"\n{'='*55}")
    print(f"STEP 4 — Storing in ChromaDB  (path: {CHROMA_DB_PATH})")
    print(f"{'='*55}")
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing '{COLLECTION_NAME}' collection (re-indexing)")
    except Exception:
        pass   # collection didn't exist yet — that's fine

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        
    )
    print(f"  Created collection '{COLLECTION_NAME}'")

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

def ingest(pdf_path: Path) -> None:
    t_start = time.time()

    print("\n" + "█" * 55)
    print("  RESUME RAG BOT — Indexing Pipeline")
    print("█" * 55)
    print(f"  PDF   : {pdf_path}")
    print(f"  Model : {EMBEDDING_MODEL}")
    print(f"  DB    : {CHROMA_DB_PATH}")
    print(f"  Chunk : {CHUNK_SIZE} words, {CHUNK_OVERLAP} overlap")
    raw_text = read_pdf(pdf_path)
    chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
    embeddings = embed_chunks(chunks)
    store_in_chromadb(chunks, embeddings)
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


