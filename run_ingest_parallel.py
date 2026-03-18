"""
run_ingest_parallel.py — Parallel ingestion runner.

Strategy:
  Phase 1 (PARALLEL)   — Docling converts all PDFs simultaneously using multiprocessing.
  Phase 2 (SEQUENTIAL) — Embed + store each file's chunks into ChromaDB one by one.

This avoids ChromaDB concurrent write conflicts while maximising CPU usage on the slow step.
"""

import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import DocumentMetadata, RAW_PDFS_DIR, VECTOR_STORE_DIR, CHROMA_COLLECTION_NAME
from embed import load_model, embed_chunks
from retrieve import get_or_create_collection, add_to_collection
import chromadb

CBSE_PHYSICS = RAW_PDFS_DIR / "cbse" / "physics"

PDF_METADATA: dict[str, DocumentMetadata] = {
    "keph101.pdf": DocumentMetadata(source_file="keph101.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Units and Measurement"),
    "keph102.pdf": DocumentMetadata(source_file="keph102.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Motion in a Straight Line"),
    "keph103.pdf": DocumentMetadata(source_file="keph103.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Motion in a Plane"),
    "keph104.pdf": DocumentMetadata(source_file="keph104.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Laws of Motion"),
    "keph105.pdf": DocumentMetadata(source_file="keph105.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Work Energy and Power"),
    "keph106.pdf": DocumentMetadata(source_file="keph106.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Systems of Particles and Rotational Motion"),
    "keph107.pdf": DocumentMetadata(source_file="keph107.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Gravitation"),
    "keph201.pdf": DocumentMetadata(source_file="keph201.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Mechanical Properties of Solids"),
    "keph202.pdf": DocumentMetadata(source_file="keph202.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Mechanical Properties of Fluids"),
    "keph203.pdf": DocumentMetadata(source_file="keph203.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Thermal Properties of Matter"),
    "keph204.pdf": DocumentMetadata(source_file="keph204.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Thermodynamics"),
    "keph205.pdf": DocumentMetadata(source_file="keph205.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Kinetic Theory"),
    "keph206.pdf": DocumentMetadata(source_file="keph206.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Oscillations"),
    "keph207.pdf": DocumentMetadata(source_file="keph207.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Waves"),
    "keph1a1.pdf": DocumentMetadata(source_file="keph1a1.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Appendices Part 1"),
    "keph1an.pdf": DocumentMetadata(source_file="keph1an.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Answers Part 1"),
    "keph1ps.pdf": DocumentMetadata(source_file="keph1ps.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Cover Part 1"),
    "keph2an.pdf": DocumentMetadata(source_file="keph2an.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Answers Part 2"),
    "keph2ps.pdf": DocumentMetadata(source_file="keph2ps.pdf", curriculum="CBSE", grade_level=11, subject="Physics", chapter="Cover Part 2"),
}


def get_already_stored() -> set[str]:
    """Query ChromaDB and return set of source_file names already fully stored."""
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    col = client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    if col.count() == 0:
        return set()
    results = col.get(include=["metadatas"])
    return set(m["source_file"] for m in results["metadatas"])


def convert_one(args):
    """Worker function — runs in a separate process. Converts one PDF via Docling."""
    pdf_path_str, metadata_dict = args
    # Re-import inside worker process (required for multiprocessing)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from ingest import ingest_document
    from config import DocumentMetadata

    pdf_path = Path(pdf_path_str)
    metadata = DocumentMetadata(**metadata_dict)
    chunks = ingest_document(pdf_path, metadata)
    return pdf_path.name, chunks


def main():
    cpu_count = os.cpu_count() or 2
    # Use half the CPUs so the machine stays responsive; minimum 2 workers
    workers = max(2, cpu_count // 2)

    # ── Step 0: Check what's already done ──────────────────────────────────
    print("Checking ChromaDB for already-stored files...")
    already_done = get_already_stored()
    print(f"  Already stored: {sorted(already_done)}\n")

    remaining = [
        (str(CBSE_PHYSICS / fname), meta.model_dump())
        for fname, meta in PDF_METADATA.items()
        if fname not in already_done
    ]

    if not remaining:
        print("All files already ingested. Nothing to do.")
        return

    print(f"Files to process: {len(remaining)} | Parallel workers: {workers}\n")
    for path_str, _ in remaining:
        print(f"  Queued: {Path(path_str).name}")
    print()

    # ── Step 1: PARALLEL Docling conversion ────────────────────────────────
    print("=" * 60)
    print("PHASE 1 — Parallel Docling conversion (all PDFs at once)")
    print("=" * 60)

    results: dict[str, list] = {}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_name = {
            executor.submit(convert_one, args): Path(args[0]).name
            for args in remaining
        }
        for future in as_completed(future_to_name):
            fname = future_to_name[future]
            try:
                name, chunks = future.result()
                results[name] = chunks
                print(f"  [DONE] {name} — {len(chunks)} chunks")
            except Exception as e:
                print(f"  [ERROR] {fname} — {e}")
                results[fname] = []

    print(f"\nAll conversions complete. Total files ready: {len(results)}\n")

    # ── Step 2: SEQUENTIAL embed + store ───────────────────────────────────
    print("=" * 60)
    print("PHASE 2 — Embedding + storing into ChromaDB (sequential)")
    print("=" * 60)

    print("\nLoading BGE-M3 model...")
    model = load_model()
    print("Model loaded.\n")

    collection = get_or_create_collection()

    for fname, chunks in results.items():
        if not chunks:
            print(f"[SKIP] {fname} — 0 chunks")
            continue

        texts     = [c[0] for c in chunks]
        metadatas = [c[1] for c in chunks]

        print(f"[{fname}] Embedding {len(texts)} chunks...")
        embeddings = embed_chunks(texts, model)

        print(f"[{fname}] Storing in ChromaDB...")
        add_to_collection(collection, texts, embeddings, metadatas, fname)
        print(f"[{fname}] Done. {len(texts)} chunks stored.\n")

    total = collection.count()
    print(f"\nIngestion complete. Total chunks in ChromaDB: {total}")


if __name__ == "__main__":
    main()
