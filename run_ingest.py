"""
run_ingest.py — Smart ingestion runner for Testbee.

FOLDER STRUCTURE (drop PDFs here, the script detects everything automatically):

    data/raw_pdfs/
    └── cbse/
        ├── class11/
        │   ├── physics/       ← drop Class 11 Physics PDFs here
        │   ├── chemistry/     ← drop Class 11 Chemistry PDFs here
        │   └── maths/         ← drop Class 11 Maths PDFs here
        └── class12/
            ├── physics/
            ├── chemistry/
            └── maths/

AUTO-DETECTION from folder path:
    cbse/class11/physics/keph101.pdf
    → curriculum  = "CBSE"
    → grade_level = 11
    → subject     = "Physics"
    → chapter     = extracted from PDF's first heading (fallback: filename)

UPLOAD TARGET:
    - If SUPABASE_URL + SUPABASE_SERVICE_KEY present in .env → uploads to Supabase
    - Otherwise → uploads to local ChromaDB (for testing)

Run from testbee_local/:
    python run_ingest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from config import DocumentMetadata, RAW_PDFS_DIR
from ingest import ingest_document
from embed import load_model, embed_chunks

# ---------------------------------------------------------------------------
# NCERT chapter name lookup — keyed by filename stem (no extension).
# Used as primary chapter name source.  Docling heading is fallback.
# Add new entries here as you add more PDFs.
# ---------------------------------------------------------------------------
CHAPTER_MAP: dict[str, str] = {
    # --- CBSE Class 11 Physics ---
    "keph101": "Units and Measurement",
    "keph102": "Motion in a Straight Line",
    "keph103": "Motion in a Plane",
    "keph104": "Laws of Motion",
    "keph105": "Work, Energy and Power",
    "keph106": "Systems of Particles and Rotational Motion",
    "keph107": "Gravitation",
    "keph201": "Mechanical Properties of Solids",
    "keph202": "Mechanical Properties of Fluids",
    "keph203": "Thermal Properties of Matter",
    "keph204": "Thermodynamics",
    "keph205": "Kinetic Theory",
    "keph206": "Oscillations",
    "keph207": "Waves",
    "keph1a1": "Appendices Part 1",
    "keph1an": "Answers Part 1",
    "keph1ps": "Prelims Part 1",
    "keph2an": "Answers Part 2",
    "keph2ps": "Prelims Part 2",

    # --- CBSE Class 12 Physics (add when you drop the PDFs) ---
    "leph101": "Electric Charges and Fields",
    "leph102": "Electrostatic Potential and Capacitance",
    "leph103": "Current Electricity",
    "leph104": "Moving Charges and Magnetism",
    "leph105": "Magnetism and Matter",
    "leph106": "Electromagnetic Induction",
    "leph107": "Alternating Current",
    "leph108": "Electromagnetic Waves",
    "leph201": "Ray Optics and Optical Instruments",
    "leph202": "Wave Optics",
    "leph203": "Dual Nature of Radiation and Matter",
    "leph204": "Atoms",
    "leph205": "Nuclei",
    "leph206": "Semiconductor Electronics",

    # --- CBSE Class 11 Chemistry (add when ready) ---
    "kech101": "Some Basic Concepts of Chemistry",
    "kech102": "Structure of Atom",
    "kech103": "Classification of Elements and Periodicity",
    "kech104": "Chemical Bonding and Molecular Structure",
    "kech105": "Thermodynamics",
    "kech106": "Equilibrium",
    "kech107": "Redox Reactions",
    "kech108": "Organic Chemistry: Basic Principles",
    "kech109": "Hydrocarbons",

    # --- CBSE Class 12 Chemistry (add when ready) ---
    "lech101": "Solutions",
    "lech102": "Electrochemistry",
    "lech103": "Chemical Kinetics",
    "lech104": "d and f Block Elements",
    "lech105": "Coordination Compounds",
    "lech106": "Haloalkanes and Haloarenes",
    "lech107": "Alcohols, Phenols and Ethers",
    "lech108": "Aldehydes, Ketones and Carboxylic Acids",

    # --- CBSE Class 11 Maths (add when ready) ---
    "kemh101": "Sets",
    "kemh102": "Relations and Functions",
    "kemh103": "Trigonometric Functions",
    "kemh104": "Complex Numbers and Quadratic Equations",
    "kemh105": "Linear Inequalities",
    "kemh106": "Permutations and Combinations",
    "kemh107": "Binomial Theorem",
    "kemh108": "Sequences and Series",
    "kemh109": "Straight Lines",
    "kemh110": "Conic Sections",
    "kemh111": "Introduction to Three Dimensional Geometry",
    "kemh112": "Limits and Derivatives",
    "kemh113": "Statistics",
    "kemh114": "Probability",

    # --- CBSE Class 12 Maths (add when ready) ---
    "lemh101": "Relations and Functions",
    "lemh102": "Inverse Trigonometric Functions",
    "lemh103": "Matrices",
    "lemh104": "Determinants",
    "lemh105": "Continuity and Differentiability",
    "lemh106": "Application of Derivatives",
    "lemh107": "Integrals",
    "lemh108": "Application of Integrals",
    "lemh109": "Differential Equations",
    "lemh110": "Vector Algebra",
    "lemh111": "Three Dimensional Geometry",
    "lemh112": "Linear Programming",
    "lemh113": "Probability",
}


# ---------------------------------------------------------------------------
# Auto-detect metadata from folder path
# ---------------------------------------------------------------------------

def detect_metadata(pdf_path: Path) -> DocumentMetadata | None:
    """
    Derive curriculum, grade_level, subject from the PDF's folder path.

    Expected path pattern (relative to raw_pdfs/):
        <curriculum>/<gradeX>/<subject>/<file>.pdf

    Example:
        cbse/class11/physics/keph101.pdf
        → curriculum="CBSE", grade_level=11, subject="Physics"

    Returns None and prints a warning if the path doesn't match the pattern.
    """
    try:
        # Parts relative to raw_pdfs/
        rel = pdf_path.relative_to(RAW_PDFS_DIR)
        parts = rel.parts  # e.g. ('cbse', 'class11', 'physics', 'keph101.pdf')

        if len(parts) != 4:
            print(f"  [SKIP] {pdf_path.name} — expected path: curriculum/classXX/subject/file.pdf")
            return None

        curriculum_raw, grade_raw, subject_raw, _ = parts

        curriculum  = curriculum_raw.upper()               # "cbse" → "CBSE"
        grade_level = int(grade_raw.lower().replace("class", ""))  # "class11" → 11
        subject     = subject_raw.capitalize()             # "physics" → "Physics"

        stem    = pdf_path.stem.lower()                   # "keph101"
        chapter = CHAPTER_MAP.get(stem, pdf_path.stem)   # lookup or fallback to filename

        return DocumentMetadata(
            source_file=pdf_path.name,
            curriculum=curriculum,
            grade_level=grade_level,
            subject=subject,
            chapter=chapter,
        )
    except Exception as exc:
        print(f"  [SKIP] {pdf_path.name} — could not detect metadata: {exc}")
        return None


# ---------------------------------------------------------------------------
# Upload backends
# ---------------------------------------------------------------------------

def upload_to_supabase(all_rows: list[dict]) -> None:
    """Batch-upload to Supabase textbook_chunks table."""
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")

    client = create_client(url, key)
    batch_size = 100
    total = len(all_rows)

    print(f"\nUploading {total} chunks to Supabase in batches of {batch_size}...")
    for i in range(0, total, batch_size):
        batch = all_rows[i : i + batch_size]
        client.table("textbook_chunks").insert(batch).execute()
        print(f"  Uploaded {min(i + batch_size, total)}/{total} chunks...")

    print(f"Supabase upload complete. {total} chunks stored.")


def upload_to_chromadb(all_rows: list[dict]) -> None:
    """Fallback: store locally in ChromaDB."""
    from retrieve import get_or_create_collection, add_to_collection

    collection = get_or_create_collection()
    total = len(all_rows)
    print(f"\nUploading {total} chunks to ChromaDB (local)...")

    # Group by source_file for add_to_collection
    from itertools import groupby
    rows_by_file = {}
    for row in all_rows:
        src = row["source_file"]
        rows_by_file.setdefault(src, []).append(row)

    for src, rows in rows_by_file.items():
        texts      = [r["text"] for r in rows]
        embeddings = [r["embedding"] for r in rows]
        metadatas  = [{k: v for k, v in r.items() if k not in ("text", "embedding")} for r in rows]
        add_to_collection(collection, texts, embeddings, metadatas, src)
        print(f"  Stored {len(rows)} chunks from {src}")

    print(f"ChromaDB upload complete. Total: {collection.count()} chunks.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Testbee Ingestion Runner")
    print("=" * 60)

    # Detect upload target
    use_supabase = bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"))
    target = "Supabase" if use_supabase else "ChromaDB (local)"
    print(f"\nUpload target: {target}")

    # Discover all PDFs
    all_pdfs = sorted(RAW_PDFS_DIR.rglob("*.pdf"))
    print(f"Found {len(all_pdfs)} PDFs under {RAW_PDFS_DIR}\n")

    if not all_pdfs:
        print("No PDFs found. Drop your textbook PDFs into:")
        print("  data/raw_pdfs/cbse/class11/physics/")
        print("  data/raw_pdfs/cbse/class11/chemistry/")
        print("  data/raw_pdfs/cbse/class11/maths/")
        print("  data/raw_pdfs/cbse/class12/physics/  ... etc.")
        return

    # Load BGE-M3 model once
    print("Loading BGE-M3 model...")
    model = load_model()
    print("Model loaded.\n")

    all_rows: list[dict] = []

    for pdf_path in all_pdfs:
        print(f"[{pdf_path.relative_to(RAW_PDFS_DIR)}]")

        metadata = detect_metadata(pdf_path)
        if metadata is None:
            continue

        print(f"  curriculum={metadata.curriculum}  grade={metadata.grade_level}"
              f"  subject={metadata.subject}  chapter={metadata.chapter}")

        print(f"  Ingesting with Docling...")
        chunks = ingest_document(pdf_path, metadata)
        print(f"  {len(chunks)} chunks extracted.")

        if not chunks:
            print(f"  WARNING: no chunks — skipping.\n")
            continue

        texts     = [c[0] for c in chunks]
        metadatas = [c[1] for c in chunks]

        print(f"  Embedding {len(texts)} chunks...")
        embeddings = embed_chunks(texts, model)

        for text, meta, emb in zip(texts, metadatas, embeddings):
            all_rows.append({
                "text":            text,
                "embedding":       emb,
                "source_file":     meta.get("source_file", ""),
                "curriculum":      meta.get("curriculum", ""),
                "grade_level":     meta.get("grade_level", 0),
                "subject":         meta.get("subject", ""),
                "chapter":         meta.get("chapter", ""),
                "page_number":     meta.get("page_number", -1),
                "section_heading": meta.get("section_heading", ""),
            })

        print(f"  Done. {len(texts)} chunks ready.\n")

    if not all_rows:
        print("No chunks produced. Check your PDFs.")
        return

    print(f"\nTotal chunks ready to upload: {len(all_rows)}")

    if use_supabase:
        upload_to_supabase(all_rows)
    else:
        upload_to_chromadb(all_rows)

    print("\n" + "=" * 60)
    print(f"Ingestion complete. {len(all_rows)} chunks uploaded to {target}.")
    print("=" * 60)


if __name__ == "__main__":
    main()
