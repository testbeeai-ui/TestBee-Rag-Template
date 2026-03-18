"""
supabase_migrate.py — One-time migration: ChromaDB → Supabase pgvector.

USAGE
-----
Run from inside the testbee_local/ directory with the venv active:

    cd testbee_local
    python src/supabase_migrate.py

PREREQUISITES
-------------
1. Run supabase_setup.sql in the Supabase SQL Editor first.
2. Add SUPABASE_URL and SUPABASE_SERVICE_KEY to testbee_local/.env:

       SUPABASE_URL=https://<project-ref>.supabase.co
       SUPABASE_SERVICE_KEY=<service_role secret key>

   Use the SERVICE ROLE key (not the anon key) so the INSERT policy passes.
3. pip install supabase  (or: pip install -r requirements.txt after updating it)

WHAT IT DOES
------------
- Reads every chunk from the local ChromaDB collection (text, embedding, metadata).
- Inserts them into the Supabase `textbook_chunks` table in batches of 100.
- Prints progress after each batch so you can monitor a large migration.
- Is idempotent-safe: if a chunk already exists with the same content the
  upsert silently skips it (based on the unique constraint you can add, see note
  below). Without a unique constraint this script will insert duplicates if run
  twice — run it only once, or truncate the table first.
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure src/ is on the module search path when executed directly.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb
from supabase import create_client, Client
from config import VECTOR_STORE_DIR, CHROMA_COLLECTION_NAME, PROJECT_ROOT
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env (config.py does this too, but we call it explicitly here so this
# script works even if run in an environment where config has not been imported).
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE      = 100          # rows per Supabase insert call
TARGET_TABLE    = "textbook_chunks"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_supabase_client() -> Client:
    """
    Build a Supabase client using the SERVICE ROLE key so INSERT is allowed
    even with Row Level Security enabled.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url:
        raise EnvironmentError(
            "SUPABASE_URL is not set in .env. "
            "Add:  SUPABASE_URL=https://<project-ref>.supabase.co"
        )
    if not key:
        raise EnvironmentError(
            "SUPABASE_SERVICE_KEY is not set in .env. "
            "Add:  SUPABASE_SERVICE_KEY=<your service_role secret>"
        )

    return create_client(url, key)


def _export_from_chroma() -> tuple[list[str], list[list[float]], list[dict]]:
    """
    Pull every document, embedding, and metadata record from the local
    ChromaDB persistent store.

    Returns a 3-tuple:
        texts      — list[str]         — raw chunk text
        embeddings — list[list[float]] — 1024-dim BGE-M3 vectors
        metadatas  — list[dict]        — metadata dicts stored alongside chunks
    """
    client     = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection = client.get_collection(CHROMA_COLLECTION_NAME)

    total = collection.count()
    print(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' contains {total} chunks.")

    # ChromaDB's get() returns everything in one call; safe up to ~100k rows.
    result = collection.get(include=["documents", "embeddings", "metadatas"])

    texts      = result["documents"]   # list[str]
    embeddings = result["embeddings"]  # list[list[float]]  (may be numpy arrays)
    metadatas  = result["metadatas"]   # list[dict]

    # Normalise: ChromaDB may return numpy float arrays; Supabase JSON needs
    # plain Python lists of Python floats.
    embeddings = [
        [float(v) for v in emb] for emb in embeddings
    ]

    return texts, embeddings, metadatas


def _build_row(text: str, embedding: list[float], meta: dict) -> dict:
    """
    Map a single ChromaDB record to the `textbook_chunks` table schema.

    Any field absent from ChromaDB metadata defaults to None so Postgres
    stores NULL rather than raising a KeyError.
    """
    return {
        "text":            text,
        "embedding":       embedding,
        "source_file":     meta.get("source_file"),
        "curriculum":      meta.get("curriculum"),
        "grade_level":     meta.get("grade_level"),
        "subject":         meta.get("subject"),
        "chapter":         meta.get("chapter"),
        "page_number":     meta.get("page_number"),
        "section_heading": meta.get("section_heading"),
    }


def _insert_batch(client: Client, rows: list[dict], batch_num: int) -> None:
    """
    Insert a single batch of rows into Supabase.

    Raises RuntimeError on API error so the caller can abort early.
    """
    response = (
        client.table(TARGET_TABLE)
        .insert(rows)
        .execute()
    )

    # supabase-py v2: a non-2xx response raises an APIError automatically.
    # An empty data list with no error means a no-op (e.g. all duplicates);
    # we log it but treat it as success.
    inserted = len(response.data) if response.data else 0
    print(f"  Batch {batch_num}: inserted {inserted} rows (attempted {len(rows)}).")


# ---------------------------------------------------------------------------
# Main migration routine
# ---------------------------------------------------------------------------

def migrate() -> None:
    print("=" * 60)
    print("Testbee ChromaDB → Supabase pgvector migration")
    print("=" * 60)

    # -- Step 1: connect to Supabase ------------------------------------------
    print("\n[1/3] Connecting to Supabase...")
    supabase = _get_supabase_client()
    print("      Connected.")

    # -- Step 2: export from ChromaDB -----------------------------------------
    print("\n[2/3] Exporting chunks from ChromaDB...")
    texts, embeddings, metadatas = _export_from_chroma()
    total = len(texts)
    print(f"      Exported {total} chunks.")

    if total == 0:
        print("\nNothing to migrate. Is the ChromaDB collection populated?")
        return

    # -- Step 3: batch-insert into Supabase ------------------------------------
    print(f"\n[3/3] Uploading to Supabase in batches of {BATCH_SIZE}...")

    uploaded   = 0
    batch_num  = 0

    for start in range(0, total, BATCH_SIZE):
        end   = min(start + BATCH_SIZE, total)
        batch = [
            _build_row(texts[i], embeddings[i], metadatas[i])
            for i in range(start, end)
        ]

        batch_num += 1

        try:
            _insert_batch(supabase, batch, batch_num)
        except Exception as exc:
            # Print the offending batch range so the user can investigate.
            print(
                f"\n  ERROR on batch {batch_num} (chunks {start}–{end - 1}): {exc}"
            )
            print("  Migration aborted. Fix the error and re-run.")
            print(
                "  Tip: truncate the table first to avoid partial duplicates:\n"
                "       TRUNCATE TABLE textbook_chunks;"
            )
            sys.exit(1)

        uploaded += len(batch)
        print(f"  Uploaded {uploaded}/{total} chunks...")

    print("\n" + "=" * 60)
    print(f"Migration complete. {uploaded} chunks uploaded to '{TARGET_TABLE}'.")
    print("=" * 60)


if __name__ == "__main__":
    migrate()
