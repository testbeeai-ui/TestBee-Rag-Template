"""
supabase_retrieve.py — Drop-in replacement for retrieve.py using Supabase pgvector.

INTERFACE CONTRACT
------------------
This module exposes the same two public functions that telegram_bot.py calls:

    get_or_create_collection()  →  returns an opaque "collection" handle
    query_collection(collection, query_embedding, filters, n_results)  →  list[dict]

The collection handle here is simply the Supabase Client object; no ChromaDB
collection concept exists on the Supabase side.  telegram_bot.py does not
inspect the handle at all — it only passes it back into query_collection() —
so this substitution is transparent.

RETURN FORMAT
-------------
query_collection() returns:
    [
        {
            "text":     str,   — raw passage text
            "metadata": dict,  — {source_file, curriculum, grade_level, subject,
                                   chapter, page_number, section_heading}
            "distance": float, — cosine SIMILARITY (1.0 = perfect match)
        },
        ...
    ]

The distance convention (higher = more relevant) matches what the Postgres
match_chunks function returns:  1 - (embedding <=> query_embedding).

generate.py filters on `distance <= 0.55`, which was calibrated against
ChromaDB's cosine DISTANCE (lower = better).  Cosine similarity and cosine
distance are related by:  similarity = 1 - distance.

So a ChromaDB distance of 0.55  ↔  a Supabase similarity of 0.45.
The threshold in generate.py will need updating once you switch fully, but the
data shape returned here is identical so both backends can co-exist during
a gradual rollout.  See MIGRATION_NOTE.md if you want to track this.

SWITCHING THE BOT
-----------------
In telegram_bot.py, replace:

    import retrieve

with:

    import supabase_retrieve as retrieve

Everything else stays the same.

PREREQUISITES
-------------
- SUPABASE_URL and SUPABASE_KEY must be in testbee_local/.env.
  Use the anon key here (read-only queries during bot operation).
- supabase_setup.sql must have been run in Supabase to create the table and
  the match_chunks RPC function.
"""

import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure src/ is on the module path when executed directly.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from supabase import create_client, Client
from config import PROJECT_ROOT
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Module-level singleton: create the client once, reuse on every query.
# ---------------------------------------------------------------------------
_supabase_client: Client | None = None


def _get_client() -> Client:
    """
    Return a cached Supabase client, creating it on first call.

    Uses SUPABASE_URL + SUPABASE_KEY (anon key) from .env.
    The anon key is sufficient for SELECT queries when Row Level Security
    is configured with `allow_anon_select` as defined in supabase_setup.sql.
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url:
        raise EnvironmentError(
            "SUPABASE_URL is not set in .env. "
            "Add:  SUPABASE_URL=https://<project-ref>.supabase.co"
        )
    if not key:
        raise EnvironmentError(
            "SUPABASE_KEY is not set in .env. "
            "Add:  SUPABASE_KEY=<your anon public key>"
        )

    _supabase_client = create_client(url, key)
    return _supabase_client


# ---------------------------------------------------------------------------
# Public API — mirrors retrieve.py exactly
# ---------------------------------------------------------------------------

def get_or_create_collection() -> Client:
    """
    Return the Supabase client as the "collection" handle.

    This stub exists so telegram_bot.py can call:
        collection = retrieve.get_or_create_collection()
    without modification.  The returned object is opaque to the caller; it is
    passed straight back into query_collection().
    """
    return _get_client()


def query_collection(
    collection: Any,           # Client (the return value of get_or_create_collection)
    query_embedding: list[float],
    filters: dict | None = None,
    n_results: int = 5,
) -> list[dict]:
    """
    Query Supabase pgvector for the most relevant chunks.

    Calls the match_chunks Postgres function via Supabase RPC.
    Filters on grade_level and subject only — curriculum is excluded
    intentionally (see telegram_bot.py comment; curriculum affects prompt
    depth, not vector retrieval).

    Args:
        collection:      Supabase Client (ignored beyond being passed in;
                         the module-level client is used for type safety).
        query_embedding: 1024-dim BGE-M3 vector for the user's question.
        filters:         dict with optional keys "grade_level" and "subject".
                         Any other keys are silently ignored.
        n_results:       Maximum number of chunks to return.

    Returns:
        List of dicts, each with keys:
            "text"     — passage text
            "metadata" — dict of all chunk metadata fields
            "distance" — cosine similarity score (float, higher = more relevant)

    Raises:
        RuntimeError if the RPC call fails.
    """
    client: Client = _get_client()

    # -- Extract supported filter values from the filters dict ----------------
    grade_level: int  | None = None
    subject:     str  | None = None

    if filters:
        grade_level = filters.get("grade_level")
        subject     = filters.get("subject")

    # -- Validate required filters --------------------------------------------
    # match_chunks requires both grade_filter and subject_filter.  If either
    # is absent the function would scan the full table unfiltered, which would
    # return irrelevant results.  Raise early with a clear message.
    if grade_level is None:
        raise ValueError(
            "query_collection: 'grade_level' is required in filters. "
            "Pass: filters={'grade_level': 11, 'subject': 'Physics'}"
        )
    if subject is None:
        raise ValueError(
            "query_collection: 'subject' is required in filters. "
            "Pass: filters={'grade_level': 11, 'subject': 'Physics'}"
        )

    # -- Call the Postgres RPC -------------------------------------------------
    try:
        response = client.rpc(
            "match_chunks",
            {
                "query_embedding": query_embedding,
                "grade_filter":    grade_level,
                "subject_filter":  subject,
                "match_count":     n_results,
            },
        ).execute()
    except Exception as exc:
        raise RuntimeError(
            f"Supabase RPC 'match_chunks' failed: {exc}"
        ) from exc

    rows = response.data or []

    # -- Reshape into the same format as retrieve.query_collection() ----------
    output: list[dict] = []
    for row in rows:
        metadata = {
            "source_file":     row.get("source_file"),
            "curriculum":      row.get("curriculum"),
            "grade_level":     row.get("grade_level"),
            "subject":         row.get("subject"),
            "chapter":         row.get("chapter"),
            "page_number":     row.get("page_number"),
            "section_heading": row.get("section_heading"),
        }
        output.append(
            {
                "text":     row.get("text", ""),
                "metadata": metadata,
                "distance": float(row.get("distance", 0.0)),
            }
        )

    return output
