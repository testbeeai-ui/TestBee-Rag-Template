-- =============================================================================
-- supabase_setup.sql
-- Run this once in the Supabase SQL Editor (Dashboard → SQL Editor → New Query).
-- Order matters: enable the extension before creating any vector columns.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- 1. Enable pgvector
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;


-- ---------------------------------------------------------------------------
-- 2. Drop existing table if you need a clean re-run (idempotent pattern).
--    Comment this line out once the table is populated to avoid accidental loss.
-- ---------------------------------------------------------------------------
-- DROP TABLE IF EXISTS textbook_chunks;


-- ---------------------------------------------------------------------------
-- 3. Core table
--    embedding is vector(1024) to match BGE-M3 output dimensions.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS textbook_chunks (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    text            TEXT        NOT NULL,
    embedding       vector(1024),
    source_file     TEXT,
    curriculum      TEXT,
    grade_level     INTEGER,
    subject         TEXT,
    chapter         TEXT,
    page_number     INTEGER,
    section_heading TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);


-- ---------------------------------------------------------------------------
-- 4. HNSW index for approximate nearest-neighbour cosine similarity search.
--    HNSW has faster query times than IVFFlat for small-to-medium datasets
--    (< 1M rows) at the cost of slightly higher memory and build time.
--
--    ef_construction=128, m=16 are safe defaults; tune upward if recall drops.
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS textbook_chunks_embedding_idx
    ON textbook_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (ef_construction = 128, m = 16);


-- ---------------------------------------------------------------------------
-- 5. Supporting B-tree indexes for the metadata pre-filters used in queries.
--    These allow Postgres to prune rows cheaply before the vector scan.
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS textbook_chunks_grade_subject_idx
    ON textbook_chunks (grade_level, subject);


-- ---------------------------------------------------------------------------
-- 6. match_chunks RPC function
--
--    Called from Python as:
--        supabase.rpc("match_chunks", {...}).execute()
--
--    Returns rows ranked by cosine similarity (highest first).
--    distance is expressed as similarity score (1 - cosine_distance) so that
--    higher values indicate closer matches — consistent with the existing
--    generate.py threshold logic (distance <= 0.55 treated as relevant).
--
--    NOTE: the <=> operator computes cosine DISTANCE (0=identical, 2=opposite),
--    so we subtract from 1 to flip it into a similarity score (1=identical).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(1024),
    grade_filter    INTEGER,
    subject_filter  TEXT,
    match_count     INTEGER DEFAULT 8
)
RETURNS TABLE (
    id              UUID,
    text            TEXT,
    source_file     TEXT,
    curriculum      TEXT,
    grade_level     INTEGER,
    subject         TEXT,
    chapter         TEXT,
    page_number     INTEGER,
    section_heading TEXT,
    distance        FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        tc.id,
        tc.text,
        tc.source_file,
        tc.curriculum,
        tc.grade_level,
        tc.subject,
        tc.chapter,
        tc.page_number,
        tc.section_heading,
        -- Cosine SIMILARITY (1 = perfect match, 0 = orthogonal).
        -- Matches the convention in generate.py: chunks where distance > 0.55
        -- are considered low-relevance.
        (1 - (tc.embedding <=> query_embedding))::FLOAT AS distance
    FROM textbook_chunks tc
    WHERE tc.grade_level = grade_filter
      AND tc.subject      = subject_filter
    ORDER BY tc.embedding <=> query_embedding   -- ascending distance = best first
    LIMIT match_count;
END;
$$;


-- ---------------------------------------------------------------------------
-- 7. Row Level Security (RLS) — recommended for production Supabase projects.
--    The anon key used by supabase-py needs SELECT + INSERT access.
--    Adjust policies to match your actual auth strategy.
-- ---------------------------------------------------------------------------
ALTER TABLE textbook_chunks ENABLE ROW LEVEL SECURITY;

-- Allow anonymous reads (the Telegram bot reads but never needs to write at
-- query time, so keep the write policy restricted to the service_role key
-- used only during migration).
CREATE POLICY "allow_anon_select"
    ON textbook_chunks
    FOR SELECT
    TO anon
    USING (true);

-- Allow inserts only when the caller presents the service_role JWT.
-- The migration script must be initialised with the service_role key, not anon.
CREATE POLICY "allow_service_insert"
    ON textbook_chunks
    FOR INSERT
    TO service_role
    WITH CHECK (true);
