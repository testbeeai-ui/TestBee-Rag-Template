"""
config.py — Central configuration for Testbee RAG pipeline.

All paths are derived dynamically from this file's location using pathlib.
No hardcoded absolute paths (no C:\\Users\\...). This file lives at:
    testbee_local/src/config.py

So:
    __file__          → testbee_local/src/config.py
    .parent           → testbee_local/src/
    .parent.parent    → testbee_local/          ← PROJECT_ROOT
"""

from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

# ---------------------------------------------------------------------------
# Root anchor — every other path derives from this one line.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
RAW_PDFS_DIR    = PROJECT_ROOT / "data" / "raw_pdfs"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
MODEL_DIR        = PROJECT_ROOT / "data" / "models"   # BGE-M3 lives here

# ---------------------------------------------------------------------------
# BGE-M3 model path — loaded from local disk, never downloaded at runtime.
# Run `python -c "from sentence_transformers import SentenceTransformer;
#   SentenceTransformer('BAAI/bge-m3').save('data/models/bge-m3')"` once
# from inside the testbee_local/ directory to populate this path.
# ---------------------------------------------------------------------------
BGE_M3_MODEL_PATH = MODEL_DIR / "bge-m3"

# ---------------------------------------------------------------------------
# ChromaDB collection name
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_NAME = "testbee_physics"

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / ".env")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
SARVAM_API_KEY     = os.getenv("SARVAM_API_KEY")

# ---------------------------------------------------------------------------
# Curriculum enum — enforces valid values at schema validation time.
# ---------------------------------------------------------------------------
class Curriculum(str, Enum):
    CBSE         = "CBSE"
    JEE_MAIN     = "JEE_Main"
    JEE_ADVANCED = "JEE_Advanced"

# ---------------------------------------------------------------------------
# DocumentMetadata — the strict Pydantic model that EVERY chunk must carry.
#
# This is the backbone of the pipeline. Every chunk stored in ChromaDB must
# have all five of these fields attached. This enables precise metadata
# pre-filtering at query time — e.g. "only search CBSE Grade 11 Physics".
# ---------------------------------------------------------------------------
class DocumentMetadata(BaseModel):
    source_file : str        = Field(..., description="Original PDF filename, e.g. 'keph101.pdf'")
    curriculum  : str        = Field(..., description="Curriculum tag: CBSE | JEE_Main | JEE_Advanced")
    grade_level : int        = Field(..., description="Grade level: 11 or 12")
    subject     : str        = Field(..., description="Subject name, e.g. 'Physics'")
    chapter     : str        = Field(..., description="Chapter name, e.g. 'Units and Measurement'")
