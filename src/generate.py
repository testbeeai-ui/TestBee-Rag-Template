"""
generate.py — Sarvam AI answer generation for the Testbee RAG pipeline.

FLOW:
    retrieved chunks from ChromaDB
         │
         ▼
    build_prompt()  → formats system context + user question into a prompt string
         │
         ▼
    call_sarvam_api()  → POST to https://api.sarvam.ai/v1/chat/completions
         │
         ▼
    returns clean answer string to telegram_bot.py
"""

import os
import httpx
from config import SARVAM_API_KEY

# ---------------------------------------------------------------------------
# Sarvam AI API constants
# ---------------------------------------------------------------------------
SARVAM_BASE_URL = "https://api.sarvam.ai/v1"
SARVAM_MODEL    = "sarvam-m"
REQUEST_TIMEOUT = 30.0   # seconds


_CURRICULUM_PREAMBLES = {
    "CBSE":         "Answer at CBSE board exam level — clear explanations, standard examples.",
    "JEE_Main":     "Answer at JEE Main level — formula-heavy, problem-solving focused.",
    "JEE_Advanced": "Answer at JEE Advanced level — deep derivations, multi-concept reasoning.",
}


def build_prompt(question: str, chunks: list[dict], curriculum: str = "CBSE") -> str:
    """
    Build the prompt string from the question and retrieved chunks.

    Each chunk dict has keys:
        'text'      — the raw passage text from the textbook
        'metadata'  — dict with keys: source_file, curriculum, grade_level,
                       subject, chapter
        'distance'  — cosine distance score from ChromaDB (lower = more relevant)

    Args:
        question:   The user's Physics question as a plain string.
        chunks:     List of chunk dicts returned by retrieve.query_collection().
        curriculum: The user's selected curriculum (CBSE, JEE_Main, JEE_Advanced).
                    Used to prepend a depth-calibration preamble line.

    Returns a single string that will be sent as the user message to the API.
    The system role is handled separately inside call_sarvam_api().
    """
    preamble = _CURRICULUM_PREAMBLES.get(curriculum, _CURRICULUM_PREAMBLES["CBSE"])

    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta     = chunk.get("metadata", {})
        chapter  = meta.get("chapter", "Unknown Chapter")
        subject  = meta.get("subject", "Unknown Subject")
        page_num = meta.get("page_number", None)
        text     = chunk.get("text", "").strip()

        header = f"[Passage {i} | {subject} — {chapter}"
        if page_num is not None:
            header += f", page {page_num}"
        header += "]"

        context_parts.append(f"{header}\n{text}")

    context_block = "\n\n".join(context_parts)

    prompt = (
        f"{preamble}\n\n"
        "Use the textbook passages provided below as your primary source. "
        "Even if the passages cover related or adjacent concepts, use them to build "
        "the best possible answer. Only say you don't know if there is truly no "
        "relevant information at all.\n\n"
        "--- TEXTBOOK PASSAGES ---\n"
        f"{context_block}\n"
        "--- END OF PASSAGES ---\n\n"
        f"Question: {question}"
    )
    return prompt


def call_sarvam_api(prompt: str, api_key: str) -> str:
    """
    Call the Sarvam AI chat completions endpoint with the given prompt.

    Uses the OpenAI-compatible format:
        POST https://api.sarvam.ai/v1/chat/completions
        Authorization: Bearer <api_key>
        Body: { model, messages: [{role, content}] }

    Returns the assistant's reply as a plain string.
    Raises RuntimeError if the HTTP request fails or the response is malformed.
    """
    url = f"{SARVAM_BASE_URL}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": SARVAM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Testbee, an expert Physics tutor. "
                    "Answer student questions clearly and accurately using "
                    "only the provided textbook context. "
                    "Be concise, structured, and pedagogically helpful."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    try:
        response = httpx.post(
            url,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Sarvam API returned HTTP {exc.response.status_code}: "
            f"{exc.response.text}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Network error while calling Sarvam API: {exc}"
        ) from exc

    data = response.json()

    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected Sarvam API response structure: {data}"
        ) from exc

    return answer.strip()


def generate_answer(question: str, chunks: list[dict], curriculum: str = "CBSE") -> str:
    """
    Top-level function called by telegram_bot.py.

    Reads SARVAM_API_KEY from the environment (populated by config.py via
    python-dotenv), builds the RAG prompt, calls the Sarvam AI API, and
    returns the clean answer string.

    Args:
        question:   The user's Physics question as a plain string.
        chunks:     List of chunk dicts returned by retrieve.query_collection().
                    Each has keys 'text', 'metadata', 'distance'.
        curriculum: The user's selected curriculum (CBSE, JEE_Main, JEE_Advanced).
                    Passed through to build_prompt() to calibrate answer depth.

    Returns:
        A plain string answer from the Sarvam AI model.

    Raises:
        RuntimeError: If SARVAM_API_KEY is missing or the API call fails.
    """
    api_key = SARVAM_API_KEY or os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY is not set. "
            "Add it to testbee_local/.env and restart."
        )

    if not chunks:
        return (
            "I could not find any relevant passages in the textbook for your question. "
            "Please try rephrasing, or check that the relevant material has been ingested."
        )

    # Filter out chunks that are too dissimilar (distance > 0.55 means low relevance).
    # Keep at least 3 chunks even if all are above threshold, so LLM always has context.
    filtered = [c for c in chunks if c.get("distance", 1.0) <= 0.55]
    if len(filtered) < 3:
        filtered = sorted(chunks, key=lambda c: c.get("distance", 1.0))[:3]

    prompt = build_prompt(question, filtered, curriculum)
    answer = call_sarvam_api(prompt, api_key)
    return answer
