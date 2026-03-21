"""
telegram_bot.py — Telegram interface for the Testbee RAG pipeline.

FLOW when a user sends a question:
    User message on Telegram
         │
         ▼
    question_handler() receives the text
         │
         ▼
    embed the question → single BGE-M3 vector
         │
         ▼
    build filters from user's session context
    (curriculum, grade_level set via /start or /setcurriculum command)
         │
         ▼
    retrieve.query_collection() → top-5 relevant chunks
         │
         ▼
    generate.generate_answer() → calls Sarvam AI with chunks as context
         │
         ▼
    send formatted answer back to user
"""

import io
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Ensure the src/ directory is on the path so all local imports resolve.
sys.path.insert(0, str(Path(__file__).parent))

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from config import TELEGRAM_BOT_TOKEN, Curriculum
import embed
import supabase_retrieve as retrieve  # chunks live in Supabase (pgvector), not local Chroma
import generate
from latex_formatter import format_response

# ---------------------------------------------------------------------------
# BGE-M3 model — loaded ONCE at bot startup so every message reuses it.
# ---------------------------------------------------------------------------
_model = embed.load_model()

# Valid curriculum values for quick membership testing
_VALID_CURRICULA = {c.value for c in Curriculum}

# Must match folder names after capitalize() in run_ingest (physics → Physics, maths → Maths)
_VALID_SUBJECTS = {"Physics", "Chemistry", "Maths"}
_TELEGRAM_MAX_TEXT = 4000
_MAX_USER_QUESTION_CHARS = 1200


def _split_for_telegram(text: str, max_len: int = _TELEGRAM_MAX_TEXT) -> list[str]:
    """
    Split long text into Telegram-safe chunks.
    Prefer splitting on paragraph boundaries, then line boundaries.
    """
    text = (text or "").strip()
    if not text:
        return [""]
    if len(text) <= max_len:
        return [text]

    parts: list[str] = []
    remaining = text
    while len(remaining) > max_len:
        cut = remaining.rfind("\n\n", 0, max_len)
        if cut == -1:
            cut = remaining.rfind("\n", 0, max_len)
        if cut == -1:
            cut = max_len
        chunk = remaining[:cut].strip()
        if not chunk:
            chunk = remaining[:max_len]
            cut = max_len
        parts.append(chunk)
        remaining = remaining[cut:].strip()
    if remaining:
        parts.append(remaining)
    return parts


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /start command.

    - Sends a welcome message explaining the bot's purpose.
    - Initialises the user's session context with sensible defaults
      (curriculum=CBSE, grade_level=11, subject=Physics).
    - Prompts the user to customise via /setcurriculum if needed.
    """
    context.user_data["curriculum"] = Curriculum.CBSE.value
    # Default — change with /setgrade and /setsubject to match ingested PDFs (metadata filters).
    context.user_data["grade_level"] = 12
    context.user_data["subject"] = "Chemistry"

    welcome = (
        "Welcome to Testbee! \U0001f4da\n"
        "I search your ingested NCERT chunks in Supabase by **grade + subject**.\n\n"
        "Your session is set to:\n"
        "  Curriculum : CBSE\n"
        "  Grade      : 12\n"
        "  Subject    : Chemistry\n\n"
        "Before asking, match the book you indexed:\n"
        "  /setgrade 11  or  /setgrade 12\n"
        "  /setsubject Physics  |  Chemistry  |  Maths\n"
        "  /setcurriculum CBSE | JEE_Main | JEE_Advanced\n\n"
        "Then send your question as plain text."
    )
    await update.message.reply_text(welcome)


async def set_curriculum_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /setcurriculum command.

    Parses the curriculum argument from the command, validates it against
    the Curriculum enum in config.py, and stores it in context.user_data.

    Example:
        /setcurriculum CBSE       → stores {"curriculum": "CBSE"}
        /setcurriculum JEE_Main   → stores {"curriculum": "JEE_Main"}
    """
    args = context.args  # list of words after the command

    if not args:
        await update.message.reply_text(
            "Please provide a curriculum name.\n\n"
            "Usage: /setcurriculum <curriculum>\n"
            "Valid options: CBSE, JEE_Main, JEE_Advanced"
        )
        return

    requested = args[0].strip()

    if requested not in _VALID_CURRICULA:
        await update.message.reply_text(
            f"'{requested}' is not a valid curriculum.\n\n"
            "Valid options are:\n"
            "  CBSE\n"
            "  JEE_Main\n"
            "  JEE_Advanced\n\n"
            "Example: /setcurriculum JEE_Main"
        )
        return

    context.user_data["curriculum"] = requested

    await update.message.reply_text(
        f"Curriculum set to: {requested}\n\n"
        "You can now ask questions (grade/subject must still match ingested PDFs)."
    )


async def set_grade_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """e.g. /setgrade 12 — must match paths like raw_pdfs/cbse/class12/..."""
    args = context.args
    if not args or not args[0].isdigit():
        await update.message.reply_text("Usage: /setgrade 11  or  /setgrade 12")
        return
    g = int(args[0])
    if g not in (11, 12):
        await update.message.reply_text("Only grades 11 and 12 are supported.")
        return
    context.user_data["grade_level"] = g
    await update.message.reply_text(f"Grade set to {g}.")


async def set_subject_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """e.g. /setsubject Chemistry — must match folder name (Physics, Chemistry, Maths)."""
    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage: /setsubject Physics\n"
            "       /setsubject Chemistry\n"
            "       /setsubject Maths"
        )
        return
    raw = " ".join(args).strip()
    # Title-case single words; keep "Maths" as stored in DB
    subj = raw[0].upper() + raw[1:].lower() if len(raw) > 1 else raw.upper()
    if subj == "Math":
        subj = "Maths"
    if subj not in _VALID_SUBJECTS:
        await update.message.reply_text(
            f"Unknown subject '{raw}'. Use: Physics, Chemistry, or Maths."
        )
        return
    context.user_data["subject"] = subj
    await update.message.reply_text(f"Subject set to {subj}.")


async def question_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle any plain text message as a Physics question.

    Steps:
        1. Extract question text from update.message.text.
        2. Read curriculum, grade_level, and subject from context.user_data.
           If missing, prompt user to run /start first.
        3. Send a "Thinking..." holding message so the user knows work is in progress.
        4. Load BGE-M3 model (module-level singleton) → embed question vector.
        5. Build filters dict for ChromaDB metadata pre-filtering.
        6. Call retrieve.query_collection() → top-5 relevant chunks.
        7. Call generate.generate_answer() → Sarvam AI answer.
        8. Send the answer back via update.message.reply_text().
    """
    question = update.message.text.strip()
    if not question:
        return

    if len(question) > _MAX_USER_QUESTION_CHARS:
        await update.message.reply_text(
            f"Your question is too long ({len(question)} chars). "
            f"Please keep it under {_MAX_USER_QUESTION_CHARS} characters."
        )
        return

    # Ensure session has been initialised; if not, nudge the user.
    if "curriculum" not in context.user_data:
        await update.message.reply_text(
            "Please run /start first to initialise your session."
        )
        return

    curriculum = context.user_data.get("curriculum", Curriculum.CBSE.value)
    grade_level = context.user_data.get("grade_level", 12)
    subject = context.user_data.get("subject", "Chemistry")
    chat_id = update.effective_chat.id if update.effective_chat else -1
    user_id = update.effective_user.id if update.effective_user else -1
    request_id = f"tg-{update.update_id}"
    start_ts = time.perf_counter()

    # Let the user know we're working on it.
    thinking_msg = await update.message.reply_text("Thinking...")
    chunks_count = 0

    try:
        # Embed the question using the module-level BGE-M3 model.
        query_embedding = embed.embed_chunks([question], _model)[0]

        # Build metadata filters for vector search (Chroma or Supabase RPC).
        # NOTE: curriculum is intentionally excluded here — all chunks in the
        # database are stored with curriculum=CBSE regardless of the user's
        # selected curriculum.  Filtering on curriculum would return zero results
        # for JEE_Main / JEE_Advanced users.  Instead, curriculum is passed
        # directly to generate_answer() so the LLM prompt is depth-calibrated.
        metadata_filters = {
            "grade_level": grade_level,
            "subject":     subject,
        }

        # Retrieve the top-5 most relevant chunks.
        collection = retrieve.get_or_create_collection()
        chunks     = retrieve.query_collection(
            collection=collection,
            query_embedding=query_embedding,
            filters=metadata_filters,
            n_results=8,
        )
        chunks_count = len(chunks)

        # Generate an answer from the retrieved context.
        # Pass curriculum so build_prompt() prepends the correct depth preamble.
        # Supabase match_chunks returns cosine similarity as `distance` (higher = better).
        answer = generate.generate_answer(
            question,
            chunks,
            curriculum,
            retrieval_score_higher_is_better=True,
        )

    except Exception as exc:  # noqa: BLE001
        # Do not expose exception text to Telegram users (info disclosure).
        elapsed_ms = int((time.perf_counter() - start_ts) * 1000)
        logger.exception(
            "question_handler failed request_id=%s chat_id=%s user_id=%s subject=%s grade=%s elapsed_ms=%s",
            request_id,
            chat_id,
            user_id,
            subject,
            grade_level,
            elapsed_ms,
        )
        answer = (
            "Sorry, something went wrong while processing your question. "
            "Please try again in a moment. If it keeps happening, check that "
            "the bot service is running and your session is set (/start)."
        )

    # Convert LaTeX to Unicode/images for Telegram.
    formatted_text, formula_images = format_response(answer)

    # Edit the "Thinking..." message in place for short answers.
    # Long answers must be split to avoid Telegram BadRequest: Message_too_long.
    text_parts = _split_for_telegram(formatted_text)
    if len(text_parts) == 1:
        await thinking_msg.edit_text(text_parts[0] or "(empty response)")
    else:
        await thinking_msg.edit_text("Answer is long; sending in parts...")
        for idx, part in enumerate(text_parts, start=1):
            await update.message.reply_text(f"[Part {idx}/{len(text_parts)}]\n{part}")

    # Send each display-math formula as a separate photo.
    for i, img_bytes in enumerate(formula_images):
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=io.BytesIO(img_bytes),
            caption=f"Formula {i + 1}",
        )

    elapsed_ms = int((time.perf_counter() - start_ts) * 1000)
    logger.info(
        "question_handler ok request_id=%s chat_id=%s user_id=%s subject=%s grade=%s chunks=%s reply_parts=%s formulas=%s elapsed_ms=%s",
        request_id,
        chat_id,
        user_id,
        subject,
        grade_level,
        chunks_count,
        len(text_parts),
        len(formula_images),
        elapsed_ms,
    )


async def _post_init_delete_webhook(application) -> None:
    """Avoid Conflict: another getUpdates — clear webhook so polling is the only mode."""
    await application.bot.delete_webhook(drop_pending_updates=True)


def run_bot() -> None:
    """
    Entry point. Build the Telegram Application and register all handlers.

    Registers:
        /start            → start_handler
        /setcurriculum    → set_curriculum_handler
        <any text>        → question_handler
    """
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(_post_init_delete_webhook)
        .build()
    )

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("setcurriculum", set_curriculum_handler))
    app.add_handler(CommandHandler("setgrade", set_grade_handler))
    app.add_handler(CommandHandler("setsubject", set_subject_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, question_handler))

    # drop_pending_updates: recover cleanly after crashes / duplicate runs
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    run_bot()
