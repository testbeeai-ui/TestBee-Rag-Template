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
import sys
from pathlib import Path

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
import retrieve
import generate
from latex_formatter import format_response

# ---------------------------------------------------------------------------
# BGE-M3 model — loaded ONCE at bot startup so every message reuses it.
# ---------------------------------------------------------------------------
_model = embed.load_model()

# Valid curriculum values for quick membership testing
_VALID_CURRICULA = {c.value for c in Curriculum}


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /start command.

    - Sends a welcome message explaining the bot's purpose.
    - Initialises the user's session context with sensible defaults
      (curriculum=CBSE, grade_level=11, subject=Physics).
    - Prompts the user to customise via /setcurriculum if needed.
    """
    context.user_data["curriculum"]  = Curriculum.CBSE.value
    context.user_data["grade_level"] = 11
    context.user_data["subject"]     = "Physics"

    welcome = (
        "Welcome to Testbee! \U0001f4da\n"
        "I'm your NCERT Physics Class\u00a011 assistant. "
        "Ask me any Physics question!\n\n"
        "Your session is set to:\n"
        "  Curriculum : CBSE\n"
        "  Grade      : 11\n"
        "  Subject    : Physics\n\n"
        "To switch curriculum, use:\n"
        "  /setcurriculum CBSE\n"
        "  /setcurriculum JEE_Main\n"
        "  /setcurriculum JEE_Advanced\n\n"
        "Just type your question and I'll find the answer from your textbook."
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
        "You can now ask your Physics questions!"
    )


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

    # Ensure session has been initialised; if not, nudge the user.
    if "curriculum" not in context.user_data:
        await update.message.reply_text(
            "Please run /start first to initialise your session."
        )
        return

    curriculum  = context.user_data.get("curriculum",  Curriculum.CBSE.value)
    grade_level = context.user_data.get("grade_level", 11)
    subject     = context.user_data.get("subject",     "Physics")

    # Let the user know we're working on it.
    thinking_msg = await update.message.reply_text("Thinking...")

    try:
        # Embed the question using the module-level BGE-M3 model.
        query_embedding = embed.embed_chunks([question], _model)[0]

        # Build ChromaDB metadata filters.
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

        # Generate an answer from the retrieved context.
        # Pass curriculum so build_prompt() prepends the correct depth preamble.
        answer = generate.generate_answer(question, chunks, curriculum)

    except Exception as exc:  # noqa: BLE001
        answer = (
            f"Sorry, something went wrong while processing your question.\n\n"
            f"Error: {exc}"
        )

    # Convert LaTeX to Unicode/images for Telegram.
    formatted_text, formula_images = format_response(answer)

    # Edit the "Thinking..." message in place with the formatted answer.
    await thinking_msg.edit_text(formatted_text)

    # Send each display-math formula as a separate photo.
    for i, img_bytes in enumerate(formula_images):
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=io.BytesIO(img_bytes),
            caption=f"Formula {i + 1}",
        )


def run_bot() -> None:
    """
    Entry point. Build the Telegram Application and register all handlers.

    Registers:
        /start            → start_handler
        /setcurriculum    → set_curriculum_handler
        <any text>        → question_handler
    """
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("setcurriculum", set_curriculum_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, question_handler))

    app.run_polling()


if __name__ == "__main__":
    run_bot()
