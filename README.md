# TestBee RAG Template

Ingest NCERT-style PDFs, chunk with Docling, embed with BGE-M3, store in **Supabase** (or local ChromaDB).

> **Note:** Ingestion is a **single Python pipeline** (`run_ingest.py`), not a multi-agent runner. Cursor/Claude “agents” are optional helpers in the IDE; they do not need to run for PDF ingestion.

## Requirements

- Python 3.10+ and project dependencies (`pip install -r requirements.txt`)
- GPU recommended (Docling + embeddings)
- For Supabase: project URL + **service role** key in `.env`

## Environment (`.env`)

| Variable | Purpose |
|----------|---------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Service role key (ingest writes) |
| `SUPABASE_KEY` | Supabase anon/public key (bot read queries) |
| `SARVAM_API_KEY` | Sarvam chat completions API key |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `INGEST_SUBJECT` | Optional. If set (e.g. `maths`), only that folder is ingested **unless** you pass `--all`. |

Omit `SUPABASE_*` to upload to local ChromaDB instead.

## PDF layout (required for metadata)

Paths must be **four segments** under `data/raw_pdfs/`:

```text
data/raw_pdfs/<curriculum>/<classXX>/<subject>/<file>.pdf
```

Example: `data/raw_pdfs/cbse/class11/physics/keph101.pdf`

## Run ingestion — **all PDFs, no subject restriction**

```bash
# Recommended if you ever set INGEST_SUBJECT in .env — forces full run
python run_ingest.py --all

# Default: all PDFs when INGEST_SUBJECT is unset
python run_ingest.py
```

## Optional: only one subject folder

```bash
python run_ingest.py --subject maths
```

## Database setup

Run `supabase_setup.sql` once in the Supabase SQL editor (enables `pgvector`, `textbook_chunks`, `match_chunks`).

## Telegram bot notes

- Start bot: `python src/telegram_bot.py`
- Session filters must match ingested metadata:
  - `/setgrade 11` or `/setgrade 12`
  - `/setsubject Physics | Chemistry | Maths`
- User question limit: **1200 characters** (longer messages are rejected early).

## Retrieval score semantics (important)

The `distance` field means different things by backend:

- **Chroma (`retrieve.py`)**: cosine **distance** (lower is better), cutoff `<= 0.55`
- **Supabase (`supabase_retrieve.py`)**: cosine **similarity** (higher is better), cutoff `>= 0.45`

`src/telegram_bot.py` already calls:

```python
generate.generate_answer(..., retrieval_score_higher_is_better=True)
```

when using `supabase_retrieve`, which applies the correct Supabase cutoff.

## Observability

`telegram_bot.py` logs one line per request with:

- `request_id`, `chat_id`, `user_id`
- `subject`, `grade`
- retrieved `chunks`, reply parts, rendered formulas
- total `elapsed_ms`

On errors, users get a generic failure message while full details are logged server-side.

## Security

- Never commit `.env` or API keys.
- Rotate keys immediately if they were exposed (e.g. in screenshots or chat).
