# Agent definitions (Claude / Cursor)

**Preference:** Use specialist agents **when the task requires** that expertise (security, architecture, RAG, etc.); keep answers normal for trivial edits. See **`.cursor/rules/use-agents-when-required.mdc`** (`alwaysApply: true`).

Use specialist **agent markdown** files (YAML frontmatter + playbook) with this project. Resolution order for Cursor is in **`.cursor/rules/claude-agents-bridge.mdc`**.

## Install / refresh Cursor rules (`cursor-install.sh`)

The script copies `agency-agents/.cursor/rules/*.mdc` into **whatever directory is your shell’s current working directory** — so you **must `cd` to this project root first**, then run the script.

### Git Bash (Windows) — typical paths

```bash
cd "/c/Users/rentk/Downloads/TestBee-Rag-Template"
bash "/c/Users/rentk/Desktop/agency-agents/scripts/cursor-install.sh"
```

Then **restart Cursor** (or reload the window) so new rules load.

If your project or `agency-agents` live elsewhere, change those two paths.

### PowerShell (same result, no bash)

```powershell
$dst = "c:\Users\rentk\Downloads\TestBee-Rag-Template\.cursor\rules"
$src = "c:\Users\rentk\Desktop\agency-agents\.cursor\rules"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item "$src\*.mdc" $dst -Force
```

---

## Option A — `agency-agents` inside this repo (recommended)

Keeps agents versioned with the app or as a **submodule**.

### A1. Git submodule

From this repo root:

```bash
git submodule add <YOUR_AGENCY_AGENTS_REPO_URL> agency-agents
git submodule update --init --recursive
```

Commit the submodule pointer. Clone for others:

```bash
git clone --recurse-submodules <THIS_REPO_URL>
# or after clone:
git submodule update --init --recursive
```

Put agent `.md` files in `agency-agents/` **or** `agency-agents/agents/` (both are supported by the bridge rule).

### A2. Plain folder clone (not a submodule)

```bash
cd c:\Users\rentk\Downloads\TestBee-Rag-Template
git clone <YOUR_AGENCY_AGENTS_REPO_URL> agency-agents
```

Add `agency-agents/` to **`.gitignore`** if you do **not** want it committed:

```gitignore
agency-agents/
```

### A3. Symlink / junction (Windows)

If the repo lives elsewhere:

```powershell
# Run in project root; adjust source path
New-Item -ItemType Junction -Path "agency-agents" -Target "C:\path\to\agency-agents"
```

---

## Option B — Path file (agents stay anywhere on disk)

1. Copy **`agency-agents.path.example`** → **`agency-agents.path`** (same directory).
2. Edit **`agency-agents.path`**: one line, absolute path to the folder that contains `*.md` agents (no quotes).
3. `agency-agents.path` is **gitignored** — safe for your machine only.

Cursor will read that file when you ask to use agents (and use the path inside).

---

## Option C — Claude Code default folder

If you don’t use A or B, agents can still live at:

| OS | Path |
|----|------|
| Windows | `%USERPROFILE%\.claude\agents\` |
| macOS / Linux | `~/.claude/agents/` |

You may need to **allow Cursor** to read files outside the workspace when prompted.

---

## Using agents in Cursor

1. Say e.g. *“Use the Security Engineer agent”* or *“follow engineering-code-reviewer.md”*.
2. Ensure agents resolve via **A, B, or C** above.
3. For **Option A**, no extra “outside workspace” permission is needed if `agency-agents/` is inside the project.

---

## Your `agency-agents` repo layout

Common patterns (both work):

```text
agency-agents/
  engineering-security-engineer.md
  ...
```

or:

```text
agency-agents/
  agents/
    engineering-security-engineer.md
```

If your repo uses a different subfolder, either set **`agency-agents.path`** to that full path (**Option B**) or add a junction/symlink.

---

## Ingestion (unrelated but same repo)

```bash
python run_ingest.py --all
```

See **`README.md`** for PDF paths and `.env`.
