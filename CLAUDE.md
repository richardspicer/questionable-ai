# Questionable AI (qAI)

Cross-vendor multi-model debate and consensus engine. Sends queries to multiple AI models simultaneously, runs reflection rounds where each model critiques the others, then synthesizes a final answer.

Part of richardspicer.io research portfolio. Research angle: cross-vendor disagreement patterns, convergence dynamics, consensus poisoning.

## Project Layout

```
src/questionable_ai/
├── cli.py              # Click CLI (ask command with --panel, --synthesizer, --rounds)
├── orchestrator.py     # Core pipeline: fan-out → reflect → synthesize
├── client.py           # OpenRouter API client (async httpx)
├── config.py           # Configuration loading (API key, model mappings)
├── models.py           # Data models (DebateTranscript, DebateRound, ModelResponse)
├── prompts.py          # Prompt templates for initial/reflection/synthesis rounds
├── transcript.py       # JSON transcript save/load
├── display.py          # Rich console output formatting
└── __init__.py
tests/                  # pytest suite
docs/
├── Architecture.md     # Components, data models, extension points
└── Roadmap.md          # Phased development plan
```

## How It Works

1. **Fan out** — Query goes to panel models (Claude, GPT, Gemini, Grok) via OpenRouter
2. **Reflect** — Each model sees others' responses and argues back (configurable rounds)
3. **Synthesize** — User-selected model distills the debate into a final answer
4. **Log** — Full debate transcript saved as structured JSON

All API calls within a round are parallel (async).

## Architecture

- `orchestrator.py` is the core — `run_debate()` manages the full lifecycle
- `client.py` wraps OpenRouter API with async httpx
- `prompts.py` formats system/user messages for each debate phase
- `models.py` has Pydantic-free dataclasses (DebateTranscript, DebateRound, ModelResponse)
- `transcript.py` handles JSON serialization of full debate history
- `display.py` renders Rich panels for each model's response

## Code Standards

- **Docstrings:** Google-style on all public functions and classes (Args, Returns, Raises)
- **Type hints:** Required on all function signatures
- **Line length:** 100 chars (ruff)
- **Python version:** 3.14+ (target-version in ruff and mypy)
- **Imports:** Sorted by ruff (isort rules)

## Testing

- Framework: pytest
- Tests in `tests/`
- Run: `uv run pytest -q`

## Git Workflow

**Never commit directly to main.** Pre-commit hook blocks it.

```
git checkout main && git pull
git checkout -b feature/description    # or fix/, docs/, refactor/
# ... work ...
git add .
git commit -F .commitmsg               # see shell quoting note below
git push -u origin feature/description
# Create PR on GitHub, merge
```

### Shell Quoting (CRITICAL)

CMD shell corrupts `git commit -m "message with spaces"`. Always use:
```
echo "feat: description here" > .commitmsg
git commit -F .commitmsg
rm .commitmsg
```

## Pre-commit Hooks

Hooks run automatically on `git commit`:
- trailing-whitespace, end-of-file-fixer, check-yaml, check-toml
- check-added-large-files, check-merge-conflict
- **no-commit-to-branch** (blocks direct commits to main)
- **ruff-check** (lint + auto-fix) + **ruff-format**
- **gitleaks** (secrets detection)
- **mypy** (type checking)

## Dependencies

Managed via `uv` with `pyproject.toml`:
```
uv sync               # production deps only
uv sync --group dev   # includes ruff, mypy, pre-commit, pytest
```

## Environment

Requires `OPENROUTER_API_KEY` environment variable. Set in PowerShell profile or `.env`.

## Key Patterns

- Debate transcripts are structured JSON with full response text per model per round
- OpenRouter provides unified API across Claude, GPT, Gemini, Grok — no vendor-specific clients
- Reflection prompts include all other models' responses for that round
- Synthesis prompt includes formatted transcript of entire debate history
- CLI aliases: `qai` and `questionable-ai` both work

## CLI Usage

```powershell
questionable-ai ask "Your query here"
questionable-ai ask "Your query here" --synthesizer claude
questionable-ai ask "Your query here" --rounds 2
questionable-ai ask "Your query here" --panel claude,gpt,gemini
qai ask "Your query here"    # short alias
```

After changes, smoke test: `questionable-ai --help`

## Claude Code Guardrails

### Verification Scope
- Run only the tests for new/changed code, not the full suite
- Smoke test the CLI after changes
- Full suite verification is the developer's responsibility before merging

### Timeout Policy
- If any test run exceeds 60 seconds, stop and identify the stuck test
- Do not set longer timeouts and wait — diagnose instead

### Process Hygiene
- Before running tests, kill any orphaned python/node processes from previous runs
- After killing a stuck process, clean up zombies before retrying

### Failure Mode
- If verification hits a problem you can't resolve in 2 attempts, commit the work to the branch and report what failed
- Do not spin on the same failure

### Boundaries
- Do not create PRs or install tools. Push the branch and stop. The developer creates PRs manually.
- Do not attempt to install CLI tools (gh, hub, etc.)

## Session Discipline

- **Architecture.md:** Update at end of session if new modules, endpoints, or data models were introduced
- **Running checks manually:**
```powershell
ruff check .
ruff format --check .
mypy src/questionable_ai/
pre-commit run --all-files
```
