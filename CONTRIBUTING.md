# Contributing to Mutual Dissent

Mutual Dissent is an early-stage research tool. Contributions are welcome — whether that's new model integrations, debate topologies, bug fixes, or documentation improvements.

## Quick Start

```bash
git clone https://github.com/richardspicer/mutual-dissent.git
cd mutual-dissent
uv sync
pre-commit install
export OPENROUTER_API_KEY="sk-or-..."   # or set in ~/.mutual-dissent/config.toml
uv run mutual-dissent ask "test query" --verbose
```

## How to Contribute

1. **Open an issue first** to discuss your idea before writing code
2. Fork the repo and create a feature branch (`feature/`, `fix/`, `docs/`)
3. Follow the code standards below
4. Submit a PR referencing the issue

## Development Setup

```powershell
# Clone and setup
git clone https://github.com/richardspicer/mutual-dissent.git
cd mutual-dissent

# Install dependencies (creates .venv automatically)
uv sync

# Activate venv
.venv\Scripts\Activate.ps1  # Windows
# or: source .venv/bin/activate  # Linux/Mac

# Install pre-commit hooks
pre-commit install

# Verify
mutual-dissent --help
```

## Dev Tooling

Pre-commit hooks run automatically on every `git commit`. They enforce:

- **File hygiene** — trailing whitespace, EOF newlines, YAML/TOML syntax, large file detection, merge conflict markers
- **No commits to main** — all work must happen on feature branches
- **Ruff** — linting (pycodestyle, pyflakes, isort, bugbear, complexity, pyupgrade, bandit) and formatting
- **Gitleaks** — secrets detection (API keys, tokens, passwords)
- **Mypy** — type checking with `check_untyped_defs` enabled

If a hook fails, the commit is blocked. Auto-fixable issues (whitespace, import order, formatting) are corrected automatically — just re-stage and commit again.

### Running checks manually

```powershell
ruff check .                    # Lint
ruff format --check .           # Format check (dry run)
ruff format .                   # Format (apply)
mypy src/mutual_dissent/        # Type check
pre-commit run --all-files      # Run all hooks against entire repo
```

## Code Standards

- **Docstrings:** Google-style on all public functions, classes, and modules (Args, Returns, Raises, Example)
- **Type annotations:** All new public functions should have type hints. Mypy checks function bodies even without annotations, but annotated code is preferred.
- **Formatting:** Handled by ruff. Line length is 100 characters.
- **Commits:** One logical change per commit, descriptive messages
- **Testing:** Verify the CLI still works after changes: `mutual-dissent --help`

## Git Workflow

**Never work directly on main.** The pre-commit hook blocks this.

```powershell
# Start work
git checkout main
git pull
git checkout -b feature/your-description   # or fix/, docs/, refactor/

# Work and commit (hooks run automatically)
git add .
git commit -m "feat: description of change"

# Push and open PR
git push -u origin feature/your-description
```

### Branch naming

| Prefix | Use |
|--------|-----|
| `feature/` | New functionality |
| `fix/` | Bug fixes |
| `docs/` | Documentation only |
| `refactor/` | Code restructuring |

## Questions?

Open an issue or start a discussion. Happy to help you get oriented.
