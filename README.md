# Mutual Dissent

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Docs](https://img.shields.io/badge/docs-mutual--dissent.dev-8b5cf6)](https://mutual-dissent.dev)

Cross-vendor multi-model debate and consensus engine for AI response distillation.

**Phase 2 complete** — full CLI research tool with direct vendor APIs, replay, ground-truth scoring, cost tracking, and markdown export. 325+ tests across Windows and Linux CI.

Sends a user query to multiple AI models simultaneously, shares competing responses back to each model for reflection and critique, then synthesizes a final answer through a user-selected model.

## How It Works

1. **Fan out** — Query goes to Claude, GPT, Gemini, and Grok (direct APIs or via OpenRouter)
2. **Reflect** — Each model sees the others' responses and argues back
3. **Synthesize** — A user-selected model distills the debate into a final answer
4. **Log** — Full debate transcript saved as structured JSON with cost and token data

## Why Cross-Vendor?

Single-vendor multi-agent systems (Grok's 4-agent debate, Anthropic's agent teams) share the same training data and blind spots. Cross-vendor debate surfaces disagreements that correlated architectures can't — different training data, different safety postures, different failure modes.

## Installation

```bash
git clone https://github.com/q-uestionable-AI/mutual-dissent.git
cd mutual-dissent
uv sync
```

## Usage

```bash
# Run a debate
dissent ask "Your query here"
dissent ask "Your query here" --synthesizer claude --rounds 2 --panel claude,gpt,gemini

# Attach file context
dissent ask "Summarize this" --file report.pdf

# Score against a known answer
dissent ask "What is the speed of light?" --ground-truth "299,792,458 m/s"

# Manage transcripts
dissent list
dissent show <transcript-id>

# Replay with a different synthesizer
dissent replay <transcript-id> --synthesizer grok

# Export to markdown
dissent show <transcript-id> --file output.md

# View config and provider status
dissent config show
dissent config test
```

`mutual-dissent` also works as the full command name.

## Status

| Phase | Status |
|-------|--------|
| **Phase 1: Foundation** | ✅ Complete — core debate loop, OpenRouter integration |
| **Phase 1.5: Provider Abstraction** | ✅ Complete — direct Anthropic API, mixed-panel routing |
| **Phase 2: CLI Expansion** | ✅ Complete — replay, scoring, cost tracking, markdown export |
| **Phase 3: Web GUI** | 🔜 Next — NiceGUI debate view + research dashboard |
| **Phase 4: Documentation** | Planned — Mintlify docs site with AI assistant |
| **Phase 5: Maturity** | Planned — Tauri desktop app, batch mode, public release |

See [Roadmap](docs/Roadmap.md) for the full plan.

## Research Platform

Full debate transcripts are logged as structured JSON, enabling analysis of:

- **Disagreement patterns** — where do models consistently diverge?
- **Convergence dynamics** — how many rounds until consensus? Which models cave first?
- **Consensus poisoning** — can a deliberately wrong claim propagate through reflection?
- **Unanimous hallucinations** — when all models confidently agree on the wrong answer

## Documentation

- [Architecture](docs/Architecture.md) — Components, data models, extension points
- [Roadmap](docs/Roadmap.md) — Phased development plan
- [Environment](docs/Environment.md) — API keys and environment variables
- [Contributing](CONTRIBUTING.md) — Development setup and workflow

## License

MIT — see [LICENSE](LICENSE) for details.

## Author

[Richard Spicer](https://richardspicer.io) — Security research at [MLSecOps Lab](https://mlsecopslab.io)
