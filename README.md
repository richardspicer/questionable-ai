# Mutual Dissent

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

Cross-vendor multi-model debate and consensus engine for AI response distillation.

**Phase 1 complete** — working CLI with cross-vendor debate + reflection rounds. First live 4-vendor debate: 41k tokens across Claude, GPT, Gemini, and Grok.

Sends a user query to multiple AI models simultaneously, shares competing responses back to each model for reflection and critique, then synthesizes a final answer through a user-selected model.

## How It Works

1. **Fan out** — Query goes to Claude, GPT, Gemini, and Grok via OpenRouter
2. **Reflect** — Each model sees the others' responses and argues back
3. **Synthesize** — A user-selected model distills the debate into a final answer
4. **Log** — Full debate transcript saved as structured JSON

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
dissent ask "Your query here"
dissent ask "Your query here" --synthesizer claude
dissent ask "Your query here" --rounds 2
dissent ask "Your query here" --panel claude,gpt,gemini
```

`mutual-dissent` also works as the full command name.

## Status

**Phase 1: Foundation** — Complete. Phase 1.5 (provider abstraction for direct vendor APIs) in progress. See [Roadmap](docs/Roadmap.md) for the full plan.

## Research Platform

Full debate transcripts are logged as structured JSON, enabling analysis of:

- **Disagreement patterns** — where do models consistently diverge?
- **Convergence dynamics** — how many rounds until consensus? Which models cave first?
- **Consensus poisoning** — can a deliberately wrong claim propagate through reflection?
- **Unanimous hallucinations** — when all models confidently agree on the wrong answer

## Documentation

- [Architecture](docs/Architecture.md) — Components, data models, extension points
- [Roadmap](docs/Roadmap.md) — Phased development plan
- [Contributing](CONTRIBUTING.md) — Development setup and workflow

## License

MIT — see [LICENSE](LICENSE) for details.

## Author

[Richard Spicer](https://richardspicer.io) — Security research at [MLSecOps Lab](https://mlsecopslab.io)
