# Mutual Dissent — Roadmap

## Problem Statement

AI models have different training data, architectures, and reasoning biases. Relying on a single model means inheriting its blind spots. Power users already work around this by manually querying multiple models, comparing responses, and cross-pollinating insights between conversations. This works but is slow and tedious.

Every existing multi-model tool either operates within a single vendor's ecosystem (Grok 4.20's 4-agent system, Anthropic's agent teams) or does cross-vendor comparison without a reflection loop (LLM Council, PolyCouncil). Nobody builds the full cycle: fan-out → reflection → refinement → synthesis across different vendors.

Mutual Dissent automates the workflow that power users already do manually — and logs the full debate as structured data for analysis.

---

## Phased Delivery

### Phase 1: Foundation ✅ COMPLETE

**Goal:** Working CLI that executes the core debate loop — fan-out, reflection, synthesis — and saves transcripts.

**Deliverables:**
- OpenRouter integration with async parallel model calls
- Core debate orchestrator: initial round → reflection round → synthesis
- Model alias system (claude, gpt, gemini, grok → OpenRouter model IDs)
- CLI with `ask` command and core flags (--panel, --synthesizer, --rounds)
- JSON transcript logging to `~/.mutual-dissent/transcripts/`
- Terminal output formatting (Rich panels, color-coded models)
- Configurable reflection and synthesis prompt templates
- Config file for API key and default settings

**Completed:** 2026-02-21. First live 4-vendor debate: 41,476 tokens, full reflection loop.

---

### Phase 1.5: Provider Abstraction (Direct API)

**Goal:** Replace the single-provider OpenRouter client with a provider
abstraction layer that supports direct vendor API keys alongside OpenRouter,
starting with Anthropic.

**Why now:** This is an architectural change to the client layer. Every feature
built after this (replay, cost tracking, GUI) depends on the provider interface.
Refactoring later means rewriting integrations.

**Build order:** Config schema first (drives router interface) → core types →
Provider ABC → OpenRouterProvider refactor → AnthropicProvider → ProviderRouter →
schema upgrades → routing smoke test.

**Deliverables:**

*Core types:*
- `Vendor` enum (`ANTHROPIC`, `OPENAI`, `GOOGLE`, `XAI`, `GROQ`, `OPENROUTER`, `OLLAMA`)
- `RoutedRequest` dataclass (`vendor`, `model_id`, `model_alias`, `round_number`, `messages`)
- `RoutingDecision` dataclass (`vendor`, `mode`, `via_openrouter`)

*Provider abstraction:*
- `Provider` ABC with `complete()` accepting `messages` or `prompt` (exactly one), plus `complete_parallel()`
- `OpenRouterProvider` — refactored from existing `client.py`
- `AnthropicProvider` — direct Anthropic Messages API client
- `ProviderRouter` — resolves alias → routing decision → dispatches to correct provider. Groups `RoutedRequest` by provider for `complete_parallel`.

*Config:*
- Config schema with `[providers]`, `[routing]`, `[model_aliases]` sections
- Per-alias dual model IDs: `claude.openrouter` and `claude.direct`
- Env var support: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.
- Backward compatibility: existing `api_key` field still works for OpenRouter

*Schema upgrades (concurrent with provider work):*
- `ModelResponse.role` — `"initial"` | `"reflection"` | `"synthesis"`
- `ModelResponse.routing` — serialized `RoutingDecision`
- `ModelResponse.analysis` — empty dict, reserved for future scoring
- `DebateTranscript.metadata.resolved_config` — full effective config (not hash)
- `DebateTranscript.metadata.stats` — `total_tokens`, `per_model`, `total_cost_usd`, placeholder convergence fields
- All timestamps UTC as `YYYY-MM-DDTHH:MM:SSZ`

*Routing smoke test:*
- `dissent config test` — for each alias: resolve routing → send minimal prompt → report provider, latency, errors

*Minimal ProviderCapabilities:*
- `max_context_tokens` only, pulled dynamically from OpenRouter `/api/v1/models`
- No hardcoded pricing — model prices change weekly

**Future provider stubs (not in this phase):**
- OpenAI, Google, xAI, Groq — same `Provider` interface, implemented when needed

**Exit criteria:**
- `dissent config test` passes for claude (direct) and gpt (OpenRouter)
- Mixed-panel debate works end-to-end (at least one model direct, one via OpenRouter)
- `provider` and `routing` info present in all transcript `ModelResponse` entries
- `metadata.stats` precomputed and present in transcripts
- `metadata.resolved_config` present in transcripts

**Design decisions (multi-model review consensus):**

| Decision | Rationale |
|----------|-----------|
| `resolved_config` dict over `config_hash` | Hash of TOML sections is fragile (ordering, whitespace). Full dict is more bytes, zero ambiguity. |
| Dynamic pricing from OpenRouter API | Model prices change weekly. Hardcoded pricing = constant maintenance. |
| Config schema before router implementation | Config shape drives the router interface, not the other way around. |
| Schema upgrades concurrent with provider work | Avoids migration later. Richer schema from the first multi-provider transcript. |
| Topology/roles/RAG deferred to Phase 3+ | They're the research payload — deferred because plumbing isn't ready, not because they're optional. |

---

### Phase 2: CLI Expansion ✅ COMPLETE

**Goal:** Replay capability, additional output formats, file input, ground truth
scoring, and cost tracking. The CLI becomes a complete research tool.

**Completed:** 2026-02-28. All deliverables merged: replay, list/show, markdown
output + --file, --ground-truth scoring, cost tracking, config commands,
environment variable docs, ProviderCapabilities, pinned dependencies. 325+ tests.

**Deliverables:**
- `replay` command — re-run synthesis or add rounds to existing transcripts
- `list` and `show` commands for transcript management
- Markdown output format
- `--file` flag for text extraction and context injection
- `--ground-truth` flag with post-debate scoring
- Cost tracking per debate (token counts, estimated cost by provider)
- `config` command for managing defaults, providers, and routing

**Done when:** Can replay past debates with different synthesizers, attach files
to queries, score debates against known answers, and see per-debate cost
breakdowns.

---

### Phase 3: Web GUI

**Goal:** NiceGUI-based web interface with two modes — a power-tool debate view
for running debates and a research dashboard for analyzing transcripts.

**Stack:**
- NiceGUI >= 3.7 (Python, async-native, Tailwind built in)
- WebSocket transport for live model response streaming
- Same Orchestrator backend as CLI — no separate API server

**Deliverables:**

**Debate View (power tool):**
- Dark mode, dense layout, monospace-heavy, minimal chrome
- Split-pane model responses with live streaming as tokens arrive
- Keyboard shortcuts: `Ctrl+Enter` submit, `Tab` panels, `/` focus query
- Panel config bar: model toggles, round count, synthesizer picker
- Inline diff: highlight changes between initial → reflection rounds
- One-click re-run with different synthesizer

**Research Dashboard (data focus):**
- Transcript browser: search, filter by date/query/models, sort
- Convergence visualization: model shift per round (bar/radar charts)
- Influence heatmap: which models move which (NxN matrix)
- Cost tracking: per-debate and cumulative spend by model and provider
- Export: filtered transcript sets as JSON/CSV

**Infrastructure:**
- `mutual-dissent serve` command (with `--port`, `--host`, `--no-open`)
- Config panel in web UI for provider keys and routing

**Done when:** `mutual-dissent serve` opens a browser with a functional debate
interface that streams responses live, and a dashboard that visualizes patterns
across saved transcripts.

---

### Phase 4: Documentation & Developer Experience

**Goal:** Comprehensive, user-facing documentation powered by Mintlify with
built-in AI assistant, MCP server integration, and LLM-optimized content.
Documentation that teaches through examples, screenshots, and video — not just
reference material.

**Platform:** Mintlify (OSS program — Pro plan, free for non-commercial open
source). Git-based workflow: markdown/MDX in-repo, GitHub sync to hosted docs
site.

**Deliverables:**

*Infrastructure:*
- Mintlify setup with GitHub sync from `docs/` directory
- Custom domain (e.g., docs.mutual-dissent.dev)
- AI Assistant configured and trained on project docs
- MCP server for querying docs from Cursor, Claude, VS Code
- `llms.txt` for AI-optimized content indexing

*User-facing content:*
- Quickstart guide — install, first debate, transcript walkthrough with
  screenshots
- CLI reference — all commands, flags, examples, terminal output screenshots
- Transcript schema docs — JSON format, field reference, example transcripts
- Config reference — TOML schema, provider setup, routing modes, environment
  variables
- Architecture overview — system diagram, data flow, extension points

*Research & multimedia:*
- Research methodology guide — using Mutual Dissent for AI behavior analysis
- Video/GIF content — CLI workflow recordings, debate-in-action demos
- Cross-repo docs consistency (CounterAgent, CounterSignal, mcp-audit)

**Done when:** `docs.mutual-dissent.dev` (or equivalent) serves a complete,
searchable documentation site with AI assistant. A new user can go from zero to
running their first debate using only the docs.

---

### Phase 5: Maturity

**Goal:** Desktop packaging, advanced research tooling, alternative topologies,
local model support, and public release polish.

**Deliverables:**
- Tauri 2 desktop wrapper (~5-15 MB native binary, system tray, global hotkey)
- Transcript analysis tooling (convergence metrics, disagreement scoring,
  influence quantification)
- Alternative debate topologies (ring, star, adversarial)
- Local model support via Ollama (hybrid panel: cloud + local)
- Additional direct providers (OpenAI, Google, xAI, Groq) as demand warrants
- Batch mode for running same query across multiple configurations
- README, documentation, and examples suitable for public release

**Done when:** Tool is useful as both a personal productivity tool and a research
platform for multi-model behavior analysis. Desktop app available. Ready for
public repo.

---

## What Success Looks Like

- A tool I actually use when the answer matters — replacing my manual
  cross-conversation workflow
- Structured dataset of multi-model debate transcripts for behavior analysis
- At least one publishable finding from consensus poisoning or convergence
  pattern research
- If public: a tool other AI power users adopt because nothing else does
  cross-vendor reflection
- A research dashboard that makes transcript analysis visual and fast

---

## Out of Scope (for now)

- Integration with AnythingLLM, LibreChat, or other frontends
- Autonomous continuous debate (human initiates, human decides when to stop)
- Fine-tuning or training based on debate outcomes
- Mobile app (Tauri 2 supports it, but not a priority)
