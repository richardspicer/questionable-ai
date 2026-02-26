# Mutual Dissent — Design

## Architecture

```
┌─────────────────────────────────────────────────┐
│              CLI / Web UI / Desktop              │
│  query, --panel, --synthesizer, --rounds, etc.   │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│                 Orchestrator                     │
│  Fan-out, round management, reflection injection │
└──────────────────────┬──────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────┐
│               Provider Router                    │
│  Routes each model to its configured provider    │
│  auto / direct / openrouter per model alias      │
└──────┬───────────┬───────────┬──────────────────┘
       │           │           │
       ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│ Anthropic│ │ OpenAI   │ │  OpenRouter   │
│ Provider │ │ Provider │ │  Provider     │
│ (direct) │ │ (direct) │ │  (fallback)   │
└────┬─────┘ └────┬─────┘ └──────┬───────┘
     │            │              │
     ▼            ▼              ▼
  Anthropic    OpenAI        OpenRouter
  Messages     Chat          Unified
  API          API           API
```

### Debate Flow (unchanged)

```
Fan-out (parallel) → Reflection Router → Reflection (parallel)
                                              │
                                     (repeat for N rounds)
                                              │
                                              ▼
                                         Synthesizer
                                              │
                              ┌────────────┼────────────┐
                              ▼            ▼            ▼
                        Terminal      JSON Log     Markdown
```

### Component Descriptions

**CLI** — Entry point. Parses user query, panel selection, synthesizer choice,
round count, and output options. Thin layer — delegates immediately to
Orchestrator.

**Web UI** — NiceGUI-based web interface with two modes: a debate view (power
tool aesthetic — dark, dense, keyboard-driven, live streaming) and a research
dashboard (card-based, data-focused, transcript browser with visualizations).
Started via `mutual-dissent serve`. Calls the same Orchestrator as the CLI.

**Desktop App** — Tauri 2 wrapper around the Web UI. Native window, system tray,
~5-15 MB binary. Optional — the web UI runs standalone in any browser.

**Orchestrator** — Core engine. Manages the debate lifecycle: initial fan-out,
round tracking, reflection injection, and synthesis invocation. Provider-agnostic
— calls the Provider Router, not any specific API client.

**Provider Router** — Dispatch layer between the Orchestrator and API providers.
For each model in a request, resolves which provider handles the call based on
config routing rules. Implements the same async interface as individual providers
so the Orchestrator doesn't know or care about routing.

**Providers** — Each provider implements a common interface (`complete()` and
`complete_parallel()`) and returns `ModelResponse`. Providers handle
vendor-specific auth, endpoints, request formatting, and response normalization.

**Reflection Router** — Between rounds, constructs reflection prompts by
injecting other models' responses. Each model sees the other panelists' output
but not its own restated. Configurable reflection prompt template.

**Synthesizer** — Final step. The user-selected model receives the full debate
context (query + all rounds) and produces a consolidated answer. Same provider
interface as panel models.

**Transcript Logger** — Writes full structured JSON for every debate. Also
produces optional Markdown summary for quick review.

---

## Provider Abstraction

### Interface

All providers implement the same async interface:

```python
from abc import ABC, abstractmethod

class Provider(ABC):
    """Base class for all model API providers."""

    @abstractmethod
    async def complete(
        self,
        model_id: str,
        prompt: str,
        *,
        model_alias: str = "",
        round_number: int = 0,
    ) -> ModelResponse: ...

    async def complete_parallel(
        self,
        requests: list[dict[str, Any]],
    ) -> list[ModelResponse]:
        """Default: fan out with asyncio.gather."""
        tasks = [self.complete(**req) for req in requests]
        return list(await asyncio.gather(*tasks))

    @abstractmethod
    async def __aenter__(self) -> Provider: ...

    @abstractmethod
    async def __aexit__(self, *exc: Any) -> None: ...
```

### Providers

| Provider | Module | Auth | Endpoint | Models |
|----------|--------|------|----------|--------|
| OpenRouter | `providers/openrouter.py` | Bearer token | `openrouter.ai/api/v1/chat/completions` | All (unified) |
| Anthropic | `providers/anthropic.py` | `x-api-key` header | `api.anthropic.com/v1/messages` | Claude family |
| OpenAI | `providers/openai.py` | Bearer token | `api.openai.com/v1/chat/completions` | GPT family |
| Google | `providers/google.py` | API key param | `generativelanguage.googleapis.com` | Gemini family |
| xAI | `providers/xai.py` | Bearer token | `api.x.ai/v1/chat/completions` | Grok family |
| Groq | `providers/groq.py` | Bearer token | `api.groq.com/openai/v1/chat/completions` | Llama, Mixtral, DeepSeek |

**Implementation order:** OpenRouter (refactor from existing client.py) → Anthropic → others (OpenAI, Google, xAI, Groq) as needed.

Each provider normalizes vendor-specific response formats into `ModelResponse`. The Orchestrator never sees raw API responses.

### Provider Router

The router sits between the Orchestrator and individual providers. It:

1. Reads routing config for the requested model alias
2. Resolves which provider handles the call
3. Initializes providers lazily (only when first needed)
4. Delegates `complete()` / `complete_parallel()` calls

```python
class ProviderRouter:
    """Routes model requests to the correct API provider.

    Supports three routing modes per model alias:
    - "auto" (default): Use direct provider if API key is available,
      fall back to OpenRouter.
    - "direct": Use the vendor's native API only. Error if no key.
    - "openrouter": Always route through OpenRouter regardless of
      available direct keys.
    """
```

For `complete_parallel()`, the router groups requests by resolved provider, fans
out each group to its provider's `complete_parallel()`, then reassembles results
in the original request order.

### Routing Resolution

```
Model alias → vendor mapping (hardcoded: claude→anthropic, gpt→openai, etc.)
                    │
                    ▼
         Config routing mode?
         ┌──────────┼──────────┐
         ▼          ▼          ▼
      "direct"   "auto"    "openrouter"
         │          │          │
         │     Key exists?     │
         │     ┌────┴────┐     │
         │     ▼         ▼     │
         │   Yes        No     │
         │     │         │     │
         ▼     ▼         ▼     ▼
      Direct  Direct  OpenRouter  OpenRouter
      Provider Provider Provider  Provider
```

---

## Configuration

### Config File

`~/.mutual-dissent/config.toml`

```toml
# Legacy — still works, maps to providers.openrouter.api_key
api_key = "sk-or-..."

[providers]
# Direct vendor API keys. Each also checks its env var.
# Env vars: OPENROUTER_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY,
#           GOOGLE_API_KEY, XAI_API_KEY, GROQ_API_KEY
openrouter = "sk-or-..."
anthropic = ""     # or set ANTHROPIC_API_KEY env var
# openai = ""      # future
# google = ""       # future
# xai = ""          # future
# groq = ""         # future

[routing]
# Per-model routing: "auto" | "direct" | "openrouter"
# Default for all models is "auto"
claude = "auto"
gpt = "openrouter"
gemini = "openrouter"
grok = "openrouter"

[defaults]
panel = ["claude", "gpt", "gemini", "grok"]
synthesizer = "claude"
rounds = 1

[model_aliases]
claude = "anthropic/claude-sonnet-4-5"
gpt = "openai/gpt-5.2"
gemini = "google/gemini-2.5-pro"
grok = "x-ai/grok-4"
```

**Backward compatibility:** The top-level `api_key` field continues to work and
maps to `providers.openrouter`. If both exist, `[providers].openrouter` takes
precedence.

### Environment Variables

| Variable | Maps to |
|----------|---------|
| `OPENROUTER_API_KEY` | `providers.openrouter` |
| `ANTHROPIC_API_KEY` | `providers.anthropic` |
| `OPENAI_API_KEY` | `providers.openai` |
| `GOOGLE_API_KEY` | `providers.google` |
| `XAI_API_KEY` | `providers.xai` |
| `GROQ_API_KEY` | `providers.groq` |

Env vars override config file values.


---

## Data Models

```python
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ModelResponse:
    """Single response from one model in one round."""
    model_id: str           # Provider-specific model identifier
    model_alias: str        # Human-readable name (e.g., "claude", "gpt")
    round_number: int       # 0 = initial, 1+ = reflection, -1 = synthesis
    content: str            # Full response text
    timestamp: datetime     # When response was received (UTC)
    token_count: int | None = None      # Tokens used (if available)
    latency_ms: int | None = None       # Response time in milliseconds
    error: str | None = None            # Error message if call failed
    provider: str | None = None         # Which provider handled this call


@dataclass
class DebateRound:
    """All responses from one round of the debate."""
    round_number: int
    round_type: str         # "initial" | "reflection" | "synthesis"
    responses: list[ModelResponse]


@dataclass
class DebateTranscript:
    """Complete record of a debate session."""
    transcript_id: str              # UUID
    query: str                      # Original user query
    panel: list[str]                # Model IDs that participated
    synthesizer_id: str             # Model ID selected for synthesis
    max_rounds: int                 # Configured reflection rounds
    rounds: list[DebateRound] = field(default_factory=list)
    synthesis: ModelResponse | None = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)  # Version, config, etc.
```

---

## Schema

### Transcript JSON Format

```json
{
  "transcript_id": "uuid",
  "query": "user's original question",
  "panel": ["anthropic/claude-sonnet-4-5", "openai/gpt-5.2"],
  "synthesizer_id": "anthropic/claude-sonnet-4-5",
  "max_rounds": 1,
  "created_at": "2026-02-21T15:30:00Z",
  "rounds": [
    {
      "round_number": 0,
      "round_type": "initial",
      "responses": [
        {
          "model_id": "anthropic/claude-sonnet-4-5",
          "model_alias": "claude",
          "round_number": 0,
          "content": "...",
          "timestamp": "2026-02-21T15:30:01Z",
          "token_count": 450,
          "latency_ms": 2100,
          "error": null,
          "provider": "anthropic"
        }
      ]
    }
  ],
  "synthesis": { "..." },
  "metadata": {
    "version": "0.1.0",
    "providers_used": ["anthropic", "openrouter"]
  }
}
```

### Transcript Storage

```
~/.mutual-dissent/
└── transcripts/
    └── 2026-02-21_uuid-short.json
```

---

## CLI Interface

> **Shorthand:** `dissent` is a built-in alias. Both `dissent` and `mutual-dissent` work identically.

```
mutual-dissent
  ask         Submit a query to the panel for debate
  replay      Re-run synthesis on an existing transcript
  list        List saved transcripts
  show        Display a transcript
  config      Manage default settings
  serve       Start the web UI server
```

### ask

```
mutual-dissent ask "What is the most effective approach to securing MCP servers?"
```

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--panel` | str (comma-sep) | `claude,gpt,gemini,grok` | Models to include on the panel |
| `--synthesizer` | str | `claude` | Model to perform final synthesis |
| `--rounds` | int | `1` | Number of reflection rounds (1-3) |
| `--output` | str | `terminal` | Output format: `terminal`, `json`, `markdown` |
| `--no-save` | flag | false | Don't save transcript to disk |
| `--verbose` | flag | false | Show individual model responses as they arrive |
| `--ground-truth` | str | None | Known correct answer for post-debate scoring |
| `--file` | path | None | File to include as context |

### serve

```
mutual-dissent serve
mutual-dissent serve --port 8080 --host 0.0.0.0
```

Starts the NiceGUI web server. Opens browser automatically unless `--no-open`.

### replay

```
mutual-dissent replay <transcript-id> --synthesizer grok --rounds 1
```

### Model Aliases

| Alias | OpenRouter Model ID | Direct Provider |
|-------|---------------------|-----------------|
| `claude` | `anthropic/claude-sonnet-4-5` | Anthropic Messages API |
| `gpt` | `openai/gpt-5.2` | OpenAI Chat API |
| `gemini` | `google/gemini-2.5-pro` | Google Generative AI API |
| `grok` | `x-ai/grok-4` | xAI Chat API |

Model IDs verified against OpenRouter offerings as of 2026-02-21. Direct provider
model IDs may differ from OpenRouter IDs — each provider maps accordingly.


---

## Web UI Architecture

### Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Framework | NiceGUI | Pure Python web UI with Tailwind, async-native |
| Styling | Tailwind CSS (built into NiceGUI) | Dark mode, utility-first |
| Desktop | Tauri 2 | Optional native wrapper (~5-15 MB) |
| Transport | WebSocket (NiceGUI default) | Live streaming of model responses |

### Why NiceGUI

- Pure Python — no JS build chain, no React, no npm
- Async-native — pairs naturally with the async orchestrator
- Tailwind built in — dark mode and dense layouts without CSS files
- No full-script re-run — only changed UI elements update (unlike Streamlit)
- WebSocket by default — live streaming without polling

### Two-Mode Design

The web UI has two distinct views optimized for different workflows:

**Debate View — Power Tool**

Dense, keyboard-driven interface for running and reviewing debates.

- Dark mode, monospace-heavy, minimal chrome
- Split panes: query input left, model responses right (or vertical stack)
- Live streaming: model responses appear as tokens arrive
- Keyboard shortcuts: `Ctrl+Enter` submit, `Tab` between panels, `/` focus query
- Panel config bar: model toggles, round count, synthesizer picker
- Inline diff: highlight what changed between initial and reflection rounds
- One-click re-run with different synthesizer

**Research Dashboard — Data Focus**

Card-based interface for analyzing debate transcripts and patterns.

- Transcript browser: search, filter by date/query/models, sort by token count
- Convergence visualization: how much each model shifted per round (bar/radar charts)
- Influence heatmap: which models move which (NxN matrix across transcripts)
- Agreement/disagreement tagging: highlight consensus vs. contested claims
- Cost tracking: per-debate and cumulative spend by model and provider
- Export: filtered transcript sets as JSON/CSV for external analysis

### Server Architecture

```
mutual-dissent serve
        │
        ▼
┌────────────────────┐
│   NiceGUI Server   │
│   (uvicorn/ASGI)   │
├────────────────────┤
│   Debate View      │──→ Orchestrator (same as CLI)
│   Research Dash    │──→ Transcript store (read-only)
│   Config Panel     │──→ Config (read/write)
└────────────────────┘
```

The web UI is a **client** of the same Orchestrator the CLI uses. No separate
backend — NiceGUI runs Python server-side and pushes UI updates over WebSocket.

### Desktop Wrapper (Tauri 2)

Optional. Wraps the NiceGUI web UI in a native window.

- Tauri 2 with Rust backend, no Electron bloat
- Points webview at `localhost:{port}` where NiceGUI runs
- System tray icon with global hotkey
- Auto-starts NiceGUI server on launch, shuts down on close
- Single binary distribution (~5-15 MB)
- Zero JavaScript required — Tauri just wraps the existing web UI

---

## Prompt Templates

### Initial Round

```
You are participating in a multi-model panel discussion. Answer the following
query to the best of your ability. Be thorough but concise.

Query: {query}
```

### Reflection Round

```
You previously answered a query as part of a multi-model panel. Below is your
original response, followed by how other models on the panel responded.

Your previous response:
{own_response}

Other panel members' responses:
{other_responses}

Reflect on the other responses. Where do you agree? Where do you disagree?
What did they identify that you missed? What did you get right that they missed?
Provide your refined answer to the original query.

Original query: {query}
```

### Synthesis Prompt

```
You are the designated synthesizer for a multi-model panel discussion. Below is
the full debate transcript including initial responses and any reflection rounds.

Original query: {query}

{formatted_transcript}

Synthesize the strongest elements from all panel members into a single,
well-reasoned response. Note where the panel reached consensus and where
significant disagreements remain. Do not simply concatenate — produce a
coherent, unified answer.
```

---

## Extension Points

### Adding a New Provider

1. Create `src/mutual_dissent/providers/{vendor}.py`
2. Implement `Provider` abstract class (`complete`, `complete_parallel`, context manager)
3. Register in `ProviderRouter` vendor mapping
4. Add env var and config key to `Config`
5. Add model alias mapping if new models are introduced

### Adding a New Model (existing provider)

1. Add alias → model ID mapping to config
2. No code changes required

### Adding a New Model (OpenRouter only)

1. Add alias and OpenRouter model ID to config mapping
2. No code changes required — the OpenRouter provider is generic

### Adding a New Debate Topology

Currently: full mesh (every model sees every other model's response). Future:

- **Ring** — each model sees only the previous model's response
- **Star** — all responses route through synthesizer as coordinator
- **Adversarial** — explicitly assign "devil's advocate" role to one model

Add topology as a strategy class that implements the reflection routing interface.

### Adding a New Output Format

1. Create a formatter that takes a `DebateTranscript` and produces output
2. Register in CLI `--output` choices

### Adding Ground Truth Scoring

The `--ground-truth` flag enables post-debate analysis:
- Score each model's initial response against the known answer
- Score each model's reflection response — did reflection improve or degrade?
- Score the synthesis — is the final answer better than any individual?
- Output a scoring summary alongside the transcript

---

## Security Considerations

- **API key management** — Provider keys stored in environment variables or
  `~/.mutual-dissent/config.toml`, never in code or transcripts. Multiple
  keys now (one per provider) — gitleaks pre-commit hook catches leaks.
- **Transcript sanitization** — Transcripts may contain sensitive query content.
  The `provider` field in responses reveals which API was used. No automatic
  sharing — user controls what leaves their machine.
- **Prompt injection via model responses** — A model's response in round 1
  becomes part of another model's prompt in round 2. A malicious or manipulated
  response could influence other models through the reflection prompt. This is
  an inherent property of the architecture and a research area, not a bug to fix.
- **Cost control** — Each debate is 4 + 4 + 1 = 9 API calls minimum. No runaway
  loops — rounds are hard-capped at 3.
- **Web UI exposure** — `serve` binds to localhost by default. Binding to
  `0.0.0.0` is opt-in and should only be used on trusted networks. No auth
  layer in MVP — the UI has full access to config and API keys.
- **Direct API key isolation** — Each provider only receives its own key.
  The Anthropic provider never sees the OpenAI key and vice versa.

---

## Documentation Standards

- Google-style docstrings (Args, Returns, Raises, Example)
- New modules get docstrings when created, not retrofitted
- Inline comments for non-obvious logic only
