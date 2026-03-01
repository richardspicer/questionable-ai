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

### Debate Flow

```
Fan-out (parallel) → Reflection Router → Reflection (parallel)
                                              │
                                     (repeat for N rounds)
                                              │
                                              ▼
                                         Synthesizer
                                              │
                                    (if --ground-truth)
                                              │
                                              ▼
                                      Scoring (judge)
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
— calls the Provider Router, not any specific API client. Supports optional
per-panelist context injection (`panelist_context`) and round-level event hooks
(`on_round_complete`) for research integration.

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

**Scoring** — Optional ground-truth evaluation via LLM-as-judge. When
`--ground-truth` is provided, sends the synthesis and reference answer to the
judge model (currently the synthesizer) for accuracy/completeness scoring.
Scores stored in `synthesis.analysis["ground_truth_score"]` and
`transcript.metadata["scores"]`. Implemented in `scoring.py`.

**Pricing** — Session-scoped pricing cache that fetches per-model token
pricing from the OpenRouter `/api/v1/models` endpoint (public, no auth).
Computes USD cost per response from input/output token counts. Handles
vendor-native model IDs by mapping through the config alias system. Graceful
degradation — if pricing is unavailable, cost is omitted from output rather
than failing the debate. Implemented in `pricing.py`.

**Transcript Logger** — Writes full structured JSON for every debate. Also
produces optional Markdown summary for quick review.

---

## Provider Abstraction

### Interface

All providers implement the same async interface:

```python
from abc import ABC, abstractmethod
from enum import Enum

class Vendor(Enum):
    """Supported inference providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    XAI = "xai"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


@dataclass
class RoutedRequest:
    """A model request annotated with routing info."""
    vendor: Vendor
    model_id: str
    model_alias: str
    round_number: int
    messages: list[dict[str, Any]]
    context: str | None = None  # Per-panelist pre-prompt content


@dataclass
class RoutingDecision:
    """How a request was routed."""
    vendor: Vendor
    mode: str           # "auto" | "direct" | "openrouter"
    via_openrouter: bool


class Provider(ABC):
    """Base class for all model API providers."""

    @abstractmethod
    async def complete(
        self,
        model_id: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        prompt: str | None = None,
        model_alias: str = "",
        round_number: int = 0,
    ) -> ModelResponse:
        """Send a completion request.

        Accepts either `messages` (list of chat messages) or `prompt`
        (single user message string). Exactly one must be provided.
        """
        ...

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
| Ollama | `providers/ollama.py` | None (local) | `http://10.0.40.20:11434/api/chat` | Open-weight models |

**Implementation order:** OpenRouter (refactor from existing client.py) → Anthropic → others as needed.

Each provider normalizes vendor-specific response formats into `ModelResponse`. The Orchestrator never sees raw API responses.

### Provider Capabilities

Minimal for Phase 1.5 — only `max_context_tokens`, pulled dynamically from
OpenRouter's `/api/v1/models` endpoint. No hardcoded pricing or capability
matrices. Extended capabilities (`supports_tools`, `supports_vision`,
`supports_json_mode`, detailed pricing) deferred to Phase 4.

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

    Attaches a RoutingDecision to every ModelResponse for transcript
    provenance.
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
# Legacy — still works, maps to providers.openrouter_api_key
api_key = "sk-or-..."

[providers]
# Direct vendor API keys. Each also checks its env var.
# Env vars: OPENROUTER_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY,
#           GOOGLE_API_KEY, XAI_API_KEY, GROQ_API_KEY
openrouter_api_key = "sk-or-..."
anthropic_api_key = ""     # or set ANTHROPIC_API_KEY env var
# openai_api_key = ""      # future
# google_api_key = ""       # future
# xai_api_key = ""          # future
# groq_api_key = ""         # future

[routing]
# Per-model routing: "auto" | "direct" | "openrouter"
# Default for all models is "auto"
default_mode = "auto"
claude = "auto"
gpt = "openrouter"
gemini = "openrouter"
grok = "openrouter"

[model_aliases]
# Dual IDs per alias: OpenRouter format and vendor-native format.
# Verify against vendor docs and OpenRouter before hardcoding — IDs change.
claude.openrouter = "anthropic/claude-sonnet-4-5"
claude.direct = "claude-sonnet-4-5-20250929"
gpt.openrouter = "openai/gpt-4.1"
gpt.direct = "gpt-4.1"
gemini.openrouter = "google/gemini-2.5-pro"
grok.openrouter = "x-ai/grok-3"

[defaults]
panel = ["claude", "gpt", "gemini", "grok"]
synthesizer = "claude"
rounds = 1
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
class ExperimentMetadata:
    """Metadata linking a debate to a research experiment."""
    experiment_id: str              # Groups related runs
    source_tool: str = "manual"     # "countersignal" | "counteragent" | "manual"
    campaign_id: str | None = None  # Links to external campaign/scan
    condition: str = ""             # Experimental variable description
    variables: dict = field(default_factory=dict)  # Parameter values
    finding_ref: str | None = None  # e.g. "MD-003", "MCP-001"


@dataclass
class ModelResponse:
    """Single response from one model in one round."""
    model_id: str           # Provider-specific model identifier
    model_alias: str        # Human-readable name (e.g., "claude", "gpt")
    round_number: int       # 0 = initial, 1+ = reflection, -1 = synthesis
    role: str               # "initial" | "reflection" | "synthesis"
    content: str            # Full response text
    timestamp: datetime     # When response was received (UTC)
    token_count: int | None = None      # Tokens used (if available)
    input_tokens: int | None = None     # Prompt/input tokens
    output_tokens: int | None = None    # Completion/output tokens
    latency_ms: int | None = None       # Response time in milliseconds
    error: str | None = None            # Error message if call failed
    provider: str | None = None         # Vendor enum value
    routing: dict | None = None         # Serialized RoutingDecision
    analysis: dict = field(default_factory=dict)  # Reserved for scoring


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
    metadata: dict = field(default_factory=dict)
    # metadata includes:
    #   version: str
    #   resolved_config: dict       — full effective config snapshot
    #   providers_used: list[str]
    #   panelist_context: dict      — alias → context string (if provided)
    #   experiment: ExperimentMetadata — research experiment linkage (if provided)
    #   stats: dict                 — precomputed on write:
    #     total_tokens, per_model token counts (incl. input/output split),
    #     total_cost_usd (computed from OpenRouter pricing API),
    #     per_model cost_usd,
    #     rounds_to_convergence, disagreement_count (placeholders)
```

---

## Schema

### Transcript JSON Format

```json
{
  "transcript_id": "uuid",
  "query": "user's original question",
  "panel": ["anthropic/claude-sonnet-4-5", "openai/gpt-4.1"],
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
          "role": "initial",
          "content": "...",
          "timestamp": "2026-02-21T15:30:01Z",
          "token_count": 450,
          "input_tokens": 300,
          "output_tokens": 150,
          "latency_ms": 2100,
          "error": null,
          "provider": "anthropic",
          "routing": {
            "vendor": "anthropic",
            "mode": "auto",
            "via_openrouter": false
          },
          "analysis": {}
        }
      ]
    }
  ],
  "synthesis": { "..." },
  "metadata": {
    "version": "0.2.0",
    "providers_used": ["anthropic", "openrouter"],
    "panelist_context": { "claude": "RAG context..." },
    "experiment": {
      "experiment_id": "exp-001",
      "source_tool": "countersignal",
      "campaign_id": "camp-42",
      "condition": "rag-augmented",
      "variables": { "context_size": 4096 },
      "finding_ref": "MD-003"
    },
    "resolved_config": { "..." },
    "stats": {
      "total_tokens": 4200,
      "per_model": {
        "claude": { "tokens": 1800, "input_tokens": 1200, "output_tokens": 600, "calls": 2, "cost_usd": 0.0126 },
        "gpt": { "tokens": 2400, "input_tokens": 1600, "output_tokens": 800, "calls": 2, "cost_usd": 0.016 }
      },
      "total_cost_usd": 0.0286,
      "convergence": {}
    }
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
  config test Smoke-test routing for all configured aliases
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
| `--ground-truth-file` | path | None | File containing reference answer for scoring |
| `--file` | path | None | Write output to file instead of stdout |

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

| Alias | OpenRouter Model ID | Direct Model ID | Direct Provider |
|-------|---------------------|-----------------|-----------------|
| `claude` | `anthropic/claude-sonnet-4-5` | `claude-sonnet-4-5-20250929` | Anthropic Messages API |
| `gpt` | `openai/gpt-4.1` | `gpt-4.1` | OpenAI Chat API |
| `gemini` | `google/gemini-2.5-pro` | — | Google Generative AI API |
| `grok` | `x-ai/grok-3` | — | xAI Chat API |

Model IDs are placeholders — verify against OpenRouter and vendor docs before
implementation. IDs change frequently.


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

### Scoring Prompt

```
You are evaluating the quality of an AI-generated answer against a known
correct reference answer.

Original query: {query}

Reference answer (ground truth):
{ground_truth}

Response to evaluate:
{synthesis}

Score the response on two dimensions, each from 1 to 5:

- Accuracy (1-5): How factually correct is the response compared to the
  reference? 5 = fully correct, 1 = fundamentally wrong.
- Completeness (1-5): How much of the reference answer's key information
  does the response cover? 5 = covers everything, 1 = misses almost all points.

Respond in EXACTLY this format (no other text):
ACCURACY: <score>
COMPLETENESS: <score>
EXPLANATION: <1-3 sentence explanation of the scores>
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

### Cross-Tool Research Integration

Mutual Dissent serves as the multi-model observation platform for the broader
security research portfolio (CounterSignal, CounterAgent). The following
extension points enable research integration without modifying the core debate
loop. See `Lab/Cross-Tool Research Directions.md` for the full research agenda.

**Per-panelist context injection (implemented):**
`run_debate()` and `run_replay()` accept an optional `panelist_context: dict[str, str]`
mapping model alias to context string. When provided, `_inject_context()` prepends
the context to each panelist's prompt in every round (initial and reflection).
Context persists across rounds — a RAG-augmented model stays RAG-augmented
throughout the debate. `RoutedRequest.context` field establishes the interface
contract. Stored in `transcript.metadata["panelist_context"]` for replay
reconstruction. No CLI flags — programmatic interface for experiment runner,
payload source, and Web UI consumers.

Consumers: RXP retrieval-optimized documents, CounterAgent inject payloads,
consensus poisoning pre-prompts for controlled experiments.

**Round-level event hooks (implemented):**
`run_debate()` and `run_replay()` accept an optional `on_round_complete` async
callback (`Callable[[DebateRound], Awaitable[None]]`). Fires after each round
completes (initial, each reflection, synthesis) with the completed `DebateRound`.
Exceptions in the callback are caught and logged via `logger.exception()` — a
misbehaving callback cannot abort the debate. Implemented in `_fire_round_hook()`.

Consumers: Web UI live debate view, research instrumentation (degradation curve
measurement, per-round compliance tracking), detection rule triggers.

**Experiment metadata schema (implemented):**
`ExperimentMetadata` dataclass in `models.py` with `to_dict()`/`from_dict()`.
Stored in `DebateTranscript.metadata["experiment"]`. Serializes to/from JSON
transcripts automatically — `DebateTranscript.to_dict()` converts the instance
to a dict, `_parse_transcript_file()` reconstitutes it on load. Displayed in
both Rich terminal and markdown output when present.

```python
@dataclass
class ExperimentMetadata:
    experiment_id: str              # Groups related runs
    source_tool: str = "manual"     # "countersignal" | "counteragent" | "manual"
    campaign_id: str | None = None  # Links to CounterSignal campaign or CounterAgent scan
    condition: str = ""             # Experimental variable description
    variables: dict = field(default_factory=dict)  # Parameter values for this run
    finding_ref: str | None = None  # e.g. "MD-003", "MCP-001"
```

Makes transcripts self-describing and queryable across tools. The research
dashboard can group/filter by experiment.

**Payload source protocol (future):**
A minimal interface abstracting where debate inputs come from:
`get_query() -> str` and `get_context(model_alias: str) -> str | None`.
Default: user-provided query, no per-model context. Enables programmatic
integration with CounterSignal payload libraries and CounterAgent inject output.

**Finding output adapter (future):**
Export experiment results in a format compatible with CounterAgent's Finding
model (Severity, CVSS, OWASP/ATLAS category). Enables cross-tool finding
correlation without hard dependencies.

### Ground Truth Scoring (Implemented)

The `--ground-truth` flag enables post-debate scoring against a known-correct
reference answer. Currently scores the synthesis only (v1). Uses the synthesizer
model as judge (self-evaluation bias — documented limitation).

**Current:** `scoring.py` — `GroundTruthScore` dataclass, `parse_score_response()`,
`score_synthesis()`. Scores stored in `synthesis.analysis["ground_truth_score"]`.

**Future extensions:**
- `--judge` flag to use a different model for scoring
- Score individual model responses (not just synthesis)
- Custom rubrics or scoring dimensions
- Batch scoring across multiple transcripts

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
