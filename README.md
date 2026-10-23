# ◐ Basal Station

**A local, multi-backend control plane for [basal typed-decision models](https://huggingface.co/Remek) — one stack for your MacBook (Apple Silicon / MLX / Metal) and your Linux box (NVIDIA, even with only 3 GB VRAM), with model + backend choice built in and decisions grounded in your [knowledge-palace](https://github.com/) MCP knowledge base.**

Basal models don't generate text: they read a *state* (message, document, trace, image, audio clip) and return **calibrated probabilities over answer options you define** — routing, yes/no, scores, tool selection — in a single forward pass. Basal Station is the missing local cockpit around them:

- **Backend auto-selection** — detects Apple Silicon vs. CUDA vs. CPU and picks the best `basal-serve` mode: `mlx` on the Mac, `gguf` with **computed partial GPU offload (`-ngl`)** on small CUDA cards, `fp8`/vLLM on big GPUs, `eager` for full omni media support.
- **Embedded model catalog** — the whole `Remek/basal-*` family (omni 1.0/1.1, 2.0, 1.5, 1.0; 1B–12B) with GGUF / MLX / FP8 / safetensors repos, cache detection and one-click downloads.
- **knowledge-palace integration** — a persistent MCP client retrieves passages from your personal knowledge base and grounds every decision state.
- **Web console + HTTP API** — build states and typed questions visually, inspect probability bars, switch engines at runtime, tail engine logs.

```
┌────────────────────────────────────────────────────────────────────┐
│  Browser console (vanilla JS)  ·  http://127.0.0.1:8800            │
├────────────────────────────────────────────────────────────────────┤
│  basal-station FastAPI                                             │
│   /api/decide ─── grounding ─── MCP stdio ── knowledge-palace (kp) │
│       │                                                            │
│       └── /v1/systemone proxy ── basal-serve subprocess            │
│                                     mode: mlx | gguf | fp8 |      │
│                                           eager | mps              │
├────────────────────────────────────────────────────────────────────┤
│  hardware: M5 Pro (unified mem)      │  RTX … 3 GB VRAM            │
│  default: MLX 8-bit (13 GB)          │  default: GGUF Q4_K_M       │
│                                      │  -ngl ≈ f(VRAM) partial     │
└────────────────────────────────────────────────────────────────────┘
```

## Install

### macOS (Apple Silicon, e.g. M5 Pro)

```bash
git clone <this repo> && cd basal-station
./scripts/setup-macos.sh
source .venv/bin/activate
basal-station download omni-1.0-max --format mlx   # ~13 GB, 8-bit MLX
cp config/basal-station.example.yaml basal-station.yaml   # set knowledge.command
basal-station serve                                # UI at http://127.0.0.1:8800
```

Default path on the Mac: **MLX 8-bit** backend (fast, Metal-native). `eager` (PyTorch) enables omni **image/audio** inputs if your unified memory fits the bf16 weights (~24 GB for the 12B).

### Linux (NVIDIA — tuned for small cards like 3 GB VRAM)

```bash
git clone <this repo> && cd basal-station
./scripts/setup-linux.sh
source .venv/bin/activate
basal-station download 2.0-4.5B --format gguf --quant Q4_K_M   # ~2.8 GB → nearly full offload
cp config/basal-station.example.yaml basal-station.yaml   # set knowledge.command
basal-station serve
```

Default path on small CUDA cards: **GGUF + computed `-ngl` partial offload**. The console shows the estimated offload percentage per backend before you start. On a 3 GB card:

| Model | GGUF Q4_K_M | expected offload |
|---|---|---|
| `basal-2.0-4.5B` | ~2.8 GB | ~80% of layers |
| `basal-omni-1.0-max` (12B) | ~7.4 GB | ~30% of layers, rest in RAM |

> **llama.cpp:** the `gguf` modes shell out to llama.cpp — install a CUDA build (`scripts/setup-linux.sh` prints exact commands) or keep it CPU-only.

## The decision API

Everything the UI does is a thin client over two HTTP calls:

```bash
curl -s localhost:8800/api/decide -H 'content-type: application/json' -d '{
  "state": "Client opened a business account 2 weeks ago. PEP screening hit 71% fuzzy match...",
  "questions": {
    "routing": {
      "type": "choice",
      "instructions": "Where should this KYC escalation go first?",
      "criteria": {
        "auto_clear": "PEP match is a false positive",
        "compliance_officer": "human review",
        "edd_team": "enhanced due diligence"
      },
      "max_error": 0.05
    },
    "needs_edd": { "type": "noul", "instructions": "Does this need EDD?" }
  },
  "knowledge": { "enabled": true, "top_k": 5 }
}'
```

Response: per-question `choice`, `probabilities`, `confidence`, optional `deferred` (certified via `max_error`), engine `usage`, plus the `knowledge` block (query, hits, latency) and the exact `grounded_state` that was sent to the model.

`POST /api/decide/raw` is a passthrough to the engine's `/v1/systemone` with any extra fields basal supports. `orders: 2` in the request mitigates option-order sensitivity at double cost.

### knowledge-palace wiring

`basal-station` is an **MCP client** (stdio). Point it at your knowledge-palace install:

```yaml
knowledge:
  enabled: true
  command: ["/Users/you/Documents/knowledge-palace/.venv/bin/kp", "mcp"]
```

Tools used: `context` (decision grounding), `search`, `ask`, `similar` (all exposed under `/api/knowledge/*`). Env shortcut: `KP_COMMAND="/path/to/kp mcp"`.

## Backends & when they're picked

| Backend key | basal-serve mode | Weights | Needs | Notes |
|---|---|---|---|---|
| `mlx` | `mlx` | MLX 8-bit | Apple Silicon | Default on the Mac |
| `gguf-cuda` | `gguf` | GGUF | Linux + NVIDIA | `-ngl` computed from VRAM (0.90 × VRAM − ctx overhead, per-layer bytes) |
| `gguf-metal` | `gguf` | GGUF | macOS | Alternative to MLX |
| `fp8-vllm` | `vllm` | FP8 | NVIDIA ≥ ~1.25× FP8 size | Hopper/Blackwell |
| `eager` | `eager` | safetensors | RAM/VRAM ≥ bf16 size | **Only backend with omni media** (`--media on`) |
| `mps` | `mps` | safetensors | macOS | PyTorch fallback |
| `gguf-cpu` | `gguf` | GGUF | anything | Last resort |

Inspect the ranked plan without starting anything:

```bash
curl -s "localhost:8800/api/backends?model=omni-1.0-max" | jq
curl -s "localhost:8800/api/plan?model=omni-1.0-max&quant=Q4_K_M" | jq
```

## CLI

```
basal-station serve                    # web console + API
basal-station status                   # hardware, knowledge base, engine
basal-station models                   # catalog with cache state
basal-station download KEY --format gguf [--quant Q4_K_M]
basal-station ask "question"           # knowledge-palace, from the terminal
basal-station search "query" [--limit 5]
basal-station decide payload.json      # one-shot against a running engine
```

## API surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/system` | hardware, engine, knowledge, suggested models |
| GET | `/api/models?refresh=` | catalog + cache state |
| POST | `/api/models/download` | background HF download w/ progress (`/api/models/downloads`) |
| GET | `/api/backends?model=` | ranked backends w/ offload estimates |
| POST | `/api/engine/start` · `/stop` · GET `/status` · `/logs` | runtime engine switching |
| POST | `/api/decide` · `/api/decide/raw` | grounded / raw typed decision |
| GET/POST | `/api/knowledge/{status,search,ask,context,similar}` | knowledge-palace |
| GET | `/api/plan?model=&quant=` | pure VRAM/-ngl plan |

## Notes & limits

- The basal engine (`basal-serve`) is a separate install — `basal-station` finds it on `PATH` or as a sibling in the same venv (`engine.command` in the YAML).
- GGUF/MLX/vLLM routes are **text-only**; images/audio require the `eager` backend and an omni model.
- No authentication is exposed by design: everything binds to `127.0.0.1`. Don't proxy it publicly without adding auth in front.
- Model sizes for VRAM planning are estimates from parameter counts (Q4_K_M ≈ 0.62 B/param, FP8 ≈ 1.15, bf16 ≈ 2.05); the console labels them as `~GB`.
- Apache-2.0, same as the basal engine and the model weights.

## Repository layout

```
src/basal_station/
  config.py            YAML + env settings
  hardware.py          platform/chip/RAM/nvidia-smi detection
  catalog.py           model family + per-format repos + size math
  backends/
    base.py            backend profiles (basal-serve modes)
    registry.py        ranking, -ngl planning, arg building
    manager.py         subprocess lifecycle, health, /v1/systemone proxy
  knowledge/
    client.py          persistent MCP stdio client (knowledge-palace)
    enrich.py          retrieval → grounded state formatting
  api/                 FastAPI routes, schemas, HF download jobs
  ui/                  the web console (no build step)
scripts/               setup-macos.sh · setup-linux.sh
tests/                 unit tests (no model download needed)
```
