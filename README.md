# otop — Real-Time Ollama Token Monitor (tokens/sec, CPU, GPU & VRAM in your terminal)

**otop** is an htop-style terminal dashboard for [Ollama](https://ollama.com) that shows
**live output token counts and tokens per second**, **CPU usage**, **threads**,
**GPU utilization**, **VRAM / unified memory**, and the **models currently loaded** —
all updating in real time while your local LLM generates.

![otop live demo — real-time Ollama token counting, tokens per second, GPU utilization and CPU usage in the terminal](assets/demo.gif)

*Real capture: gemma4 (8B) generating on an Apple M5 — output tokens climbing at
~39 tok/s, GPU at 97%, model 100% on GPU.*

## Features

- 🔢 **Live token tracking** — output tokens and prompt tokens counted *as they stream*,
  per-request tok/s, session totals, and a tok/s sparkline. Final numbers are reconciled
  against Ollama's own `eval_count`, so they're exact.
- ⚡ **Per-request table** — model, endpoint, output tokens, tokens/sec, elapsed time,
  and state (`generating`, `thinking/loading`, `done`).
- 🧠 **CPU & threads** — total and per-core CPU, load average, plus CPU%, **thread count**
  and memory of every Ollama process (server, model runner, menubar app).
- 🎮 **GPU monitoring on Apple Silicon** — device/renderer/tiler utilization % and GPU
  memory in use, read from `ioreg`. **No sudo required.**
- 📦 **Loaded models** — size, VRAM, GPU/CPU placement split, context length, and
  keep-alive expiry countdown (from `/api/ps`).
- 🪶 **Single file, no daemon** — one Python script using `rich`, `psutil`, `aiohttp`.

## Quick start

```sh
git clone https://github.com/TiniLLM/ollama-token-monitor.git
cd ollama-token-monitor
./otop.sh
```

First run creates a local `.venv` and installs the three dependencies. Quit with `Ctrl-C`.

Or manually:

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python otop.py
```

## Getting live token counts

Ollama's API doesn't expose other clients' in-flight generations, so otop runs a
**transparent proxy on port 11435** that forwards everything to Ollama (11434) and counts
streamed tokens as they pass through. Point whatever you use at the proxy:

```sh
# ollama CLI
OLLAMA_HOST=127.0.0.1:11435 ollama run llama3.2

# curl / scripts (native Ollama API)
curl http://127.0.0.1:11435/api/generate -d '{"model":"llama3.2","prompt":"hello"}'

# OpenAI-compatible clients & SDKs
export OPENAI_BASE_URL=http://127.0.0.1:11435/v1
```

```python
# python (openai sdk)
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:11435/v1", api_key="ollama")
```

All response formats are counted: native NDJSON streams (`/api/generate`, `/api/chat`),
OpenAI-style SSE (`/v1/chat/completions`, `/v1/completions`), and non-streaming responses.
Traffic sent straight to 11434 still works normally — it just isn't token-counted; the
CPU, GPU, and model panels work regardless.

![otop screenshot — Ollama terminal dashboard with token counter, GPU and CPU panels](assets/screenshot.svg)

## The panels, explained

| Panel | What it shows | Source |
|---|---|---|
| **tokens** | session output/prompt totals, live tok/s + sparkline, last requests with per-request tok/s and state | counted by the proxy, reconciled with `eval_count` / `usage` |
| **cpu / system** | total + per-core CPU, load average, RAM; CPU%, threads, RSS per Ollama process | `psutil` |
| **gpu (apple silicon)** | device / renderer / tiler utilization %, GPU memory in use vs wired | `ioreg -c IOAccelerator` (no sudo) |
| **loaded models** | size, VRAM, `100% GPU` vs partial CPU offload, context length, keep-alive expiry | Ollama `/api/ps` |

## Reasoning ("thinking") models

Reasoning models such as gemma4 spend tokens thinking before they answer:

- On newer Ollama builds that stream a `thinking` field, otop counts those tokens live.
- On builds that don't stream thinking chunks, the request shows **`thinking/loading`**
  with no token flow, then the full count (thinking + visible tokens) lands when the
  request completes — reconciled from `eval_count`, so totals stay exact.
- Pass `"think": false` in your request to skip thinking and watch tokens stream normally.

## Options

```
./otop.sh                        # dashboard + token proxy on :11435
./otop.sh --no-proxy             # dashboard only
./otop.sh --proxy-port 9999      # proxy on a different port
./otop.sh --host 0.0.0.0         # accept proxy clients from other machines
./otop.sh --upstream http://other-host:11434
./otop.sh --refresh 8            # UI redraws per second (default 4)
```

The proxy has no authentication, so `--host 0.0.0.0` lets anyone who can reach the port
use your Ollama server. Only use it on a trusted network.

## How it works

```
your client ──► otop proxy :11435 ──► ollama :11434 ──► llama-server (model runner)
   (unmodified      │ counts each                            ▲
    passthrough)    │ streamed token                         │
                    ▼                                        │
                  otop TUI ◄── psutil (CPU/threads/RAM) ◄────┘
                           ◄── ioreg  (GPU util + memory)
                           ◄── /api/ps (loaded models, VRAM)
```

The proxy is a transparent byte-for-byte passthrough — it never modifies requests or
responses, adds no measurable latency to token streaming, and every Ollama and
OpenAI-compatible endpoint works through it.

## Requirements & compatibility

- Python 3.9+
- Ollama running locally (any recent version; tested on 0.31)
- **macOS Apple Silicon** (M1–M5): all panels including GPU
- **Linux / Intel Mac**: token, CPU, and model panels work; the Apple GPU panel shows
  blanks (NVIDIA/AMD support welcome — PRs open)

## FAQ

**Tokens aren't counting?** Your client is talking to 11434 directly. Point it at the
proxy: `OLLAMA_HOST=127.0.0.1:11435`.

**Port 11435 is busy?** `./otop.sh --proxy-port 9999`.

**Is the token count exact?** While streaming it counts one chunk = one token (how Ollama
streams); on completion it snaps to Ollama's official `eval_count` / `usage` numbers.

**Does it slow generation down?** No — chunks are forwarded before they're parsed, and
counting is a JSON decode per token on localhost.

## License

[MIT](LICENSE) © TiniLLM Labs

---

*otop is useful when you want to: monitor Ollama tokens per second in the terminal, track
LLM output token usage locally, watch Apple Silicon GPU utilization during LLM inference,
count tokens for local llama / gemma / qwen / mistral models, or observe local LLM
performance without any cloud telemetry.*
