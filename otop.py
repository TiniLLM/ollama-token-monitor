#!/usr/bin/env python3
"""otop — live terminal dashboard for local Ollama.

Shows, refreshed in real time:
  * output tokens / prompt tokens / tok/s per request and session totals
    (via a transparent proxy: point your client at --proxy-port instead of 11434)
  * CPU usage and thread count of the ollama server + model runner processes
  * system CPU (overall + per-core), load average, RAM
  * Apple Silicon GPU utilization and GPU memory (via ioreg, no sudo)
  * models currently loaded: size, GPU/CPU split, context length, expiry

Usage:
    python otop.py                     # dashboard + proxy on :11435
    python otop.py --no-proxy          # dashboard only
    python otop.py --proxy-port 9999   # custom proxy port

Then send traffic through the proxy to get live token counts:
    OLLAMA_HOST=127.0.0.1:11435 ollama run llama3.2
    curl http://127.0.0.1:11435/api/generate -d '{"model":"...","prompt":"..."}'
"""

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import psutil
from aiohttp import ClientSession, ClientTimeout, TCPConnector, web
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------- state

@dataclass
class RequestStats:
    id: int
    model: str
    path: str
    started: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    done: bool = False
    ended: float = 0.0
    error: str = ""
    # timestamps of recently emitted tokens, for instantaneous tok/s
    recent: deque = field(default_factory=lambda: deque(maxlen=200))

    @property
    def elapsed(self) -> float:
        return (self.ended or time.time()) - self.started

    @property
    def avg_tps(self) -> float:
        return self.output_tokens / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def live_tps(self) -> float:
        """tok/s over the last 2 seconds."""
        now = time.time()
        n = sum(1 for t in self.recent if now - t <= 2.0)
        return n / 2.0


class State:
    def __init__(self):
        self.requests: dict[int, RequestStats] = {}
        self.next_id = 1
        self.total_output = 0
        self.total_prompt = 0
        self.total_requests = 0
        self.session_start = time.time()
        # rolling history of session-wide output tok/s for the sparkline
        self.tps_history: deque = deque(maxlen=60)
        self.models = []          # /api/ps result
        self.ollama_version = ""
        self.ollama_ok = False
        self.gpu = {}             # ioreg stats
        self.procs = []           # per-process rows
        self.cpu_percent = 0.0
        self.per_core = []
        self.proxy_error = ""

    def new_request(self, model: str, path: str) -> RequestStats:
        r = RequestStats(self.next_id, model, path, time.time())
        self.next_id += 1
        self.requests[r.id] = r
        self.total_requests += 1
        # keep only the last 8 requests around for display
        for rid in sorted(self.requests)[:-8]:
            del self.requests[rid]
        return r

    def session_live_tps(self) -> float:
        return sum(r.live_tps for r in self.requests.values() if not r.done)


STATE = State()

# ---------------------------------------------------------------- proxy

STREAM_PATHS = ("/api/generate", "/api/chat", "/v1/chat/completions", "/v1/completions")
HOP_HEADERS = {"transfer-encoding", "content-length", "connection", "keep-alive", "host"}


def _extract_model(body: bytes) -> str:
    try:
        return json.loads(body).get("model", "?")
    except Exception:
        return "?"


def _count_ndjson_line(line: bytes, r: RequestStats):
    """One NDJSON line from /api/generate or /api/chat == one token chunk."""
    try:
        obj = json.loads(line)
    except Exception:
        return
    if obj.get("done"):
        r.done = True
        r.ended = time.time()
        # trust ollama's final accounting over our chunk count
        if "eval_count" in obj:
            STATE.total_output += obj["eval_count"] - r.output_tokens
            r.output_tokens = obj["eval_count"]
        if "prompt_eval_count" in obj:
            r.prompt_tokens = obj["prompt_eval_count"]
            STATE.total_prompt += r.prompt_tokens
    else:
        msg = obj.get("message", {})
        # newer ollama builds stream reasoning tokens in a "thinking" field;
        # older builds don't stream them at all (they land in eval_count at done)
        if (obj.get("response") or obj.get("thinking")
                or msg.get("content") or msg.get("thinking")):
            r.output_tokens += 1
            STATE.total_output += 1
            r.recent.append(time.time())


def _count_sse_line(line: bytes, r: RequestStats):
    """One SSE data: line from the OpenAI-compatible endpoints."""
    if not line.startswith(b"data:"):
        return
    payload = line[5:].strip()
    if payload == b"[DONE]":
        r.done = True
        r.ended = time.time()
        return
    try:
        obj = json.loads(payload)
    except Exception:
        return
    usage = obj.get("usage")
    if usage:
        STATE.total_output += usage.get("completion_tokens", r.output_tokens) - r.output_tokens
        r.output_tokens = usage.get("completion_tokens", r.output_tokens)
        r.prompt_tokens = usage.get("prompt_tokens", 0)
        STATE.total_prompt += r.prompt_tokens
    for choice in obj.get("choices", []):
        delta = choice.get("delta", {})
        if (delta.get("content") or delta.get("reasoning_content")
                or delta.get("reasoning") or choice.get("text")):
            r.output_tokens += 1
            STATE.total_output += 1
            r.recent.append(time.time())


def _count_final_json(body: bytes, r: RequestStats):
    """Non-streaming response: read the final token accounting."""
    try:
        obj = json.loads(body)
    except Exception:
        return
    r.done = True
    r.ended = time.time()
    if "eval_count" in obj:                      # ollama native
        r.output_tokens = obj["eval_count"]
        r.prompt_tokens = obj.get("prompt_eval_count", 0)
    elif "usage" in obj:                         # openai-compatible
        r.output_tokens = obj["usage"].get("completion_tokens", 0)
        r.prompt_tokens = obj["usage"].get("prompt_tokens", 0)
    STATE.total_output += r.output_tokens
    STATE.total_prompt += r.prompt_tokens


async def proxy_handler(request: web.Request, upstream: str, session: ClientSession):
    body = await request.read()
    track = request.path in STREAM_PATHS and request.method == "POST"
    r = STATE.new_request(_extract_model(body), request.path) if track else None

    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
    try:
        async with session.request(
            request.method, upstream + request.path_qs, data=body or None, headers=headers
        ) as up:
            ctype = up.headers.get("Content-Type", "")
            streaming = "ndjson" in ctype or "event-stream" in ctype
            resp = web.StreamResponse(status=up.status)
            for k, v in up.headers.items():
                if k.lower() not in HOP_HEADERS:
                    resp.headers[k] = v
            await resp.prepare(request)

            if streaming and r:
                counter = _count_sse_line if "event-stream" in ctype else _count_ndjson_line
                buf = b""
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if line.strip():
                            counter(line.strip(), r)
            else:
                collected = b""
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                    if r:
                        collected += chunk
                if r:
                    _count_final_json(collected, r)

            await resp.write_eof()
            if r and not r.done:
                r.done, r.ended = True, time.time()
            return resp
    except Exception as e:
        if r:
            r.done, r.ended, r.error = True, time.time(), str(e)[:60]
        if isinstance(e, (ConnectionResetError, asyncio.CancelledError)):
            raise
        return web.json_response({"error": f"otop proxy: {e}"}, status=502)


async def start_proxy(host: str, port: int, upstream: str):
    session = ClientSession(
        timeout=ClientTimeout(total=None, sock_connect=5),
        connector=TCPConnector(limit=100),
        auto_decompress=False,
    )
    app = web.Application(client_max_size=512 * 1024 * 1024)
    app.router.add_route(
        "*", "/{tail:.*}", lambda req: proxy_handler(req, upstream, session)
    )
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    try:
        await web.TCPSite(runner, host, port).start()
    except OSError as e:
        STATE.proxy_error = f"proxy failed to bind {host}:{port} ({e.strerror})"
    return runner, session

# ---------------------------------------------------------------- collectors

IOREG_RE = re.compile(
    rb'"PerformanceStatistics"\s*=\s*({.*?})', re.DOTALL
)


async def poll_gpu():
    """Apple Silicon GPU stats from ioreg (no sudo required)."""
    while True:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ioreg", "-r", "-d", "1", "-c", "IOAccelerator",
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            out, _ = await proc.communicate()
            m = IOREG_RE.search(out)
            if m:
                blob = m.group(1).decode("utf-8", "replace")
                def grab(key):
                    mm = re.search(rf'"{re.escape(key)}"=(\d+)', blob)
                    return int(mm.group(1)) if mm else None
                STATE.gpu = {
                    "device": grab("Device Utilization %"),
                    "renderer": grab("Renderer Utilization %"),
                    "tiler": grab("Tiler Utilization %"),
                    "mem_used": grab("In use system memory"),
                    "mem_alloc": grab("Alloc system memory"),
                }
        except Exception:
            STATE.gpu = {}
        await asyncio.sleep(1.0)


async def poll_procs():
    """CPU%, threads, RSS for ollama server + runner processes."""
    tracked: dict[int, psutil.Process] = {}
    while True:
        rows = []
        found = {}
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                name = p.info["name"] or ""
                cmd = " ".join(p.info["cmdline"] or [])
                if "ollama" in name.lower() or "ollama" in cmd.lower():
                    if "otop" in cmd:
                        continue
                    found[p.pid] = tracked.get(p.pid) or p
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        tracked = found
        for pid, p in tracked.items():
            try:
                cmd = " ".join(p.cmdline())
                if "llama-server" in cmd or "runner" in cmd:
                    kind = "runner"      # the process actually executing the model
                elif "serve" in cmd:
                    kind = "server"
                else:
                    kind = "app"         # menubar gui etc.
                rows.append({
                    "pid": pid,
                    "kind": kind,
                    "cpu": p.cpu_percent(interval=None),  # since last call
                    "threads": p.num_threads(),
                    "rss": p.memory_info().rss,
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        STATE.procs = sorted(rows, key=lambda r: -r["cpu"])
        STATE.cpu_percent = psutil.cpu_percent(interval=None)
        STATE.per_core = psutil.cpu_percent(interval=None, percpu=True)
        await asyncio.sleep(1.0)


async def poll_ollama(upstream: str):
    async with ClientSession(timeout=ClientTimeout(total=3)) as s:
        while True:
            try:
                async with s.get(upstream + "/api/ps") as resp:
                    STATE.models = (await resp.json()).get("models", [])
                    STATE.ollama_ok = True
                if not STATE.ollama_version:
                    async with s.get(upstream + "/api/version") as resp:
                        STATE.ollama_version = (await resp.json()).get("version", "?")
            except Exception:
                STATE.ollama_ok = False
                STATE.models = []
            await asyncio.sleep(1.5)


async def poll_tps_history():
    while True:
        STATE.tps_history.append(STATE.session_live_tps())
        await asyncio.sleep(1.0)

# ---------------------------------------------------------------- rendering

def fmt_bytes(n):
    if n is None:
        return "–"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} PB"


def bar(pct, width=18, color="cyan"):
    if pct is None:
        return Text("–")
    pct = max(0.0, min(100.0, pct))
    filled = int(width * pct / 100)
    t = Text()
    t.append("█" * filled, style=color)
    t.append("░" * (width - filled), style="grey35")
    t.append(f" {pct:4.0f}%")
    return t


SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values, width=40):
    vals = list(values)[-width:]
    if not vals:
        return Text("")
    peak = max(max(vals), 1e-9)
    return Text("".join(SPARK[min(7, int(v / peak * 7.999))] for v in vals), style="magenta")


def render_header(proxy_host, proxy_port, no_proxy):
    status = Text()
    if STATE.ollama_ok:
        status.append(" ● ", style="bold green")
        status.append(f"ollama {STATE.ollama_version}  ")
    else:
        status.append(" ● ", style="bold red")
        status.append("ollama unreachable  ")
    if no_proxy:
        status.append("proxy off", style="dim")
    elif STATE.proxy_error:
        status.append(STATE.proxy_error, style="bold red")
    else:
        status.append(f"token proxy → {proxy_host}:{proxy_port}", style="bold yellow")
    up = time.time() - STATE.session_start
    status.append(f"   session {int(up // 60)}m{int(up % 60):02d}s", style="dim")
    return Panel(status, title="[bold]otop – ollama monitor[/]", border_style="blue")


def render_tokens():
    t = Table.grid(expand=True)
    t.add_column(justify="left")
    live = STATE.session_live_tps()
    summary = Text()
    summary.append(f"{STATE.total_output:,}", style="bold magenta")
    summary.append(" output tok   ")
    summary.append(f"{STATE.total_prompt:,}", style="bold cyan")
    summary.append(" prompt tok   ")
    summary.append(f"{STATE.total_requests}", style="bold")
    summary.append(" requests   ")
    summary.append(f"{live:5.1f} tok/s now", style="bold green" if live else "dim")
    t.add_row(summary)
    t.add_row(sparkline(STATE.tps_history))

    reqs = Table(box=None, pad_edge=False, expand=True)
    for col, j in (("#", "right"), ("model", "left"), ("endpoint", "left"),
                   ("out tok", "right"), ("prompt", "right"),
                   ("tok/s", "right"), ("time", "right"), ("", "left")):
        reqs.add_column(col, justify=j, style="dim" if col == "#" else "")
    for r in sorted(STATE.requests.values(), key=lambda x: -x.id)[:6]:
        tps = r.avg_tps if r.done else r.live_tps
        if r.error:
            flag = Text("✗ " + r.error, style="red")
        elif r.done:
            flag = Text("done", style="dim")
        elif r.output_tokens == 0 and r.elapsed > 2.0:
            # active but nothing streamed yet: model load, prompt eval, or a
            # reasoning model thinking on a server that doesn't stream thinking
            flag = Text("● thinking/loading", style="bold yellow")
        else:
            flag = Text("● generating", style="bold green")
        reqs.add_row(
            str(r.id), r.model, r.path.replace("/api/", "").replace("/v1/", "v1/"),
            f"{r.output_tokens:,}", f"{r.prompt_tokens:,}" if r.prompt_tokens else "–",
            f"{tps:.1f}", f"{r.elapsed:5.1f}s", flag,
        )
    if not STATE.requests:
        reqs.add_row("", Text("no requests through proxy yet", style="dim italic"),
                     "", "", "", "", "", "")
    return Panel(Group(t, reqs), title="[bold magenta]tokens[/]", border_style="magenta")


def render_system():
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(width=10)
    grid.add_column()
    grid.add_row("cpu total", bar(STATE.cpu_percent, color="cyan"))
    cores = Text()
    for i, c in enumerate(STATE.per_core):
        style = "red" if c > 80 else "yellow" if c > 40 else "green"
        cores.append(SPARK[min(7, int(c / 100 * 7.999))], style=style)
        if i % 2:
            cores.append(" ")
    grid.add_row(f"{len(STATE.per_core)} cores", cores)
    la1, la5, la15 = psutil.getloadavg()
    grid.add_row("load", Text(f"{la1:.2f} {la5:.2f} {la15:.2f}", style="dim"))
    vm = psutil.virtual_memory()
    grid.add_row("ram", bar(vm.percent, color="blue"))
    grid.add_row("", Text(f"{fmt_bytes(vm.used)} / {fmt_bytes(vm.total)}", style="dim"))

    procs = Table(box=None, pad_edge=False, expand=True)
    for col, j in (("pid", "right"), ("proc", "left"), ("cpu%", "right"),
                   ("threads", "right"), ("rss", "right")):
        procs.add_column(col, justify=j)
    for p in STATE.procs[:4]:
        style = "bold" if p["kind"] == "runner" else ""
        procs.add_row(
            Text(str(p["pid"]), style="dim"),
            Text(f"ollama {p['kind']}", style=style),
            Text(f"{p['cpu']:.0f}", style="yellow" if p["cpu"] > 100 else ""),
            str(p["threads"]),
            fmt_bytes(p["rss"]),
        )
    if not STATE.procs:
        procs.add_row("", Text("no ollama processes found", style="dim italic"), "", "", "")
    return Panel(Group(grid, Text(), procs), title="[bold cyan]cpu / system[/]",
                 border_style="cyan")


def render_gpu():
    g = STATE.gpu
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(width=10)
    grid.add_column()
    grid.add_row("device", bar(g.get("device"), color="green"))
    grid.add_row("renderer", bar(g.get("renderer"), color="green"))
    grid.add_row("tiler", bar(g.get("tiler"), color="green"))
    used, alloc = g.get("mem_used"), g.get("mem_alloc")
    grid.add_row("gpu mem", Text(f"{fmt_bytes(used)} in use / {fmt_bytes(alloc)} wired",
                                 style="dim"))
    grid.add_row("", Text("unified memory – shared with system ram", style="dim italic"))
    return Panel(grid, title="[bold green]gpu (apple silicon)[/]", border_style="green")


def render_models():
    tbl = Table(box=None, pad_edge=False, expand=True)
    for col, j in (("model", "left"), ("size", "right"), ("vram", "right"),
                   ("placement", "left"), ("ctx", "right"), ("expires", "right")):
        tbl.add_column(col, justify=j)
    now = time.time()
    for m in STATE.models:
        size, vram = m.get("size", 0), m.get("size_vram", 0)
        gpu_pct = round(vram / size * 100) if size else 0
        placement = f"{gpu_pct}% GPU" if gpu_pct else "100% CPU"
        expires = "–"
        try:
            from datetime import datetime
            # ollama emits rfc3339 with nanoseconds; trim to µs for fromisoformat
            raw = re.sub(r"(\.\d{6})\d+", r"\1", m["expires_at"])
            exp = datetime.fromisoformat(raw).timestamp()
            secs = max(0, exp - now)
            expires = "never" if secs > 3e8 else f"{int(secs // 60)}m{int(secs % 60):02d}s"
        except Exception:
            pass
        tbl.add_row(
            Text(m.get("name", "?"), style="bold"),
            fmt_bytes(size), fmt_bytes(vram),
            Text(placement, style="green" if gpu_pct >= 100 else "yellow"),
            str(m.get("context_length", "–")),
            Text(expires, style="dim"),
        )
    if not STATE.models:
        tbl.add_row(Text("no models loaded", style="dim italic"), "", "", "", "", "")
    return Panel(tbl, title="[bold blue]loaded models[/]", border_style="blue")


def render(proxy_host, proxy_port, no_proxy):
    layout = Layout()
    layout.split_column(
        Layout(render_header(proxy_host, proxy_port, no_proxy), size=3),
        Layout(render_tokens(), size=12),
        Layout(name="mid"),
        Layout(render_models(), size=6),
    )
    layout["mid"].split_row(Layout(render_system()), Layout(render_gpu()))
    return layout

# ---------------------------------------------------------------- main

async def main():
    ap = argparse.ArgumentParser(description="live ollama monitor")
    ap.add_argument("--upstream", default="http://127.0.0.1:11434",
                    help="ollama server url (default: %(default)s)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address the proxy listens on; use 0.0.0.0 for all "
                         "interfaces (default: %(default)s)")
    ap.add_argument("--proxy-port", type=int, default=11435,
                    help="token-counting proxy port (default: %(default)s)")
    ap.add_argument("--no-proxy", action="store_true", help="disable the proxy")
    ap.add_argument("--refresh", type=float, default=4.0,
                    help="ui refreshes per second (default: %(default)s)")
    args = ap.parse_args()

    tasks = [
        asyncio.create_task(poll_gpu()),
        asyncio.create_task(poll_procs()),
        asyncio.create_task(poll_ollama(args.upstream)),
        asyncio.create_task(poll_tps_history()),
    ]
    runner = session = None
    if not args.no_proxy:
        runner, session = await start_proxy(args.host, args.proxy_port, args.upstream)

    console = Console()
    try:
        with Live(render(args.host, args.proxy_port, args.no_proxy), console=console,
                  screen=True, refresh_per_second=args.refresh) as live:
            while True:
                await asyncio.sleep(1.0 / args.refresh)
                live.update(render(args.host, args.proxy_port, args.no_proxy))
    finally:
        for t in tasks:
            t.cancel()
        if session:
            await session.close()
        if runner:
            await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
