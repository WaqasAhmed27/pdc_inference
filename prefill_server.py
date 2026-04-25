"""
PDC Inference Server — Phase 3 (KV Prefill)
Extends the Phase 1 server with editor-aware KV-cache prefetching.

New endpoints vs Phase 1:
  POST /context          — editor sends context on every keystroke
  POST /complete_prefill — autocomplete with prefill hit/miss tracking
  POST /rewrite_prefill  — rewrite with prefill hit/miss tracking
  GET  /prefill/stats    — global prefill statistics
  GET  /prefill/stats/{session_id} — per-session stats

Run with prefill enabled (default):
    MODEL_PATH=models/mistral-7b-instruct-v0.2.Q8_0.gguf \
    uvicorn prefill_server:app --host 0.0.0.0 --port 8001

Run with prefill disabled (for A/B baseline):
    PREFILL_ENABLED=0 \
    MODEL_PATH=models/mistral-7b-instruct-v0.2.Q8_0.gguf \
    uvicorn prefill_server:app --host 0.0.0.0 --port 8001
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from prefill_manager import PrefillManager

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_GPU = True
except Exception:
    HAS_GPU = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH      = os.environ.get("MODEL_PATH",      "models/mistral-7b-instruct-v0.2.Q8_0.gguf")
N_GPU_LAYERS    = int(os.environ.get("N_GPU_LAYERS",    "-1"))
N_CTX           = int(os.environ.get("N_CTX",           "4096"))
N_THREADS       = int(os.environ.get("N_THREADS",       "4"))
PREFILL_ENABLED = os.environ.get("PREFILL_ENABLED", "1") == "1"
PREFILL_TOKENS  = int(os.environ.get("PREFILL_TOKENS",  "32"))
PREFILL_IDLE_MS = float(os.environ.get("PREFILL_IDLE_MS", "150"))

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ContextUpdate(BaseModel):
    session_id: str
    text: str

class PrefillCompleteRequest(BaseModel):
    session_id: str
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.2
    stream: bool = True

class PrefillRewriteRequest(BaseModel):
    session_id: str
    text: str
    instruction: str = "Improve clarity and conciseness."
    max_tokens: int = 128
    temperature: float = 0.3
    stream: bool = True

# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
def get_memory_stats() -> dict:
    vm = psutil.virtual_memory()
    stats = {
        "ram_used_mb":  round(vm.used  / 1024**2, 1),
        "ram_total_mb": round(vm.total / 1024**2, 1),
        "ram_percent":  vm.percent,
    }
    if HAS_GPU:
        info = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
        stats["vram_used_mb"]  = round(info.used  / 1024**2, 1)
        stats["vram_total_mb"] = round(info.total / 1024**2, 1)
        stats["vram_percent"]  = round(info.used  / info.total * 100, 1)
    return stats

# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------
llm = None
prefill_manager: PrefillManager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, prefill_manager
    from llama_cpp import Llama

    log.info(f"Loading model: {MODEL_PATH}")
    log.info(f"  n_gpu_layers={N_GPU_LAYERS}  n_ctx={N_CTX}")
    log.info(f"  prefill_enabled={PREFILL_ENABLED}  n_tokens={PREFILL_TOKENS}  idle_ms={PREFILL_IDLE_MS}")

    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        use_mmap=True,
        verbose=False,
    )
    prefill_manager = PrefillManager(
        llm,
        n_tokens=PREFILL_TOKENS,
        idle_ms=PREFILL_IDLE_MS,
        enabled=PREFILL_ENABLED,
    )
    log.info("Server ready.")
    yield
    llm = None
    prefill_manager = None

app = FastAPI(
    title="PDC Inference Server — Phase 3 (KV Prefill)",
    version="0.3.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Streaming generator (prefill-aware)
# ---------------------------------------------------------------------------
async def _stream_tokens_prefill(
    session_id: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    endpoint: str,
):
    cache_hit, prefill_overhead_ms = prefill_manager.maybe_prefill(session_id, prompt)

    request_start = time.perf_counter()
    first_token = True
    ttft_ms = 0.0
    token_count = 0
    mem_before = get_memory_stats()

    for chunk in llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
        echo=False,
    ):
        token_text = chunk["choices"][0]["text"]
        if not token_text:
            continue
        now = time.perf_counter()
        if first_token:
            ttft_ms = (now - request_start) * 1000
            first_token = False
        token_count += 1
        yield f"data: {token_text}\n\n"

    e2e_ms  = (time.perf_counter() - request_start) * 1000
    tpot_ms = (e2e_ms - ttft_ms) / max(token_count - 1, 1) if token_count > 1 else 0.0
    mem_after = get_memory_stats()

    log.info(
        f"[{endpoint}] session={session_id[:8]} "
        f"prefill={'HIT' if cache_hit else 'MISS'} | "
        f"TTFT={ttft_ms:.1f}ms | E2E={e2e_ms:.1f}ms | "
        f"tokens={token_count} | "
        f"VRAM={mem_after.get('vram_used_mb', 'N/A')}MB"
    )

    stats = {
        "ttft_ms":              round(ttft_ms, 2),
        "e2e_ms":               round(e2e_ms, 2),
        "tokens":               token_count,
        "tpot_ms":              round(tpot_ms, 2),
        "cache_hit":            cache_hit,
        "prefill_overhead_ms":  round(prefill_overhead_ms, 2),
        "prefill_enabled":      PREFILL_ENABLED,
        "mem_before":           mem_before,
        "mem_after":            mem_after,
    }
    yield f"event: stats\ndata: {json.dumps(stats)}\n\n"
    yield "data: [DONE]\n\n"

def _sse_headers():
    return {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/context", summary="Editor context update — fires idle-timer prefill")
async def update_context(req: ContextUpdate):
    """
    The editor frontend calls this on every keystroke (or debounced at ~100ms).
    Triggers the idle-timer prefill after PREFILL_IDLE_MS of silence.
    """
    if prefill_manager is None:
        raise HTTPException(503, "Server not ready")
    prefill_manager.update_context(req.session_id, req.text)
    return {"status": "ok", "prefill_enabled": PREFILL_ENABLED}


@app.post("/complete_prefill", summary="Autocomplete with prefill tracking")
async def complete_prefill(req: PrefillCompleteRequest):
    if llm is None:
        raise HTTPException(503, "Model not loaded")
    prompt = f"[INST] Continue this text naturally without repeating it:\n{req.prompt} [/INST]"
    return StreamingResponse(
        _stream_tokens_prefill(req.session_id, prompt, req.max_tokens, req.temperature, "complete"),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


@app.post("/rewrite_prefill", summary="Rewrite with prefill tracking")
async def rewrite_prefill(req: PrefillRewriteRequest):
    if llm is None:
        raise HTTPException(503, "Model not loaded")
    prompt = (
        f"[INST] {req.instruction}\n\n"
        f"Original text:\n{req.text}\n\n"
        f"Rewritten text: [/INST]"
    )
    return StreamingResponse(
        _stream_tokens_prefill(req.session_id, prompt, req.max_tokens, req.temperature, "rewrite"),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


@app.delete("/session/{session_id}", summary="Flush a session (tab close)")
async def flush_session(session_id: str):
    if prefill_manager:
        prefill_manager.flush_session(session_id)
    return {"status": "flushed", "session_id": session_id}


@app.get("/prefill/stats", summary="Global prefill statistics")
async def prefill_stats_global():
    if prefill_manager is None:
        raise HTTPException(503, "Server not ready")
    return {
        "prefill": prefill_manager.get_global_stats(),
        "memory":  get_memory_stats(),
    }


@app.get("/prefill/stats/{session_id}", summary="Per-session prefill statistics")
async def prefill_stats_session(session_id: str):
    if prefill_manager is None:
        raise HTTPException(503, "Server not ready")
    stats = prefill_manager.get_session_stats(session_id)
    if stats is None:
        raise HTTPException(404, f"Session {session_id!r} not found")
    return stats


@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "model_loaded":    llm is not None,
        "prefill_enabled": PREFILL_ENABLED,
        "prefill_tokens":  PREFILL_TOKENS,
        "prefill_idle_ms": PREFILL_IDLE_MS,
        "memory":          get_memory_stats(),
    }
