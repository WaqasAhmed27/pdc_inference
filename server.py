"""
PDC Inference Server — Phase 1
FastAPI streaming server backed by llama.cpp.
Logs TTFT, end-to-end latency, TPOT, and VRAM usage per request.

Usage:
    MODEL_PATH=models/mistral-7b-instruct-v0.2.Q8_0.gguf uvicorn server:app --host 0.0.0.0 --port 8000
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
from pydantic import BaseModel

# Optional GPU memory tracking via pynvml
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
# Config — override via environment variables
# ---------------------------------------------------------------------------
MODEL_PATH    = os.environ.get("MODEL_PATH",    "models/mistral-7b-instruct-v0.2.Q8_0.gguf")
N_GPU_LAYERS  = int(os.environ.get("N_GPU_LAYERS",  "-1"))   # -1 = full GPU offload
N_CTX         = int(os.environ.get("N_CTX",         "4096"))
N_THREADS     = int(os.environ.get("N_THREADS",     "4"))

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.2
    stream: bool = True

class RewriteRequest(BaseModel):
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
        "ram_used_mb":  round(vm.used   / 1024**2, 1),
        "ram_total_mb": round(vm.total  / 1024**2, 1),
        "ram_percent":  vm.percent,
    }
    if HAS_GPU:
        info = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
        stats["vram_used_mb"]  = round(info.used  / 1024**2, 1)
        stats["vram_total_mb"] = round(info.total / 1024**2, 1)
        stats["vram_percent"]  = round(info.used  / info.total * 100, 1)
    return stats

# ---------------------------------------------------------------------------
# App lifespan — load model once at startup
# ---------------------------------------------------------------------------
llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    from llama_cpp import Llama

    log.info(f"Loading model: {MODEL_PATH}")
    log.info(f"  n_gpu_layers={N_GPU_LAYERS}  n_ctx={N_CTX}  n_threads={N_THREADS}")
    mem_before = get_memory_stats()

    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        use_mmap=True,
        verbose=False,
    )

    mem_after = get_memory_stats()
    vram_delta = mem_after.get("vram_used_mb", 0) - mem_before.get("vram_used_mb", 0)
    log.info(f"Model loaded. VRAM delta: +{vram_delta:.0f} MB")
    yield
    llm = None

app = FastAPI(title="PDC Inference Server — Phase 1", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Core streaming generator
# ---------------------------------------------------------------------------
async def _stream_tokens(prompt: str, max_tokens: int, temperature: float):
    """
    Streams SSE token events.
    Emits a final 'stats' event with TTFT, e2e latency, TPOT, and memory usage.
    """
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

    e2e_ms = (time.perf_counter() - request_start) * 1000
    # time-per-output-token (excludes prefill)
    tpot_ms = (e2e_ms - ttft_ms) / max(token_count - 1, 1) if token_count > 1 else 0.0
    mem_after = get_memory_stats()

    log.info(
        f"TTFT={ttft_ms:.1f}ms | E2E={e2e_ms:.1f}ms | "
        f"tokens={token_count} | TPOT={tpot_ms:.1f}ms/tok | "
        f"VRAM={mem_after.get('vram_used_mb', 'N/A')}MB"
    )

    stats = {
        "ttft_ms":    round(ttft_ms,  2),
        "e2e_ms":     round(e2e_ms,   2),
        "tokens":     token_count,
        "tpot_ms":    round(tpot_ms,  2),
        "mem_before": mem_before,
        "mem_after":  mem_after,
    }
    yield f"event: stats\ndata: {json.dumps(stats)}\n\n"
    yield "data: [DONE]\n\n"

def _sse_headers():
    return {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/complete", summary="Autocomplete — target TTFT < 200ms")
async def complete(req: CompletionRequest):
    if llm is None:
        raise HTTPException(503, "Model not loaded")

    prompt = f"[INST] Continue this text naturally without repeating it:\n{req.prompt} [/INST]"

    if req.stream:
        return StreamingResponse(
            _stream_tokens(prompt, req.max_tokens, req.temperature),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )

    start = time.perf_counter()
    result = llm(prompt, max_tokens=req.max_tokens, temperature=req.temperature, echo=False)
    e2e_ms = (time.perf_counter() - start) * 1000
    return {"text": result["choices"][0]["text"], "e2e_ms": round(e2e_ms, 2)}


@app.post("/rewrite", summary="Short rewrite — target TTFT < 400ms")
async def rewrite(req: RewriteRequest):
    if llm is None:
        raise HTTPException(503, "Model not loaded")

    prompt = (
        f"[INST] {req.instruction}\n\n"
        f"Original text:\n{req.text}\n\n"
        f"Rewritten text: [/INST]"
    )

    if req.stream:
        return StreamingResponse(
            _stream_tokens(prompt, req.max_tokens, req.temperature),
            media_type="text/event-stream",
            headers=_sse_headers(),
        )

    start = time.perf_counter()
    result = llm(prompt, max_tokens=req.max_tokens, temperature=req.temperature, echo=False)
    e2e_ms = (time.perf_counter() - start) * 1000
    return {"text": result["choices"][0]["text"], "e2e_ms": round(e2e_ms, 2)}


@app.get("/health", summary="Health check + live memory stats")
async def health():
    return {
        "status": "ok",
        "model_loaded": llm is not None,
        "model_path": MODEL_PATH,
        "n_gpu_layers": N_GPU_LAYERS,
        "n_ctx": N_CTX,
        "memory": get_memory_stats(),
    }
