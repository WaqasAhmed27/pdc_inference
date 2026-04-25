"""
PDC Inference Server — Phase 4
Extends Phase 3 with speculative decoding and configurable KV eviction policies.

New vs Phase 3:
  - Speculative decoding via TinyLlama 1.1B draft model (configurable block size k)
  - KV eviction: LRU (default) or adaptive (via EVICTION_POLICY env var)
  - POST /complete_spec  — speculative autocomplete
  - POST /rewrite_spec   — speculative rewrite
  - GET  /spec/stats     — acceptance rate, speedup ratio, draft/target call counts
  - Existing /context + /complete_prefill + /prefill/stats from Phase 3 retained

Run:
    # Speculative + LRU eviction (default)
    MODEL_PATH=models/mistral-7b-instruct-v0.2.Q8_0.gguf \
    DRAFT_MODEL_PATH=models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    SPEC_K=4 \
    EVICTION_POLICY=lru \
    uvicorn phase4_server:app --host 0.0.0.0 --port 8002

    # Adaptive eviction
    EVICTION_POLICY=adaptive \
    uvicorn phase4_server:app --host 0.0.0.0 --port 8002

    # No speculative decoding (for ablation — uses only target model)
    SPEC_ENABLED=0 \
    uvicorn phase4_server:app --host 0.0.0.0 --port 8002
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from kv_eviction import make_policy, EvictionPolicy
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
MODEL_PATH       = os.environ.get("MODEL_PATH",       "models/mistral-7b-instruct-v0.2.Q8_0.gguf")
DRAFT_MODEL_PATH = os.environ.get("DRAFT_MODEL_PATH", "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
N_GPU_LAYERS     = int(os.environ.get("N_GPU_LAYERS",     "-1"))
N_CTX            = int(os.environ.get("N_CTX",            "2048"))
N_THREADS        = int(os.environ.get("N_THREADS",        "4"))
SPEC_ENABLED     = os.environ.get("SPEC_ENABLED",     "1") == "1"
SPEC_K           = int(os.environ.get("SPEC_K",           "4"))
EVICTION_POLICY  = os.environ.get("EVICTION_POLICY",  "lru")     # "lru" | "adaptive"
EVICTION_MAX_CTX = int(os.environ.get("EVICTION_MAX_CTX", "1024"))
PREFILL_ENABLED  = os.environ.get("PREFILL_ENABLED",  "1") == "1"
PREFILL_TOKENS   = int(os.environ.get("PREFILL_TOKENS",   "32"))
PREFILL_IDLE_MS  = float(os.environ.get("PREFILL_IDLE_MS", "150"))

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SpecRequest(BaseModel):
    session_id: str = "default"
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.2

class SpecRewriteRequest(BaseModel):
    session_id: str = "default"
    text: str
    instruction: str = "Improve clarity and conciseness."
    max_tokens: int = 128
    temperature: float = 0.3

class ContextUpdate(BaseModel):
    session_id: str
    text: str

# ---------------------------------------------------------------------------
# Helpers
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
# App state
# ---------------------------------------------------------------------------

llm              = None   # target model (Mistral 7B)
draft_llm        = None   # draft model (TinyLlama 1.1B)
spec_engine      = None
eviction_policy: Optional[EvictionPolicy] = None
prefill_manager: Optional[PrefillManager] = None

# Cumulative speculative decoding stats
_spec_stats = {
    "total_generated":  0,
    "total_accepted":   0,
    "total_rejected":   0,
    "total_requests":   0,
    "total_target_calls": 0,
    "total_draft_calls":  0,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, draft_llm, spec_engine, eviction_policy, prefill_manager
    from llama_cpp import Llama

    log.info(f"Loading target model: {MODEL_PATH}")
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        logits_all=SPEC_ENABLED,
        use_mmap=True,
        verbose=False,
    )

    if SPEC_ENABLED:
        log.info(f"Loading draft model (k={SPEC_K}): {DRAFT_MODEL_PATH}")
        from speculative_engine import SpeculativeEngine
        spec_engine = SpeculativeEngine(
            draft_model_path=DRAFT_MODEL_PATH,
            target_model_path=MODEL_PATH,
            k=SPEC_K,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            verbose=False,
        )
        log.info("Speculative engine ready.")
    else:
        log.info("Speculative decoding disabled.")

    log.info(f"Eviction policy: {EVICTION_POLICY.upper()}  max_ctx={EVICTION_MAX_CTX}")
    eviction_policy = make_policy(
        EVICTION_POLICY,
        max_ctx=EVICTION_MAX_CTX,
        sink_tokens=4,
        recent_tokens=128,
    )

    prefill_manager = PrefillManager(
        llm,
        n_tokens=PREFILL_TOKENS,
        idle_ms=PREFILL_IDLE_MS,
        enabled=PREFILL_ENABLED,
    )

    log.info("Phase 4 server ready.")
    yield
    llm = draft_llm = spec_engine = eviction_policy = prefill_manager = None


app = FastAPI(
    title="PDC Inference Server — Phase 4 (Speculative + Eviction)",
    version="0.4.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# Speculative endpoints
# ---------------------------------------------------------------------------

@app.post("/complete_spec", summary="Speculative autocomplete (k draft tokens per step)")
async def complete_spec(req: SpecRequest):
    if llm is None:
        raise HTTPException(503, "Model not loaded")

    prompt = f"[INST] Continue this text naturally without repeating it:\n{req.prompt} [/INST]"

    # Apply eviction policy to trim long prompts
    if eviction_policy:
        prompt_tokens = llm.tokenize(prompt.encode())
        trimmed_tokens = eviction_policy.trim(prompt_tokens)
        if len(trimmed_tokens) < len(prompt_tokens):
            prompt = llm.detokenize(trimmed_tokens).decode(errors="replace")
            log.debug(f"Prompt trimmed: {len(prompt_tokens)} → {len(trimmed_tokens)} tokens")

    mem_before = get_memory_stats()
    t_start = time.perf_counter()

    if SPEC_ENABLED and spec_engine:
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: spec_engine.generate(prompt, req.max_tokens, req.temperature),
        )

        if result.ok:
            _update_spec_stats(result)
            log.info(
                f"[spec/complete] k={SPEC_K} | "
                f"TTFT={result.ttft_ms:.1f}ms | E2E={result.e2e_ms:.1f}ms | "
                f"accept_rate={result.acceptance_rate:.2f} | "
                f"speedup={result.speedup_ratio:.2f}x | "
                f"tokens={result.tokens_generated}"
            )
            return {
                "text":            result.text,
                "ttft_ms":         round(result.ttft_ms, 2),
                "e2e_ms":          round(result.e2e_ms, 2),
                "tpot_ms":         round(result.tpot_ms, 2),
                "tokens":          result.tokens_generated,
                "acceptance_rate": round(result.acceptance_rate, 3),
                "speedup_ratio":   round(result.speedup_ratio, 3),
                "spec_k":          SPEC_K,
                "eviction_policy": EVICTION_POLICY,
                "mem":             get_memory_stats(),
            }
        else:
            raise HTTPException(500, result.error)

    else:
        # Fallback: standard autoregressive (for ablation)
        result = llm(prompt, max_tokens=req.max_tokens, temperature=req.temperature, echo=False)
        e2e_ms = (time.perf_counter() - t_start) * 1000
        return {
            "text":   result["choices"][0]["text"],
            "e2e_ms": round(e2e_ms, 2),
            "spec_enabled": False,
        }


@app.post("/rewrite_spec", summary="Speculative rewrite")
async def rewrite_spec(req: SpecRewriteRequest):
    if llm is None:
        raise HTTPException(503, "Model not loaded")

    prompt = (
        f"[INST] {req.instruction}\n\n"
        f"Original text:\n{req.text}\n\n"
        f"Rewritten text: [/INST]"
    )

    if eviction_policy:
        prompt_tokens = llm.tokenize(prompt.encode())
        trimmed_tokens = eviction_policy.trim(prompt_tokens)
        if len(trimmed_tokens) < len(prompt_tokens):
            prompt = llm.detokenize(trimmed_tokens).decode(errors="replace")

    import asyncio
    loop = asyncio.get_event_loop()

    if SPEC_ENABLED and spec_engine:
        result = await loop.run_in_executor(
            None,
            lambda: spec_engine.generate(prompt, req.max_tokens, req.temperature),
        )
        if not result.ok:
            raise HTTPException(500, result.error)
        _update_spec_stats(result)
        return {
            "text":            result.text,
            "ttft_ms":         round(result.ttft_ms, 2),
            "e2e_ms":          round(result.e2e_ms, 2),
            "acceptance_rate": round(result.acceptance_rate, 3),
            "speedup_ratio":   round(result.speedup_ratio, 3),
            "eviction_policy": EVICTION_POLICY,
        }
    else:
        t_start = time.perf_counter()
        result = await loop.run_in_executor(
            None,
            lambda: llm(prompt, max_tokens=req.max_tokens, temperature=req.temperature, echo=False),
        )
        e2e_ms = (time.perf_counter() - t_start) * 1000
        return {"text": result["choices"][0]["text"], "e2e_ms": round(e2e_ms, 2)}


# ---------------------------------------------------------------------------
# Prefill endpoints (carried over from Phase 3)
# ---------------------------------------------------------------------------

@app.post("/context")
async def update_context(req: ContextUpdate):
    if prefill_manager:
        prefill_manager.update_context(req.session_id, req.text)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Stats endpoints
# ---------------------------------------------------------------------------

@app.get("/spec/stats", summary="Speculative decoding global stats")
async def spec_stats():
    total = _spec_stats["total_accepted"] + _spec_stats["total_rejected"]
    return {
        "spec_enabled":    SPEC_ENABLED,
        "k":               SPEC_K,
        "eviction_policy": EVICTION_POLICY,
        **_spec_stats,
        "overall_acceptance_rate": round(_spec_stats["total_accepted"] / max(total, 1), 3),
        "avg_speedup_ratio": round(
            _spec_stats["total_generated"] / max(_spec_stats["total_target_calls"], 1), 3
        ),
        "eviction_stats":  eviction_policy.stats if eviction_policy else {},
        "memory":          get_memory_stats(),
    }


@app.get("/prefill/stats")
async def prefill_stats():
    if prefill_manager is None:
        raise HTTPException(503, "Server not ready")
    return {"prefill": prefill_manager.get_global_stats(), "memory": get_memory_stats()}


@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "model_loaded":    llm is not None,
        "spec_enabled":    SPEC_ENABLED,
        "spec_k":          SPEC_K,
        "eviction_policy": EVICTION_POLICY,
        "prefill_enabled": PREFILL_ENABLED,
        "memory":          get_memory_stats(),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _update_spec_stats(result):
    _spec_stats["total_generated"]    += result.tokens_generated
    _spec_stats["total_accepted"]     += result.tokens_accepted
    _spec_stats["total_rejected"]     += result.tokens_rejected
    _spec_stats["total_requests"]     += 1
    _spec_stats["total_target_calls"] += result.target_calls
    _spec_stats["total_draft_calls"]  += result.draft_calls
