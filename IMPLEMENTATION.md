# Implementation Summary
## Real-Time AI Text Editing via Low-Latency and Memory-Efficient Transformer Inference
### Waqas Ahmed (i232540) · Huzaifa Khalid (i232508)

---

## What Has Been Implemented (Phases 1–3)

---

### Phase 1 — Infrastructure (`server.py`)

A production-ready FastAPI inference server backed by llama.cpp.

**What it does:**
- Loads Mistral 7B (GGUF / INT8 quantised) once at startup via `llama_cpp.Llama`
- Full GPU offload via `n_gpu_layers=-1`, configurable via environment variables
- Exposes two streaming SSE endpoints:
  - `POST /complete` — cursor autocomplete, target TTFT < 200ms
  - `POST /rewrite`  — short rewrite/grammar correction, target TTFT < 400ms
- Each SSE stream emits a final `stats` event containing:
  - `ttft_ms` — time to first token (ms)
  - `e2e_ms`  — end-to-end latency (ms)
  - `tpot_ms` — time per output token (ms)
  - `mem_before` / `mem_after` — RAM + VRAM before and after the request
- `GET /health` — live server status + memory snapshot

**Technical decisions:**
- `use_mmap=True` — model weights are memory-mapped, avoiding a full load into RAM
- Non-streaming fallback for both endpoints (set `stream: false` in the request body)
- CUDA VRAM tracked via `pynvml`; graceful fallback if no NVIDIA driver present

---

### Phase 2 — Baseline Benchmarking (`benchmark.py`, `workload_gen.py`)

An async benchmarking harness and synthetic workload generator.

**`workload_gen.py` — three workload types:**

| Workload | Generator | Description |
|---|---|---|
| Autocomplete | `generate_autocomplete_workload(n)` | 30 mid-sentence prompts (NLP/systems/Wikipedia topics), max_tokens ∈ {32, 48, 64} |
| Rewrite | `generate_rewrite_workload(n)` | 15 informal sentences with randomised rewrite instructions |
| Revision history | `generate_revision_history_workload(n)` | Single base document sliced at n evenly-spaced offsets — simulates incrementally growing context |
| Mixed | `generate_mixed_workload(n)` | Randomly interleaved autocomplete + rewrite — for concurrent-session tests |

**`benchmark.py` — what it measures:**
- TTFT mean, p50, p95, min, max
- E2E latency mean, p50
- TPOT (time per output token) mean, p50
- Target compliance: % requests meeting < 200ms (autocomplete) and < 400ms (rewrite) TTFT targets
- All results saved to `results/*.csv` for paper tables

**Concurrency support:** `--concurrency N` sends N simultaneous requests via `asyncio.Semaphore`, simulating multi-user load.

**Warmup:** 2 requests are sent and discarded before timing begins to avoid cold-model bias.

---

### Phase 3 — Editor-Aware KV Prefill (`prefill_manager.py`, `prefill_server.py`, `experiment_prefill.py`)

The core technical contribution of this project.

#### `prefill_manager.py` — PrefillManager

Implements a non-invasive editor-aware KV-cache prefetch with no modifications to decoding internals.

**Algorithm:**
1. Editor calls `update_context(session_id, text)` on every keystroke
2. Each call resets a `call_later` idle timer (default: 150ms)
3. After 150ms of silence the timer fires, running a 1-token completion on the last 32 tokens of context
4. This forces llama.cpp to compute and cache the KV state for that context
5. On the real `/complete` request, `maybe_prefill()` checks if the prompt shares a prefix with the warm cache
6. Cache hit → TTFT reduced (prefill already done); cache miss → cold inference as normal

**Key implementation details:**
- Blocking `llm()` call is offloaded to `asyncio.run_in_executor` — does not block the event loop
- Per-session state via `SessionState` dataclass (supports concurrent editor tabs)
- Thread-safe via `threading.Lock` for all shared state mutations
- Timer cancellation on new keystrokes prevents redundant prefills during active typing
- Exposes `hit_rate`, `avg_prefill_ms`, `n_hits`, `n_misses` per session and globally

#### `prefill_server.py` — Phase 3 Server

Extends the Phase 1 server with prefill-aware endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /context` | Receives keystroke context updates; triggers idle timer |
| `POST /complete_prefill` | Autocomplete with cache hit/miss tracking in stats event |
| `POST /rewrite_prefill` | Rewrite with cache hit/miss tracking |
| `GET  /prefill/stats` | Global hit rate, prefill count, overhead |
| `GET  /prefill/stats/{id}` | Per-session stats |
| `DELETE /session/{id}` | Flush session state (tab close) |

Runs on port **8001** to allow Phase 1 and Phase 3 servers to run side-by-side for comparison.

`PREFILL_ENABLED=0` env var disables prefill globally — used for the cold baseline in the experiment.

#### `experiment_prefill.py` — Phase 3 Experiment

Automated A/B experiment that produces the core Phase 3 result:

- **Pass A (cold):** idle_ms=0 — prefill timer never fires before the request
- **Pass B (warm):** idle_ms=300ms — prefill always fires 150ms before the request
- Records TTFT, E2E, cache hit rate for both passes
- Saves `results/phase3_prefill_cold.csv`, `results/phase3_prefill_warm.csv`
- Prints and saves `results/phase3_summary.txt` — a formatted comparison table ready for the paper

---

## File Map

```
pdc_inference/
│
├── server.py              Phase 1 — FastAPI inference server (port 8000)
├── workload_gen.py        Phase 2 — Synthetic editor workload generator
├── benchmark.py           Phase 2 — Async benchmark harness (CSV output)
│
├── prefill_manager.py     Phase 3 — KV prefill logic (PrefillManager class)
├── prefill_server.py      Phase 3 — Prefill-aware FastAPI server (port 8001)
├── experiment_prefill.py  Phase 3 — Cold vs warm A/B experiment
│
├── requirements.txt       Python dependencies
├── QUICKSTART.md          Step-by-step run instructions
└── IMPLEMENTATION.md      This file
```

---

## Metrics Collected

| Metric | Unit | Collected in |
|---|---|---|
| TTFT | ms | All servers, all experiments |
| E2E latency | ms | All servers, all experiments |
| TPOT | ms/token | All servers |
| p50 / p95 | ms | benchmark.py, experiment_prefill.py |
| VRAM usage | MB | All servers via pynvml |
| RAM usage | MB | All servers via psutil |
| Cache hit rate | % | prefill_server.py, experiment_prefill.py |
| Prefill overhead | ms | prefill_manager.py |
| Target compliance | % | benchmark.py |

---

---

### Phase 4 — Speculative Decoding + KV Eviction (`speculative_engine.py`, `kv_eviction.py`, `phase4_server.py`, `experiment_speculative.py`, `experiment_eviction.py`)

#### `speculative_engine.py` — SpeculativeEngine

Implements the draft-target speculative sampling loop from Leviathan et al. (2023).

**Algorithm per generation step:**
1. Draft model (TinyLlama 1.1B) autoregressively generates k candidate tokens
2. Target model (Mistral 7B) scores all k positions in a **single forward pass** — this is the key parallelism
3. Each draft token is accepted if `p_target ≥ p_draft`, otherwise accepted with probability `p_target / p_draft`
4. On first rejection: use target's corrected token and restart the draft loop
5. If all k accepted: take a bonus token from the target's distribution

**Tracked per request:** `tokens_accepted`, `tokens_rejected`, `acceptance_rate`, `speedup_ratio` (tokens/target_call), `ttft_ms`, `tpot_ms`

**Important finding (hypothesis to verify):** Speculative decoding improves TPOT (throughput), not TTFT. TTFT still requires a full prefill pass. This distinction must be clearly reported.

#### `kv_eviction.py` — Eviction Policies

Two policies implemented as context management wrappers:

| Policy | Class | Behaviour |
|---|---|---|
| LRU | `LRUEvictionPolicy` | Keeps last `max_ctx` tokens (oldest evicted) — mirrors llama.cpp default |
| Adaptive | `AdaptiveEvictionPolicy` | Keeps first `sink_tokens` (attention sinks) + last `recent_tokens` + highest-scored middle tokens |

The adaptive policy is based on the StreamingLLM insight (Xiao et al., 2023, ref [8] in paper): the first few tokens receive disproportionately high attention in all layers and must not be evicted.

Both policies expose `.trim(tokens) → tokens` — a pure function that can be applied to any prompt before inference without modifying the model.

#### `phase4_server.py` — Phase 4 Server (port 8002)

Combines all three optimisations: prefill (Phase 3) + speculative decoding + eviction policy.

Key env vars: `SPEC_K`, `SPEC_ENABLED`, `EVICTION_POLICY` (`lru`/`adaptive`), `EVICTION_MAX_CTX`

#### `experiment_speculative.py` — Experiment 1

Sweeps k ∈ {1, 4, 8}. For each k, auto-launches the Phase 4 server as a subprocess, runs 20 requests, records per-request metrics, tears down the server, then moves to the next k.

Output: `results/phase4_speculative_results.csv` + `results/phase4_speculative_summary.txt`

#### `experiment_eviction.py` — Experiment 2

Three conditions: unconstrained (max_ctx=2048), LRU (max_ctx=512), Adaptive (max_ctx=512). Uses the revision history workload (long prompts that exceed the window) as the stress case.

Output: `results/phase4_eviction_results.csv` + `results/phase4_eviction_summary.txt`

---

## File Map

```
pdc_inference/
│
├── server.py                Phase 1 — FastAPI inference server (port 8000)
├── workload_gen.py          Phase 2 — Synthetic editor workload generator
├── benchmark.py             Phase 2 — Async benchmark harness (CSV output)
│
├── prefill_manager.py       Phase 3 — KV prefill logic (PrefillManager class)
├── prefill_server.py        Phase 3 — Prefill-aware FastAPI server (port 8001)
├── experiment_prefill.py    Phase 3 — Cold vs warm A/B experiment
│
├── speculative_engine.py    Phase 4 — Draft-target speculative sampling engine
├── kv_eviction.py           Phase 4 — LRU and adaptive eviction policies
├── phase4_server.py         Phase 4 — Combined server: prefill+spec+eviction (port 8002)
├── experiment_speculative.py Phase 4 — k ∈ {1,4,8} sweep experiment
├── experiment_eviction.py   Phase 4 — LRU vs adaptive eviction experiment
│
├── requirements.txt         Python dependencies
├── QUICKSTART.md            Step-by-step run instructions
└── IMPLEMENTATION.md        This file
```

---

## Metrics Collected

| Metric | Unit | Collected in |
|---|---|---|
| TTFT | ms | All servers, all experiments |
| E2E latency | ms | All servers, all experiments |
| TPOT | ms/token | All servers |
| p50 / p95 | ms | benchmark.py, all experiment scripts |
| VRAM usage | MB | All servers via pynvml |
| RAM usage | MB | All servers via psutil |
| Cache hit rate | % | prefill_server.py, experiment_prefill.py |
| Prefill overhead | ms | prefill_manager.py |
| Acceptance rate | % | speculative_engine.py, experiment_speculative.py |
| Speedup ratio | tokens/call | speculative_engine.py |
| Tokens evicted | count | kv_eviction.py |
| Target compliance | % | benchmark.py |

---

## Remaining Phases

- **Phase 5:** Final paper write-up — Experiments, Results & Discussion, Conclusion sections; matplotlib plots of all CSV results
- **Phase 6:** Presentation and viva (4–8 May)
