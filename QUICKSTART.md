# Quickstart
## PDC Project — Phases 1–3 on RTX 4060 Ti

---

## Step 1 — Python environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

---

## Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

Then install llama-cpp-python **with CUDA support** (required — plain pip install gives CPU-only):

```bash
# Windows (PowerShell)
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Linux
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

Verify CUDA is active:
```python
from llama_cpp import llama_supports_gpu_offload
print(llama_supports_gpu_offload())   # must print True
```

---

## Step 3 — Download the model

```bash
pip install huggingface-hub
mkdir models

# 16 GB VRAM (recommended — INT8, matches paper)
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q8_0.gguf --local-dir models

# 8 GB VRAM (use this instead if needed)
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir models
```

---

## Phase 1 — Start the baseline server

```bash
# Windows
$env:MODEL_PATH="models\mistral-7b-instruct-v0.2.Q8_0.gguf"
$env:N_GPU_LAYERS="-1"
uvicorn server:app --host 0.0.0.0 --port 8000

# Linux / macOS
MODEL_PATH=models/mistral-7b-instruct-v0.2.Q8_0.gguf \
N_GPU_LAYERS=-1 \
uvicorn server:app --host 0.0.0.0 --port 8000
```

Confirm it's running:
```bash
curl http://localhost:8000/health
# Expect: {"status":"ok","model_loaded":true,...}
```

---

## Phase 2 — Run the baseline benchmark

Open a **second terminal** (keep the server running in the first).

```bash
# Full baseline: autocomplete + rewrite, 20 requests each
python benchmark.py --mode both --n 20

# Concurrent sessions (4 simultaneous users)
python benchmark.py --mode autocomplete --n 40 --concurrency 4

# Context length sweep
python benchmark.py --mode revision --n 20
```

Results are saved to `results/*.csv`. These are your **Table 1 baseline numbers**.

---

## Phase 3 — Run the KV prefill experiment

Start the prefill server in a new terminal (port 8001, keep Phase 1 server untouched):

```bash
# Windows
$env:MODEL_PATH="models\mistral-7b-instruct-v0.2.Q8_0.gguf"
$env:N_GPU_LAYERS="-1"
$env:PREFILL_TOKENS="32"
$env:PREFILL_IDLE_MS="150"
uvicorn prefill_server:app --host 0.0.0.0 --port 8001

# Linux / macOS
MODEL_PATH=models/mistral-7b-instruct-v0.2.Q8_0.gguf \
N_GPU_LAYERS=-1 \
PREFILL_TOKENS=32 \
PREFILL_IDLE_MS=150 \
uvicorn prefill_server:app --host 0.0.0.0 --port 8001
```

Then run the experiment (in a third terminal):

```bash
# Autocomplete workload (30 requests, cold vs warm comparison)
python experiment_prefill.py --n 30 --workload autocomplete

# Revision history / context length sweep
python experiment_prefill.py --n 30 --workload revision
```

This produces:
- `results/phase3_prefill_cold.csv`
- `results/phase3_prefill_warm.csv`
- `results/phase3_summary.txt` ← paste into your paper

---

## Verify the CUDA build is working correctly

Watch the server terminal when a request comes in. You should see log lines like:

```
TTFT=180.3ms | E2E=2340.1ms | tokens=48 | TPOT=45.2ms/tok | VRAM=7812.0MB
```

If TTFT is > 5000ms, the model is running on CPU. Reinstall with the CUDA flag.

---

## Phase 4 — Speculative decoding + KV eviction experiments

Download the draft model (TinyLlama 1.1B — only ~700 MB):
```bash
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir models
```

**Experiment 1 — Speculative decoding (k sweep):**
```bash
# Auto-manages server subprocesses — just run this
python experiment_speculative.py --n 20

# Saves: results/phase4_speculative_summary.txt
#        results/phase4_speculative_results.csv
```

**Experiment 2 — KV eviction (LRU vs adaptive):**
```bash
python experiment_eviction.py --n 20

# Saves: results/phase4_eviction_summary.txt
#        results/phase4_eviction_results.csv
```

Both experiment scripts automatically launch and tear down the Phase 4 server
for each condition. No manual server management needed.

If you prefer to manage the server yourself (e.g. for debugging):
```bash
# Start Phase 4 server manually
MODEL_PATH=models/mistral-7b-instruct-v0.2.Q8_0.gguf \
DRAFT_MODEL_PATH=models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
SPEC_K=4 \
EVICTION_POLICY=lru \
uvicorn phase4_server:app --port 8002

# Then run experiments in manual mode
python experiment_speculative.py --n 20 --manual --k-values 4
python experiment_eviction.py --n 20 --manual
```

---

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/mistral-7b-instruct-v0.2.Q8_0.gguf` | Path to GGUF model |
| `N_GPU_LAYERS` | `-1` | GPU layers (-1 = full offload) |
| `N_CTX` | `4096` | Context window size |
| `N_THREADS` | `4` | CPU threads (for CPU-offloaded layers) |
| `PREFILL_ENABLED` | `1` | Set to `0` to disable prefill for cold baseline |
| `PREFILL_TOKENS` | `32` | Tokens to prefetch into KV cache |
| `PREFILL_IDLE_MS` | `150` | Idle time (ms) before prefill fires |
