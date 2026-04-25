# PDC Inference — Phase 1 & 2

**Real-Time AI Text Editing via Low-Latency and Memory-Efficient Transformer Inference**

---

## Hardware target

RTX 4060 Ti (8 GB or 16 GB VRAM).

| VRAM  | Recommended model file              | VRAM usage |
|-------|-------------------------------------|------------|
| 8 GB  | `mistral-7b-instruct-v0.2.Q4_K_M.gguf` | ~4.5 GB |
| 16 GB | `mistral-7b-instruct-v0.2.Q8_0.gguf`   | ~7.8 GB |

The Q8_0 variant is preferred for this project because INT8 quantization is
part of the experimental design. If you only have 8 GB, use Q4_K_M and note
the quantization level in your paper.

---

## 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux / macOS

pip install -r requirements.txt

# Install llama-cpp-python with CUDA support (RTX 4060 Ti)
# Windows (PowerShell):
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python --force-reinstall --no-cache-dir

# Linux:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

> **Note**: You need the CUDA toolkit installed and your NVIDIA driver up to date.
> Verify with `nvcc --version` and `nvidia-smi`.

---

## 2. Download the model

```bash
mkdir models

# Option A — Hugging Face CLI (recommended)
pip install huggingface-hub
huggingface-cli download \
  TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q8_0.gguf \
  --local-dir models

# Option B — direct wget (Linux)
wget -P models https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf
```

---

## 3. Start the server

```bash
# Full GPU offload (recommended for RTX 4060 Ti)
MODEL_PATH=models/mistral-7b-instruct-v0.2.Q8_0.gguf \
N_GPU_LAYERS=-1 \
N_CTX=4096 \
uvicorn server:app --host 0.0.0.0 --port 8000

# Windows PowerShell equivalent:
$env:MODEL_PATH="models\mistral-7b-instruct-v0.2.Q8_0.gguf"
$env:N_GPU_LAYERS="-1"
$env:N_CTX="4096"
uvicorn server:app --host 0.0.0.0 --port 8000
```

Check the server is up:
```bash
curl http://localhost:8000/health
```
Expected output includes `"model_loaded": true` and live VRAM stats.

---

## 4. Run the baseline benchmark (Phase 2)

```bash
# Full baseline: autocomplete + rewrite, 20 requests each, single user
python benchmark.py --mode both --n 20

# Concurrent sessions test (simulates 4 simultaneous users)
python benchmark.py --mode autocomplete --n 40 --concurrency 4

# Context length sweep (revision history workload)
python benchmark.py --mode revision --n 20
```

Results are saved as CSV files in `results/`. These are your **baseline numbers**
for comparison in Phases 3 and 4.

---

## 5. File structure

```
pdc_inference/
├── server.py          # FastAPI inference server (Phase 1)
├── benchmark.py       # Async benchmarking harness (Phase 2)
├── workload_gen.py    # Synthetic editor workload generator
├── requirements.txt   # Python dependencies
├── models/            # Place .gguf model files here
└── results/           # CSV benchmark output (auto-created)
```

---

## 6. Key metrics being recorded

| Metric | Description | Target |
|--------|-------------|--------|
| TTFT   | Time to first token (ms) | < 200ms autocomplete, < 400ms rewrite |
| p50 / p95 | Median and 95th percentile latency | — |
| TPOT  | Time per output token (ms) | — |
| VRAM  | GPU memory used during inference | — |
| Acceptance rate | Draft token acceptance (Phase 4) | — |

---

## 7. Troubleshooting

**`CUDA out of memory`** — Switch to Q4_K_M and set `N_CTX=2048`.

**`llama_cpp` not using GPU** — Confirm the CUDA build: in Python, run:
```python
from llama_cpp import llama_supports_gpu_offload
print(llama_supports_gpu_offload())  # should print True
```
If `False`, reinstall with the `CMAKE_ARGS` flag as shown above.

**`Connection refused` on benchmark** — Make sure `uvicorn` is still running
in a separate terminal before running `benchmark.py`.
