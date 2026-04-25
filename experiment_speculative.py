"""
PDC Phase 4 — Experiment 1: Speculative Decoding
Sweeps block size k ∈ {1, 4, 8} and records:
  - TTFT (p50, p95)
  - End-to-end latency (p50, p95)
  - Acceptance rate
  - Speedup ratio (tokens/target_call vs baseline = 1.0)
  - TPOT (time per output token)

Key hypothesis: higher k → better throughput but potentially lower
acceptance rate. The optimal k depends on TinyLlama/Mistral alignment.

IMPORTANT finding to verify: speculative decoding primarily improves
TPOT/throughput, NOT TTFT. This is expected and should be reported clearly.

Usage:
    # Start phase4_server for each k value automatically (subprocess mode)
    python experiment_speculative.py --n 20

    # Manual mode (server already running on 8002)
    python experiment_speculative.py --n 20 --manual --k 4
"""

import argparse
import asyncio
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import aiohttp

HOST = "http://localhost:8002"
STARTUP_WAIT = 45  # seconds to wait for model to load


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SpecExpResult:
    k: int
    prompt: str
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    tpot_ms: float = 0.0
    tokens: int = 0
    acceptance_rate: float = 0.0
    speedup_ratio: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------

async def send_spec_request(
    session: aiohttp.ClientSession,
    prompt: str,
    k: int,
    max_tokens: int = 64,
) -> SpecExpResult:
    result = SpecExpResult(k=k, prompt=prompt[:120])
    try:
        payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.3}
        async with session.post(
            f"{HOST}/complete_spec", json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                result.error = f"HTTP {resp.status}: {body[:120]}"
                return result
            data = await resp.json()
            result.ttft_ms        = data.get("ttft_ms", 0.0)
            result.e2e_ms         = data.get("e2e_ms", 0.0)
            result.tpot_ms        = data.get("tpot_ms", 0.0)
            result.tokens         = data.get("tokens", 0)
            result.acceptance_rate = data.get("acceptance_rate", 0.0)
            result.speedup_ratio  = data.get("speedup_ratio", 0.0)
    except asyncio.TimeoutError:
        result.error = "Timeout"
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
    return result


# ---------------------------------------------------------------------------
# Wait for server
# ---------------------------------------------------------------------------

async def wait_for_server(timeout: int = STARTUP_WAIT) -> bool:
    deadline = time.time() + timeout
    print(f"  Waiting for server (up to {timeout}s)...", end="", flush=True)
    async with aiohttp.ClientSession() as s:
        while time.time() < deadline:
            try:
                async with s.get(f"{HOST}/health", timeout=aiohttp.ClientTimeout(total=3)) as r:
                    h = await r.json()
                    if h.get("model_loaded"):
                        print(" ready.")
                        return True
            except Exception:
                pass
            print(".", end="", flush=True)
            await asyncio.sleep(3)
    print(" TIMEOUT.")
    return False


# ---------------------------------------------------------------------------
# Run one k-value sweep
# ---------------------------------------------------------------------------

async def run_k_sweep(k: int, workload: list, warmup: int = 2) -> List[SpecExpResult]:
    results = []
    connector = aiohttp.TCPConnector(limit=4)
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"\n  [k={k}] Warmup ({warmup} requests)...")
        for p in workload[:warmup]:
            r = await send_spec_request(session, p["prompt"], k, p.get("max_tokens", 64))
            print(f"    warmup → {'ok' if r.ok else r.error}")

        print(f"  [k={k}] Running {len(workload) - warmup} requests...")
        for i, p in enumerate(workload[warmup:]):
            r = await send_spec_request(session, p["prompt"], k, p.get("max_tokens", 64))
            results.append(r)
            if r.ok:
                print(
                    f"    [{i+1:>3}] TTFT={r.ttft_ms:>6.1f}ms | "
                    f"TPOT={r.tpot_ms:>5.1f}ms/tok | "
                    f"accept={r.acceptance_rate:.2f} | "
                    f"speedup={r.speedup_ratio:.2f}x"
                )
            else:
                print(f"    [{i+1:>3}] ERROR: {r.error}")
    return results


# ---------------------------------------------------------------------------
# Statistics + reporting
# ---------------------------------------------------------------------------

def _pct(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def build_summary(all_results: dict, out_dir: Path) -> str:
    """
    all_results: {k: [SpecExpResult, ...]}
    """
    lines = [
        "=" * 72,
        "  Phase 4 — Speculative Decoding Experiment Summary",
        "=" * 72,
        f"\n  {'k':<6} {'TTFT p50':>10} {'TTFT p95':>10} {'TPOT p50':>10} "
        f"{'Accept':>8} {'Speedup':>8} {'N':>5}",
        f"  {'-'*68}",
    ]

    baseline_tpot = None

    for k in sorted(all_results.keys()):
        results = [r for r in all_results[k] if r.ok]
        if not results:
            lines.append(f"  k={k:<4} — no valid results")
            continue

        ttfts   = [r.ttft_ms        for r in results]
        tpots   = [r.tpot_ms        for r in results if r.tpot_ms > 0]
        accepts = [r.acceptance_rate for r in results]
        speedups = [r.speedup_ratio  for r in results]

        ttft_p50  = _pct(ttfts, 50)
        ttft_p95  = _pct(ttfts, 95)
        tpot_p50  = _pct(tpots, 50) if tpots else 0.0
        avg_accept = statistics.mean(accepts) if accepts else 0.0
        avg_speedup = statistics.mean(speedups) if speedups else 0.0

        if k == 1:
            baseline_tpot = tpot_p50

        lines.append(
            f"  k={k:<4} {ttft_p50:>10.1f} {ttft_p95:>10.1f} {tpot_p50:>10.2f} "
            f"{avg_accept:>8.2f} {avg_speedup:>8.2f}x {len(results):>5}"
        )

    lines += [
        f"  {'-'*68}",
        "",
        "  Columns: TTFT p50/p95 (ms) | TPOT p50 (ms/tok) | "
        "Acceptance rate | Speedup ratio",
        "",
        "  Key finding: Compare TPOT improvement across k values.",
        "  TTFT is expected to be similar across k (spec decoding is a",
        "  throughput optimisation, not a prefill optimisation).",
        "=" * 72,
    ]

    summary = "\n".join(lines)
    print(f"\n{summary}")

    path = out_dir / "phase4_speculative_summary.txt"
    path.write_text(summary)
    print(f"\n  Saved → {path}")
    return summary


def save_csv(all_results: dict, out_dir: Path):
    path = out_dir / "phase4_speculative_results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["k", "prompt", "ttft_ms", "e2e_ms", "tpot_ms",
                        "tokens", "acceptance_rate", "speedup_ratio", "error"],
        )
        writer.writeheader()
        for k, results in all_results.items():
            for r in results:
                writer.writerow({
                    "k": r.k, "prompt": r.prompt[:80],
                    "ttft_ms": r.ttft_ms, "e2e_ms": r.e2e_ms,
                    "tpot_ms": r.tpot_ms, "tokens": r.tokens,
                    "acceptance_rate": r.acceptance_rate,
                    "speedup_ratio": r.speedup_ratio,
                    "error": r.error,
                })
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _main(args):
    from workload_gen import generate_autocomplete_workload

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wl = generate_autocomplete_workload(args.n + 2)
    k_values = [int(k) for k in args.k_values.split(",")]

    all_results = {}

    for k in k_values:
        print(f"\n{'='*60}")
        print(f"  Speculative Decoding — k={k}")
        print(f"{'='*60}")

        if not args.manual:
            # Launch server subprocess with this k value
            env = os.environ.copy()
            env["SPEC_K"]          = str(k)
            env["SPEC_ENABLED"]    = "1"
            env["EVICTION_POLICY"] = args.eviction
            env["N_CTX"]           = "2048"

            print(f"  Starting server with k={k}...")
            proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "phase4_server:app",
                 "--host", "0.0.0.0", "--port", "8002"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                ready = await wait_for_server(STARTUP_WAIT)
                if not ready:
                    print(f"  Server failed to start for k={k}, skipping.")
                    proc.terminate()
                    continue

                all_results[k] = await run_k_sweep(k, wl, warmup=2)
            finally:
                proc.terminate()
                proc.wait()
                await asyncio.sleep(2)  # Allow port to free
        else:
            # Manual: assume server already running
            if not await wait_for_server(10):
                print("  Server not reachable.")
                return
            all_results[k] = await run_k_sweep(k, wl, warmup=2)

    if all_results:
        save_csv(all_results, out_dir)
        build_summary(all_results, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Exp 1 — Speculative Decoding")
    parser.add_argument("--n",         type=int,   default=20)
    parser.add_argument("--k-values",  default="1,4,8",  help="Comma-separated k values")
    parser.add_argument("--eviction",  default="lru",    help="lru | adaptive")
    parser.add_argument("--out-dir",   default="results")
    parser.add_argument("--manual",    action="store_true",
                        help="Skip server management (server already running on 8002)")
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
