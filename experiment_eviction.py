"""
PDC Phase 4 — Experiment 2: KV Eviction Policy Comparison
Compares LRU vs Adaptive eviction under a forced small context window
(EVICTION_MAX_CTX = 256 / 512 tokens) which simulates constrained VRAM.

Measures:
  - TTFT under constrained context
  - Memory footprint (VRAM before/after)
  - Output quality proxy: response length ratio vs unconstrained baseline
  - Eviction overhead: tokens evicted, eviction count

Design:
  Three conditions run against the phase4_server:
    A) Unconstrained   — max_ctx=2048, no eviction (baseline)
    B) LRU             — max_ctx=512
    C) Adaptive        — max_ctx=512 (sinks=4, recent=128)

  Workload: revision history prompts (varying context length 50–900 chars).
  These are the worst case for eviction — long contexts that exceed the window.

Usage:
    python experiment_eviction.py --n 20
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
STARTUP_WAIT = 45


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EvictionResult:
    condition: str            # "unconstrained" | "lru" | "adaptive"
    prompt_len: int           # chars in input prompt
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    response_len: int = 0     # chars in generated text (quality proxy)
    vram_used_mb: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


# ---------------------------------------------------------------------------
# Single request (non-streaming, uses /complete_spec with SPEC_ENABLED=0)
# ---------------------------------------------------------------------------

async def send_eviction_request(
    session: aiohttp.ClientSession,
    prompt: str,
    condition: str,
) -> EvictionResult:
    result = EvictionResult(condition=condition, prompt_len=len(prompt))
    try:
        payload = {"prompt": prompt, "max_tokens": 64, "temperature": 0.1}
        async with session.post(
            f"{HOST}/complete_spec", json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                result.error = f"HTTP {resp.status}: {body[:120]}"
                return result
            data = await resp.json()
            result.ttft_ms      = data.get("ttft_ms", 0.0)
            result.e2e_ms       = data.get("e2e_ms", 0.0)
            result.response_len = len(data.get("text", ""))
            mem = data.get("mem", {})
            result.vram_used_mb = mem.get("vram_used_mb", 0.0)
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
    print(f"  Waiting for server...", end="", flush=True)
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
# Run one condition
# ---------------------------------------------------------------------------

async def run_condition(
    condition: str,
    workload: list,
    warmup: int = 2,
) -> List[EvictionResult]:
    results = []
    connector = aiohttp.TCPConnector(limit=2)
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"\n  [{condition}] Warmup ({warmup})...")
        for p in workload[:warmup]:
            r = await send_eviction_request(session, p["prompt"], condition)
            print(f"    warmup → {'ok' if r.ok else r.error}")

        print(f"  [{condition}] Running {len(workload) - warmup} requests...")
        for i, p in enumerate(workload[warmup:]):
            r = await send_eviction_request(session, p["prompt"], condition)
            results.append(r)
            if r.ok:
                print(
                    f"    [{i+1:>3}] ctx={r.prompt_len:>4}chars | "
                    f"TTFT={r.ttft_ms:>6.1f}ms | "
                    f"resp={r.response_len:>3}chars | "
                    f"VRAM={r.vram_used_mb:.0f}MB"
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


def build_eviction_summary(all_results: dict, out_dir: Path) -> str:
    """all_results: {condition_name: [EvictionResult]}"""
    lines = [
        "=" * 72,
        "  Phase 4 — KV Eviction Policy Experiment Summary",
        "=" * 72,
        f"\n  {'Condition':<18} {'TTFT p50':>10} {'TTFT p95':>10} "
        f"{'VRAM avg':>10} {'Resp/Input':>12} {'N':>5}",
        f"  {'-'*68}",
    ]

    baseline_resp = None

    for cond in ["unconstrained", "lru", "adaptive"]:
        results = [r for r in all_results.get(cond, []) if r.ok]
        if not results:
            lines.append(f"  {cond:<18} — no valid results")
            continue

        ttfts  = [r.ttft_ms      for r in results]
        vrams  = [r.vram_used_mb for r in results if r.vram_used_mb > 0]
        resps  = [r.response_len for r in results]
        inputs = [r.prompt_len   for r in results]

        ttft_p50   = _pct(ttfts, 50)
        ttft_p95   = _pct(ttfts, 95)
        avg_vram   = statistics.mean(vrams) if vrams else 0.0
        avg_resp   = statistics.mean(resps) if resps else 0.0
        avg_input  = statistics.mean(inputs) if inputs else 1.0
        resp_ratio = avg_resp / max(avg_input, 1) * 100  # response length as % of input

        if cond == "unconstrained":
            baseline_resp = avg_resp

        lines.append(
            f"  {cond:<18} {ttft_p50:>10.1f} {ttft_p95:>10.1f} "
            f"{avg_vram:>10.0f} {resp_ratio:>11.1f}% {len(results):>5}"
        )

    lines += [
        f"  {'-'*68}",
        "",
        "  Resp/Input: average response length as % of prompt length (quality proxy).",
        "  Higher is better — eviction should not significantly shrink outputs.",
        "",
        "  Key finding: Adaptive eviction should maintain higher Resp/Input than",
        "  LRU under long contexts by preserving attention-sink and recent tokens.",
        "=" * 72,
    ]

    summary = "\n".join(lines)
    print(f"\n{summary}")

    path = out_dir / "phase4_eviction_summary.txt"
    path.write_text(summary)
    print(f"\n  Saved → {path}")
    return summary


def save_eviction_csv(all_results: dict, out_dir: Path):
    path = out_dir / "phase4_eviction_results.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "prompt_len", "ttft_ms", "e2e_ms",
                        "response_len", "vram_used_mb", "error"],
        )
        writer.writeheader()
        for cond, results in all_results.items():
            for r in results:
                writer.writerow({
                    "condition": r.condition, "prompt_len": r.prompt_len,
                    "ttft_ms": r.ttft_ms, "e2e_ms": r.e2e_ms,
                    "response_len": r.response_len,
                    "vram_used_mb": r.vram_used_mb, "error": r.error,
                })
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Server launch helper
# ---------------------------------------------------------------------------

def _make_server_env(eviction: str, max_ctx: int) -> dict:
    env = os.environ.copy()
    env["SPEC_ENABLED"]      = "0"    # disable spec decoding for eviction experiment
    env["EVICTION_POLICY"]   = eviction
    env["EVICTION_MAX_CTX"]  = str(max_ctx)
    env["N_CTX"]             = "2048"
    return env


async def _run_condition_with_server(
    condition: str,
    eviction: str,
    max_ctx: int,
    workload: list,
    warmup: int,
) -> List[EvictionResult]:
    env = _make_server_env(eviction, max_ctx)
    print(f"\n{'='*60}")
    print(f"  Condition: {condition.upper()}  (eviction={eviction}, max_ctx={max_ctx})")
    print(f"{'='*60}")

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
            print(f"  Server failed to start for condition={condition}.")
            return []
        return await run_condition(condition, workload, warmup)
    finally:
        proc.terminate()
        proc.wait()
        await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _main(args):
    from workload_gen import generate_revision_history_workload

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Revision history workload — long context prompts stress eviction policies
    wl = generate_revision_history_workload(args.n + 2)
    warmup = 2

    all_results = {}

    conditions = [
        ("unconstrained", "lru",      2048),
        ("lru",           "lru",      512),
        ("adaptive",      "adaptive", 512),
    ]

    for condition, eviction, max_ctx in conditions:
        if args.manual:
            if not await wait_for_server(10):
                print("Server not reachable.")
                return
            all_results[condition] = await run_condition(condition, wl, warmup)
        else:
            all_results[condition] = await _run_condition_with_server(
                condition, eviction, max_ctx, wl, warmup,
            )

        # Fetch eviction stats from server (if still up — manual mode only)
        if args.manual:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(f"{HOST}/spec/stats") as r:
                        stats = await r.json()
                        evict = stats.get("eviction_stats", {})
                        print(f"  Eviction stats: {json.dumps(evict, indent=4)}")
            except Exception:
                pass

    if all_results:
        save_eviction_csv(all_results, out_dir)
        build_eviction_summary(all_results, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Exp 2 — KV Eviction")
    parser.add_argument("--n",       type=int, default=20)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--manual",  action="store_true",
                        help="Skip server management (server already running on 8002)")
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
