"""
PDC Benchmark Harness — Phase 2
Sends workloads to the inference server, collects per-request metrics,
and prints a formatted report with p50/p95 latency statistics.

Usage examples:
    # Single-user baseline (autocomplete)
    python benchmark.py --mode autocomplete --n 20

    # Single-user baseline (rewrite)
    python benchmark.py --mode rewrite --n 20

    # Concurrent sessions stress test
    python benchmark.py --mode autocomplete --n 40 --concurrency 4

    # Full baseline suite
    python benchmark.py --mode both --n 30
"""

import argparse
import asyncio
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    prompt: str
    endpoint: str
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    tokens: int = 0
    tpot_ms: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


# ---------------------------------------------------------------------------
# Single-request runner
# ---------------------------------------------------------------------------

async def send_request(
    session: aiohttp.ClientSession,
    host: str,
    endpoint: str,
    payload: dict,
) -> RequestResult:
    url = f"{host}/{endpoint}"
    prompt = payload.get("prompt", payload.get("text", ""))
    result = RequestResult(prompt=prompt[:120], endpoint=endpoint)
    start = time.perf_counter()
    first_token = True

    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status != 200:
                body = await resp.text()
                result.error = f"HTTP {resp.status}: {body[:120]}"
                result.e2e_ms = (time.perf_counter() - start) * 1000
                return result

            async for raw_line in resp.content:
                line = raw_line.decode(errors="replace").strip()
                if not line:
                    continue

                if line == "data: [DONE]":
                    break

                # Stats event (final)
                if line.startswith("data: {"):
                    try:
                        data = json.loads(line[6:])
                        if "ttft_ms" in data:
                            result.ttft_ms = data["ttft_ms"]
                            result.e2e_ms  = data["e2e_ms"]
                            result.tokens  = data["tokens"]
                            result.tpot_ms = data["tpot_ms"]
                    except json.JSONDecodeError:
                        pass
                    continue

                # Regular token
                if line.startswith("data: ") and first_token:
                    result.ttft_ms = (time.perf_counter() - start) * 1000
                    first_token = False

    except asyncio.TimeoutError:
        result.error = "Timeout"
        result.e2e_ms = (time.perf_counter() - start) * 1000
    except aiohttp.ClientConnectorError as e:
        result.error = f"Connection error: {e}"
        result.e2e_ms = (time.perf_counter() - start) * 1000
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        result.e2e_ms = (time.perf_counter() - start) * 1000

    return result


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

async def run_benchmark(
    host: str,
    workload: List[Tuple[str, dict]],      # [(endpoint, payload), ...]
    concurrency: int = 1,
    warmup: int = 2,
    label: str = "Benchmark",
) -> List[RequestResult]:
    """
    Runs the workload against the server.
    warmup requests are sent first and excluded from results.
    """
    connector = aiohttp.TCPConnector(limit=max(concurrency, 4))
    results: List[RequestResult] = []

    async with aiohttp.ClientSession(connector=connector) as session:
        # --- Warmup ---
        warmup_items = workload[:warmup]
        print(f"\n[{label}] Warming up ({warmup} requests)...")
        for ep, payload in warmup_items:
            r = await send_request(session, host, ep, payload)
            status = f"TTFT={r.ttft_ms:.0f}ms" if r.ok else f"ERROR: {r.error}"
            print(f"  warmup → {status}")

        # --- Benchmark ---
        bench_items = workload[warmup:]
        print(f"[{label}] Running {len(bench_items)} requests (concurrency={concurrency})...")

        if concurrency == 1:
            for i, (ep, payload) in enumerate(bench_items):
                r = await send_request(session, host, ep, payload)
                results.append(r)
                if r.ok:
                    print(f"  [{i+1:>3}] TTFT={r.ttft_ms:>6.1f}ms | E2E={r.e2e_ms:>7.1f}ms | {r.prompt[:45]}...")
                else:
                    print(f"  [{i+1:>3}] ERROR: {r.error}")
        else:
            sem = asyncio.Semaphore(concurrency)

            async def bounded(ep, payload):
                async with sem:
                    return await send_request(session, host, ep, payload)

            tasks = [bounded(ep, payload) for ep, payload in bench_items]
            results = list(await asyncio.gather(*tasks))

    return results


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _pct(data: List[float], p: int) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


def print_report(results: List[RequestResult], label: str = "Results", out_dir: Path = Path(".")):
    ok = [r for r in results if r.ok]
    errs = [r for r in results if not r.ok]

    print(f"\n{'━'*62}")
    print(f"  {label}")
    print(f"{'━'*62}")
    print(f"  Requests   : {len(results)}  |  OK: {len(ok)}  |  Errors: {len(errs)}")

    if not ok:
        for r in errs[:5]:
            print(f"  ERROR: {r.error}")
        return

    ttfts = [r.ttft_ms for r in ok]
    e2es  = [r.e2e_ms  for r in ok]
    tpots = [r.tpot_ms for r in ok if r.tpot_ms > 0]

    def block(title, data):
        print(f"\n  {title}")
        print(f"    mean  : {statistics.mean(data):>8.1f} ms")
        print(f"    p50   : {_pct(data, 50):>8.1f} ms")
        print(f"    p95   : {_pct(data, 95):>8.1f} ms")
        print(f"    min   : {min(data):>8.1f} ms")
        print(f"    max   : {max(data):>8.1f} ms")

    block("TTFT (time to first token)", ttfts)
    block("End-to-end latency", e2es)

    if tpots:
        print(f"\n  Time per output token (TPOT)")
        print(f"    mean  : {statistics.mean(tpots):>8.2f} ms/tok")
        print(f"    p50   : {_pct(tpots, 50):>8.2f} ms/tok")

    # Target compliance
    ac = [r for r in ok if r.endpoint == "complete"]
    rw = [r for r in ok if r.endpoint == "rewrite"]
    if ac:
        within = sum(1 for r in ac if r.ttft_ms < 200)
        print(f"\n  Autocomplete <200ms TTFT: {within}/{len(ac)} ({within/len(ac)*100:.0f}%)")
    if rw:
        within = sum(1 for r in rw if r.ttft_ms < 400)
        print(f"  Rewrite      <400ms TTFT: {within}/{len(rw)} ({within/len(rw)*100:.0f}%)")

    # Save CSV
    slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    out_path = out_dir / f"results_{slug}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["endpoint", "prompt", "ttft_ms", "e2e_ms", "tokens", "tpot_ms", "error"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "endpoint": r.endpoint,
                "prompt":   r.prompt[:100],
                "ttft_ms":  r.ttft_ms,
                "e2e_ms":   r.e2e_ms,
                "tokens":   r.tokens,
                "tpot_ms":  r.tpot_ms,
                "error":    r.error,
            })

    print(f"\n  Saved → {out_path}")
    print(f"{'━'*62}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main(args):
    from workload_gen import (
        generate_autocomplete_workload,
        generate_rewrite_workload,
        generate_revision_history_workload,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify server is up
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{args.host}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                health = await r.json()
                print(f"Server OK — model_loaded={health['model_loaded']}")
                if not health["model_loaded"]:
                    print("ERROR: Model is not loaded. Check MODEL_PATH.")
                    return
    except Exception as e:
        print(f"ERROR: Cannot reach server at {args.host} — {e}")
        return

    # ---- Autocomplete baseline ----
    if args.mode in ("autocomplete", "both"):
        wl = [("complete", p) for p in generate_autocomplete_workload(args.n + 2)]
        results = await run_benchmark(
            args.host, wl, args.concurrency,
            warmup=2, label=f"Autocomplete (c={args.concurrency})",
        )
        print_report(results, f"Autocomplete baseline c={args.concurrency}", out_dir)

    # ---- Rewrite baseline ----
    if args.mode in ("rewrite", "both"):
        wl = [("rewrite", p) for p in generate_rewrite_workload(args.n + 2)]
        results = await run_benchmark(
            args.host, wl, args.concurrency,
            warmup=2, label=f"Rewrite (c={args.concurrency})",
        )
        print_report(results, f"Rewrite baseline c={args.concurrency}", out_dir)

    # ---- Revision history (varying context length) ----
    if args.mode in ("revision", "both"):
        wl = [("complete", p) for p in generate_revision_history_workload(args.n + 2)]
        results = await run_benchmark(
            args.host, wl, 1,
            warmup=2, label="Revision history (context sweep)",
        )
        print_report(results, "Revision history context sweep", out_dir)


def main():
    parser = argparse.ArgumentParser(description="PDC Inference Benchmark Harness")
    parser.add_argument("--host",        default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--mode",        choices=["autocomplete", "rewrite", "revision", "both"],
                        default="both")
    parser.add_argument("--n",           type=int, default=20, help="Requests per mode (excl. warmup)")
    parser.add_argument("--concurrency", type=int, default=1,  help="Concurrent requests")
    parser.add_argument("--out-dir",     default="results",    help="Directory for CSV output")
    args = parser.parse_args()

    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
