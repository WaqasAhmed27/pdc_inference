"""
PDC Phase 3 — KV Prefill Experiment
Runs two benchmark passes against prefill_server.py:
  Pass A: cold inference (PREFILL_ENABLED=0 or cache always missed)
  Pass B: warm inference (PREFILL_ENABLED=1, idle timer fires before each request)

For each pass it records TTFT, e2e latency, cache hit rate, and memory.
Outputs:
  results/phase3_prefill_cold.csv
  results/phase3_prefill_warm.csv
  results/phase3_summary.txt   ← paste this into your paper Table 1

Usage:
    # Start the prefill server first:
    MODEL_PATH=models/mistral-7b-instruct-v0.2.Q8_0.gguf uvicorn prefill_server:app --port 8001

    # Then run this script:
    python experiment_prefill.py --n 30
    python experiment_prefill.py --n 30 --workload revision   # context length sweep
"""

import argparse
import asyncio
import csv
import json
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

import aiohttp

HOST = "http://localhost:8001"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PrefillResult:
    prompt: str
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    tokens: int = 0
    tpot_ms: float = 0.0
    cache_hit: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


# ---------------------------------------------------------------------------
# Single request
# ---------------------------------------------------------------------------

async def send_prefill_request(
    session: aiohttp.ClientSession,
    session_id: str,
    payload: dict,
    simulate_idle_ms: float = 200.0,
) -> PrefillResult:
    """
    Simulates the editor interaction pattern:
    1. POST /context  (editor typing update)
    2. Wait simulate_idle_ms (simulates user pausing before triggering autocomplete)
    3. POST /complete_prefill
    """
    prompt = payload.get("prompt", payload.get("text", ""))
    result = PrefillResult(prompt=prompt[:120])

    # Step 1: push context update (simulates keystroke)
    try:
        ctx_payload = {"session_id": session_id, "text": prompt}
        async with session.post(
            f"{HOST}/context", json=ctx_payload,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as r:
            if r.status != 200:
                result.error = f"Context update failed: HTTP {r.status}"
                return result
    except Exception as e:
        result.error = f"Context update error: {e}"
        return result

    # Step 2: simulate idle time (the prefill fires during this window)
    await asyncio.sleep(simulate_idle_ms / 1000.0)

    # Step 3: send the actual completion request
    complete_payload = {
        "session_id": session_id,
        "prompt":     prompt,
        "max_tokens": payload.get("max_tokens", 64),
        "temperature": payload.get("temperature", 0.2),
        "stream":     True,
    }
    start = time.perf_counter()
    first_token = True

    try:
        async with session.post(
            f"{HOST}/complete_prefill", json=complete_payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                result.error = f"HTTP {resp.status}: {body[:120]}"
                result.e2e_ms = (time.perf_counter() - start) * 1000
                return result

            async for raw_line in resp.content:
                line = raw_line.decode(errors="replace").strip()
                if not line or line == "data: [DONE]":
                    continue

                if line.startswith("data: {"):
                    try:
                        data = json.loads(line[6:])
                        if "ttft_ms" in data:
                            result.ttft_ms   = data["ttft_ms"]
                            result.e2e_ms    = data["e2e_ms"]
                            result.tokens    = data["tokens"]
                            result.tpot_ms   = data["tpot_ms"]
                            result.cache_hit = data.get("cache_hit", False)
                    except json.JSONDecodeError:
                        pass
                    continue

                if line.startswith("data: ") and first_token:
                    result.ttft_ms = (time.perf_counter() - start) * 1000
                    first_token = False

    except asyncio.TimeoutError:
        result.error = "Timeout"
        result.e2e_ms = (time.perf_counter() - start) * 1000
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        result.e2e_ms = (time.perf_counter() - start) * 1000

    return result


# ---------------------------------------------------------------------------
# Benchmark pass
# ---------------------------------------------------------------------------

async def run_pass(
    workload: list,
    label: str,
    idle_ms: float,
    warmup: int = 2,
) -> List[PrefillResult]:
    session_id = str(uuid.uuid4())
    results: List[PrefillResult] = []

    connector = aiohttp.TCPConnector(limit=4)
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"\n[{label}] Warming up ({warmup} requests, idle={idle_ms}ms)...")
        for payload in workload[:warmup]:
            r = await send_prefill_request(session, session_id, payload, idle_ms)
            print(f"  warmup → TTFT={r.ttft_ms:.0f}ms  hit={r.cache_hit}")

        print(f"[{label}] Running {len(workload) - warmup} requests...")
        for i, payload in enumerate(workload[warmup:]):
            r = await send_prefill_request(session, session_id, payload, idle_ms)
            results.append(r)
            if r.ok:
                hit_label = "HIT " if r.cache_hit else "MISS"
                print(
                    f"  [{i+1:>3}] {hit_label} | TTFT={r.ttft_ms:>6.1f}ms | "
                    f"E2E={r.e2e_ms:>7.1f}ms | {r.prompt[:45]}..."
                )
            else:
                print(f"  [{i+1:>3}] ERROR: {r.error}")

    return results


# ---------------------------------------------------------------------------
# Statistics + reporting
# ---------------------------------------------------------------------------

def _pct(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    return s[min(int(len(s) * p / 100), len(s) - 1)]


def save_csv(results: List[PrefillResult], path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt", "ttft_ms", "e2e_ms", "tokens", "tpot_ms", "cache_hit", "error"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "prompt":    r.prompt[:100],
                "ttft_ms":   r.ttft_ms,
                "e2e_ms":    r.e2e_ms,
                "tokens":    r.tokens,
                "tpot_ms":   r.tpot_ms,
                "cache_hit": r.cache_hit,
                "error":     r.error,
            })


def build_summary(
    cold_results: List[PrefillResult],
    warm_results: List[PrefillResult],
    out_dir: Path,
) -> str:
    def stats(results):
        ok = [r for r in results if r.ok]
        ttfts = [r.ttft_ms for r in ok]
        e2es  = [r.e2e_ms  for r in ok]
        hits  = sum(1 for r in ok if r.cache_hit)
        return {
            "n":         len(ok),
            "ttft_mean": statistics.mean(ttfts) if ttfts else 0,
            "ttft_p50":  _pct(ttfts, 50),
            "ttft_p95":  _pct(ttfts, 95),
            "ttft_min":  min(ttfts, default=0),
            "ttft_max":  max(ttfts, default=0),
            "e2e_mean":  statistics.mean(e2es) if e2es else 0,
            "e2e_p50":   _pct(e2es, 50),
            "hit_rate":  hits / max(len(ok), 1),
        }

    c = stats(cold_results)
    w = stats(warm_results)

    ttft_reduction = (c["ttft_mean"] - w["ttft_mean"]) / max(c["ttft_mean"], 1) * 100
    p50_reduction  = (c["ttft_p50"]  - w["ttft_p50"])  / max(c["ttft_p50"],  1) * 100
    p95_reduction  = (c["ttft_p95"]  - w["ttft_p95"])  / max(c["ttft_p95"],  1) * 100

    lines = [
        "=" * 64,
        "  Phase 3 — KV Prefill Experiment Summary",
        "=" * 64,
        "",
        f"  {'Metric':<30} {'Cold (no prefill)':>16} {'Warm (prefill)':>16}",
        f"  {'-'*62}",
        f"  {'Requests (ok)':<30} {c['n']:>16} {w['n']:>16}",
        f"  {'TTFT mean (ms)':<30} {c['ttft_mean']:>16.1f} {w['ttft_mean']:>16.1f}",
        f"  {'TTFT p50  (ms)':<30} {c['ttft_p50']:>16.1f} {w['ttft_p50']:>16.1f}",
        f"  {'TTFT p95  (ms)':<30} {c['ttft_p95']:>16.1f} {w['ttft_p95']:>16.1f}",
        f"  {'TTFT min  (ms)':<30} {c['ttft_min']:>16.1f} {w['ttft_min']:>16.1f}",
        f"  {'E2E  p50  (ms)':<30} {c['e2e_p50']:>16.1f} {w['e2e_p50']:>16.1f}",
        f"  {'Cache hit rate':<30} {'—':>16} {w['hit_rate']:>15.1%}",
        f"  {'-'*62}",
        f"  {'TTFT mean reduction':<30} {ttft_reduction:>15.1f}%",
        f"  {'TTFT p50  reduction':<30} {p50_reduction:>15.1f}%",
        f"  {'TTFT p95  reduction':<30} {p95_reduction:>15.1f}%",
        "",
        "  Target compliance (TTFT < 200ms)",
    ]

    for label, results in [("Cold", cold_results), ("Warm", warm_results)]:
        ok = [r for r in results if r.ok and r.e2e_ms > 0]
        within = sum(1 for r in ok if r.ttft_ms < 200)
        if ok:
            lines.append(f"    {label}: {within}/{len(ok)} ({within/len(ok)*100:.0f}%)")

    lines += ["", "=" * 64]
    summary = "\n".join(lines)
    print(f"\n{summary}")

    summary_path = out_dir / "phase3_summary.txt"
    summary_path.write_text(summary)
    print(f"\n  Saved → {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _main(args):
    from workload_gen import (
        generate_autocomplete_workload,
        generate_revision_history_workload,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify server
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{HOST}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                h = await r.json()
                print(f"Server OK — prefill_enabled={h.get('prefill_enabled')}  "
                      f"idle_ms={h.get('prefill_idle_ms')}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {HOST} — {e}")
        return

    if args.workload == "autocomplete":
        wl = [p for p in generate_autocomplete_workload(args.n + 2)]
    else:
        wl = [p for p in generate_revision_history_workload(args.n + 2)]

    print(f"\nWorkload: {args.workload}  |  N={args.n}  |  Warmup=2")
    print("=" * 64)

    # Pass A — cold (idle_ms=0 so the prefill timer never fires before the request)
    print("\n--- Pass A: Cold inference (idle=0ms, prefill has no time to fire) ---")
    cold_results = await run_pass(wl, "COLD", idle_ms=0.0, warmup=2)
    save_csv(cold_results, out_dir / "phase3_prefill_cold.csv")

    # Pass B — warm (idle_ms > PREFILL_IDLE_MS so the prefill always fires)
    print("\n--- Pass B: Warm inference (idle=300ms, prefill fires before request) ---")
    warm_results = await run_pass(wl, "WARM", idle_ms=300.0, warmup=2)
    save_csv(warm_results, out_dir / "phase3_prefill_warm.csv")

    build_summary(cold_results, warm_results, out_dir)

    # Fetch global prefill stats from server
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{HOST}/prefill/stats") as r:
                pstats = await r.json()
                print(f"\nServer prefill stats: {json.dumps(pstats['prefill'], indent=2)}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Phase 3 — KV Prefill Experiment")
    parser.add_argument("--n",        type=int, default=30,
                        help="Requests per pass (excl. warmup)")
    parser.add_argument("--workload", choices=["autocomplete", "revision"],
                        default="autocomplete")
    parser.add_argument("--out-dir",  default="results")
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
