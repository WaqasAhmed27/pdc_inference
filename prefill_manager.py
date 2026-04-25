"""
PDC Phase 3 — Editor-Aware KV Prefill
Implements a non-invasive prefill that, on a configurable idle timer,
preloads the last N tokens of document context into the model's KV cache.

Design goals:
  - No modification to decoding internals
  - Per-session state (each browser tab / editor session is independent)
  - Thread-safe for concurrent sessions
  - Measurable: exposes hit/miss/overhead counters for benchmarking

Usage (imported by prefill_server.py):
    manager = PrefillManager(llm, n_tokens=32, idle_ms=150)
    manager.update_context(session_id, text)   # call on every keystroke
    hit, prefill_ms = manager.maybe_prefill(session_id, text)  # call before /complete
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-session state
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    session_id: str
    last_context: str = ""          # last text seen from this session
    prefilled_context: str = ""     # text that is currently warm in KV cache
    last_update_ts: float = 0.0     # time of last context update
    prefill_ts: float = 0.0         # time of last successful prefill
    n_prefills: int = 0             # total prefills performed
    n_hits: int = 0                 # requests that benefited from prefill
    n_misses: int = 0               # requests that were cold
    total_prefill_ms: float = 0.0   # cumulative prefill cost
    total_savings_ms: float = 0.0   # estimated TTFT savings


# ---------------------------------------------------------------------------
# PrefillManager
# ---------------------------------------------------------------------------

class PrefillManager:
    """
    Manages editor-aware KV-cache prefetching.

    Algorithm:
        1. Editor calls update_context(session_id, text) on every keystroke/change.
        2. After idle_ms of silence, _schedule_prefill fires and runs a zero-token
           completion (max_tokens=1) to warm the KV cache for the last n_tokens
           of context.
        3. On the real /complete request, maybe_prefill checks if the current
           context is a prefix-match of the warm cache. If yes → cache hit,
           TTFT is reduced. If no → cold miss, normal inference.

    KV cache hit condition:
        request_prompt.startswith(prefilled_context[:last_n_chars])
        where last_n_chars covers roughly the last n_tokens characters.
    """

    def __init__(
        self,
        llm,                        # llama_cpp.Llama instance
        n_tokens: int = 32,         # tokens to prefetch into KV cache
        idle_ms: float = 150.0,     # idle window before prefetch fires (ms)
        enabled: bool = True,       # global on/off switch for A/B experiments
        chars_per_token: int = 4,   # rough chars-per-token for prefix matching
    ):
        self.llm = llm
        self._llm_lock = Lock()
        self.n_tokens = n_tokens
        self.idle_ms = idle_ms
        self.enabled = enabled
        self.chars_per_token = chars_per_token
        self._window = n_tokens * chars_per_token  # char window for prefix check

        self._sessions: Dict[str, SessionState] = {}
        self._timers:   Dict[str, asyncio.TimerHandle] = {}
        self._lock = Lock()

        # Global counters
        self.total_hits   = 0
        self.total_misses = 0
        self.total_prefills = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_context(self, session_id: str, text: str) -> None:
        """
        Called by the editor on every content change.
        Resets the idle timer; a prefill fires after idle_ms of silence.
        """
        if not self.enabled:
            return

        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(session_id=session_id)
            state = self._sessions[session_id]
            state.last_context = text
            state.last_update_ts = time.perf_counter()

        # Reset idle timer
        self._cancel_timer(session_id)
        loop = self._get_loop()
        if loop and loop.is_running():
            handle = loop.call_later(
                self.idle_ms / 1000.0,
                lambda sid=session_id: asyncio.ensure_future(self._do_prefill(sid)),
            )
            with self._lock:
                self._timers[session_id] = handle

    def maybe_prefill(self, session_id: str, prompt: str) -> Tuple[bool, float]:
        """
        Called immediately before /complete.
        Returns (cache_hit: bool, prefill_overhead_ms: float).

        If cache_hit is True the KV cache already contains the context prefix,
        and TTFT should be reduced. prefill_overhead_ms is 0 for hits.
        If cache_hit is False, returns the prefill cost as overhead (or 0 if
        no prefill has run yet for this session).
        """
        if not self.enabled:
            return False, 0.0

        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or not state.prefilled_context:
                self.total_misses += 1
                if state:
                    state.n_misses += 1
                return False, 0.0

            # Check if the prompt starts with the warm prefix
            warm_prefix = state.prefilled_context[-self._window:]
            hit = prompt.endswith(warm_prefix) or warm_prefix in prompt[-self._window:]

            if hit:
                self.total_hits += 1
                state.n_hits += 1
                return True, 0.0
            else:
                self.total_misses += 1
                state.n_misses += 1
                return False, state.total_prefill_ms / max(state.n_prefills, 1)

    def get_session_stats(self, session_id: str) -> Optional[dict]:
        with self._lock:
            state = self._sessions.get(session_id)
            if not state:
                return None
            return {
                "session_id":       session_id,
                "n_prefills":       state.n_prefills,
                "n_hits":           state.n_hits,
                "n_misses":         state.n_misses,
                "hit_rate":         state.n_hits / max(state.n_hits + state.n_misses, 1),
                "avg_prefill_ms":   round(state.total_prefill_ms / max(state.n_prefills, 1), 2),
                "prefilled_chars":  len(state.prefilled_context),
            }

    def get_global_stats(self) -> dict:
        return {
            "enabled":        self.enabled,
            "n_tokens":       self.n_tokens,
            "idle_ms":        self.idle_ms,
            "total_prefills": self.total_prefills,
            "total_hits":     self.total_hits,
            "total_misses":   self.total_misses,
            "hit_rate":       round(
                self.total_hits / max(self.total_hits + self.total_misses, 1), 3
            ),
        }

    def flush_session(self, session_id: str) -> None:
        """Remove a session (e.g. on tab close)."""
        self._cancel_timer(session_id)
        with self._lock:
            self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
# In prefill_manager.py — don't reset in _run_llm_safe
    def _run_llm_safe(self, prompt: str, **kwargs):
        with self._llm_lock:
            # NO reset here — preserve KV cache state from prefill
            return self.llm(prompt, **kwargs)

    async def _do_prefill(self, session_id: str) -> None:
        """
        Fires after idle_ms. Runs a 1-token completion to warm the KV cache.
        The actual KV state is managed by llama.cpp internally — we just
        need to process the context through the model once.
        """
        with self._lock:
            state = self._sessions.get(session_id)
            if not state or not state.last_context:
                return
            context = state.last_context

        # Extract last n_tokens worth of text
        tail = context[-self.n_tokens * self.chars_per_token:]
        if not tail.strip():
            return

        start = time.perf_counter()
        try:
            # Run the blocking llm() call in a thread so we don't block the event loop.
            # max_tokens=1 forces a full prefill pass through the model without generating output.
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._run_llm_safe(tail),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000

            with self._lock:
                if session_id in self._sessions:
                    s = self._sessions[session_id]
                    s.prefilled_context = tail
                    s.prefill_ts = time.perf_counter()
                    s.n_prefills += 1
                    s.total_prefill_ms += elapsed_ms
                    self.total_prefills += 1

            log.debug(
                f"[prefill] session={session_id[:8]} "
                f"chars={len(tail)} elapsed={elapsed_ms:.1f}ms"
            )

        except Exception as e:
            log.warning(f"[prefill] session={session_id[:8]} failed: {e}")

    def _cancel_timer(self, session_id: str) -> None:
        with self._lock:
            handle = self._timers.pop(session_id, None)
        if handle:
            handle.cancel()

    @staticmethod
    def _get_loop() -> Optional[asyncio.AbstractEventLoop]:
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            return None
