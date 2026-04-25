"""
PDC Phase 3 - Editor-aware KV prefill manager.

This version uses explicit llama-cpp-python state snapshots:
  - idle prefill path: tokenize -> eval -> save_state
  - request path: load_state before generation when prompt prefix matches
"""

import asyncio
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional, Sequence, Tuple

log = logging.getLogger(__name__)


@dataclass
class SessionState:
    session_id: str
    last_context: str = ""
    prefilled_prompt: str = ""
    prefilled_tokens: Tuple[int, ...] = field(default_factory=tuple)
    prefilled_state: object = None
    last_update_ts: float = 0.0
    prefill_ts: float = 0.0
    n_prefills: int = 0
    n_hits: int = 0
    n_misses: int = 0
    total_prefill_ms: float = 0.0
    total_load_state_ms: float = 0.0


class PrefillManager:
    def __init__(
        self,
        llm,
        n_tokens: int = 0,
        idle_ms: float = 150.0,
        enabled: bool = True,
    ):
        self.llm = llm
        self.n_tokens = n_tokens
        self.idle_ms = idle_ms
        self.enabled = enabled

        self._sessions: Dict[str, SessionState] = {}
        self._timers: Dict[str, asyncio.TimerHandle] = {}
        self._lock = Lock()
        self._model_lock = Lock()

        self.total_hits = 0
        self.total_misses = 0
        self.total_prefills = 0

    @contextmanager
    def model_lock(self):
        self._model_lock.acquire()
        try:
            yield
        finally:
            self._model_lock.release()

    def update_context(self, session_id: str, text: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            state = self._sessions.setdefault(session_id, SessionState(session_id=session_id))
            state.last_context = text
            state.last_update_ts = time.perf_counter()
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
        with self.model_lock():
            return self.maybe_prefill_locked(session_id, prompt)

    def maybe_prefill_locked(self, session_id: str, prompt: str) -> Tuple[bool, float]:
        if not self.enabled:
            return False, 0.0
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.prefilled_state is None or not state.prefilled_tokens:
                self.total_misses += 1
                if state:
                    state.n_misses += 1
                return False, 0.0
        try:
            prompt_tokens = self._tokenize(prompt)
            with self._lock:
                expected = state.prefilled_tokens
            if len(prompt_tokens) < len(expected) or tuple(prompt_tokens[: len(expected)]) != expected:
                with self._lock:
                    self.total_misses += 1
                    state.n_misses += 1
                    avg_prefill = state.total_prefill_ms / max(state.n_prefills, 1)
                return False, avg_prefill

            t0 = time.perf_counter()
            self.llm.load_state(state.prefilled_state)
            load_ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self.total_hits += 1
                state.n_hits += 1
                state.total_load_state_ms += load_ms
            return True, load_ms
        except Exception as e:
            log.warning(f"[prefill] load_state failed for {session_id[:8]}: {e}")
            with self._lock:
                self.total_misses += 1
                state.n_misses += 1
            return False, 0.0

    def get_session_stats(self, session_id: str) -> Optional[dict]:
        with self._lock:
            state = self._sessions.get(session_id)
            if not state:
                return None
            return {
                "session_id": session_id,
                "n_prefills": state.n_prefills,
                "n_hits": state.n_hits,
                "n_misses": state.n_misses,
                "hit_rate": state.n_hits / max(state.n_hits + state.n_misses, 1),
                "avg_prefill_ms": round(state.total_prefill_ms / max(state.n_prefills, 1), 2),
                "avg_load_state_ms": round(state.total_load_state_ms / max(state.n_hits, 1), 2),
                "prefilled_prompt_chars": len(state.prefilled_prompt),
                "prefilled_tokens": len(state.prefilled_tokens),
            }

    def get_global_stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "n_tokens": self.n_tokens,
            "idle_ms": self.idle_ms,
            "total_prefills": self.total_prefills,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": round(self.total_hits / max(self.total_hits + self.total_misses, 1), 3),
        }

    def flush_session(self, session_id: str) -> None:
        self._cancel_timer(session_id)
        with self._lock:
            self._sessions.pop(session_id, None)

    async def _do_prefill(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.get(session_id)
            if not state or not state.last_context:
                return
            prompt = state.last_context
        if not prompt.strip():
            return

        start = time.perf_counter()
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._prefill_sync, prompt)
            if result is None:
                return
            prefilled_prompt, prompt_tokens, snapshot = result
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            with self._lock:
                s = self._sessions.get(session_id)
                if not s:
                    return
                s.prefilled_prompt = prefilled_prompt
                s.prefilled_tokens = tuple(prompt_tokens)
                s.prefilled_state = snapshot
                s.prefill_ts = time.perf_counter()
                s.n_prefills += 1
                s.total_prefill_ms += elapsed_ms
                self.total_prefills += 1
            log.debug(
                f"[prefill] session={session_id[:8]} tokens={len(prompt_tokens)} "
                f"elapsed={elapsed_ms:.1f}ms"
            )
        except Exception as e:
            log.warning(f"[prefill] session={session_id[:8]} failed: {e}")

    def _prefill_sync(self, prompt: str):
        with self.model_lock():
            tokens = self._tokenize(prompt)
            if not tokens:
                return None
            if self.n_tokens > 0:
                tokens = tokens[: self.n_tokens]
            token_bytes = self.llm.detokenize(tokens, special=True)
            token_prompt = token_bytes.decode("utf-8", errors="ignore")
            self.llm.reset()
            self.llm.eval(tokens)
            snapshot = self.llm.save_state()
            return token_prompt, tokens, snapshot

    def _tokenize(self, text: str) -> Sequence[int]:
        return self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)

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
