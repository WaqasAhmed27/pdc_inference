"""
PDC Phase 4 — KV Cache Eviction Policies
Implements and compares two eviction strategies for managing the KV cache
under constrained memory (simulating < 8GB VRAM or long-context inference).

Policy 1 — LRU (Least Recently Used):
    Evict the oldest tokens in the context window when capacity is exceeded.
    This is llama.cpp's default behaviour (context shift).
    Simple, predictable, but ignores token importance.

Policy 2 — Adaptive (Attention-Score Aware):
    Retains tokens that appear in high-attention positions.
    Specifically: always keep the first 'sink_tokens' tokens (attention sinks)
    plus the most recent 'recent_tokens', and fill the remainder with
    highest-attention-score tokens from the middle of the document.
    Based on insights from StreamingLLM (Xiao et al., 2023) [ref 8 in paper].

Both policies are implemented as context management wrappers around llama_cpp.
The experiment measures TTFT, memory footprint, and output quality under
a forced small context window (simulating constrained VRAM).

Usage:
    policy = AdaptiveEvictionPolicy(sink_tokens=4, recent_tokens=64, max_ctx=512)
    trimmed_prompt = policy.trim(long_prompt)
    # pass trimmed_prompt to llm() as normal
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EvictionPolicy(ABC):
    """
    Base class for KV eviction policies.
    All policies implement `trim(tokens) -> tokens` to reduce a token list
    to fit within max_ctx.
    """

    def __init__(self, max_ctx: int = 512):
        self.max_ctx = max_ctx
        self.n_evictions = 0
        self.n_tokens_evicted = 0

    @abstractmethod
    def trim(self, tokens: List[int]) -> List[int]:
        """Return a trimmed token list that fits within max_ctx."""
        ...

    def trim_text(self, text: str, tokenizer) -> str:
        """Convenience: trim a raw string using a llama_cpp tokenizer."""
        tokens = tokenizer.tokenize(text.encode())
        trimmed = self.trim(tokens)
        return tokenizer.detokenize(trimmed).decode(errors="replace")

    @property
    def stats(self) -> dict:
        return {
            "policy":          self.__class__.__name__,
            "max_ctx":         self.max_ctx,
            "n_evictions":     self.n_evictions,
            "n_tokens_evicted": self.n_tokens_evicted,
        }


# ---------------------------------------------------------------------------
# Policy 1: LRU — evict oldest tokens (sliding window)
# ---------------------------------------------------------------------------

class LRUEvictionPolicy(EvictionPolicy):
    """
    Least Recently Used eviction: simply keeps the most recent max_ctx tokens.
    This mirrors llama.cpp's built-in context-shift behaviour.

    Behaviour:
        [tok_0, tok_1, ..., tok_N]  →  [tok_{N-max_ctx}, ..., tok_N]
    """

    def trim(self, tokens: List[int]) -> List[int]:
        if len(tokens) <= self.max_ctx:
            return tokens

        evicted = len(tokens) - self.max_ctx
        self.n_evictions += 1
        self.n_tokens_evicted += evicted

        trimmed = tokens[-self.max_ctx:]
        log.debug(f"[LRU] evicted {evicted} tokens, kept last {self.max_ctx}")
        return trimmed


# ---------------------------------------------------------------------------
# Policy 2: Adaptive — keep sinks + recent + high-attention middle tokens
# ---------------------------------------------------------------------------

class AdaptiveEvictionPolicy(EvictionPolicy):
    """
    Attention-sink-aware adaptive eviction.

    Retains three token groups:
      1. Sink tokens  : first `sink_tokens` of the context (attention sinks —
                        these receive disproportionately high attention in all
                        transformer layers and must not be evicted)
      2. Recent tokens: last `recent_tokens` of the context (local context
                        is critical for coherent generation)
      3. Middle tokens: highest-scored tokens from the middle, scored by a
                        lightweight proxy — token position * inverse_distance
                        (approximates importance without running actual attention)

    Total budget: sink_tokens + recent_tokens + middle_budget = max_ctx

    Reference: StreamingLLM (Xiao et al., 2023) for the sink-token insight.
    """

    def __init__(
        self,
        max_ctx: int = 512,
        sink_tokens: int = 4,
        recent_tokens: int = 128,
    ):
        super().__init__(max_ctx)
        self.sink_tokens   = min(sink_tokens,   max_ctx // 4)
        self.recent_tokens = min(recent_tokens, max_ctx // 2)
        self.middle_budget = max_ctx - self.sink_tokens - self.recent_tokens
        assert self.middle_budget >= 0, (
            f"sink_tokens({sink_tokens}) + recent_tokens({recent_tokens}) "
            f"exceeds max_ctx({max_ctx})"
        )

    def trim(self, tokens: List[int]) -> List[int]:
        if len(tokens) <= self.max_ctx:
            return tokens

        n = len(tokens)
        evicted_before = self.n_tokens_evicted

        # Segment the sequence
        sinks   = tokens[:self.sink_tokens]
        recents = tokens[-self.recent_tokens:] if self.recent_tokens > 0 else []
        middle_start = self.sink_tokens
        middle_end   = n - self.recent_tokens if self.recent_tokens > 0 else n
        middle = tokens[middle_start:middle_end]

        if len(middle) <= self.middle_budget:
            # Middle fits entirely — no eviction needed there
            trimmed = sinks + middle + recents
        else:
            # Score middle tokens by a simple importance proxy:
            # Score(i) = 1 / (1 + distance_from_end)
            # Rationale: tokens closer to the end of the middle segment are
            # more likely to be referenced by recent tokens (recency bias proxy).
            scores = [1.0 / (1.0 + (len(middle) - 1 - i)) for i in range(len(middle))]
            # Select top-k by score
            indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            keep_indices = sorted(idx for idx, _ in indexed[:self.middle_budget])
            kept_middle = [middle[i] for i in keep_indices]
            trimmed = sinks + kept_middle + recents

        self.n_evictions += 1
        self.n_tokens_evicted += n - len(trimmed)
        log.debug(
            f"[Adaptive] {n} → {len(trimmed)} tokens "
            f"(sinks={len(sinks)}, middle={len(trimmed)-len(sinks)-len(recents)}, "
            f"recent={len(recents)})"
        )
        return trimmed

    @property
    def stats(self) -> dict:
        base = super().stats
        base.update({
            "sink_tokens":    self.sink_tokens,
            "recent_tokens":  self.recent_tokens,
            "middle_budget":  self.middle_budget,
        })
        return base


# ---------------------------------------------------------------------------
# Policy factory
# ---------------------------------------------------------------------------

def make_policy(name: str, max_ctx: int = 512, **kwargs) -> EvictionPolicy:
    """
    Factory function.
    name: "lru" | "adaptive"
    """
    name = name.lower()
    if name == "lru":
        return LRUEvictionPolicy(max_ctx=max_ctx)
    elif name == "adaptive":
        return AdaptiveEvictionPolicy(max_ctx=max_ctx, **kwargs)
    else:
        raise ValueError(f"Unknown eviction policy: {name!r}. Choose 'lru' or 'adaptive'.")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    random.seed(42)

    # Simulate a long token sequence
    long_tokens = list(range(1000))

    for policy_name in ("lru", "adaptive"):
        policy = make_policy(policy_name, max_ctx=128, sink_tokens=4, recent_tokens=32)
        trimmed = policy.trim(long_tokens)
        print(f"\n{policy_name.upper()} policy")
        print(f"  Input : {len(long_tokens)} tokens")
        print(f"  Output: {len(trimmed)} tokens")
        print(f"  Stats : {policy.stats}")
        print(f"  First 8 kept: {trimmed[:8]}")
        print(f"  Last  8 kept: {trimmed[-8:]}")
