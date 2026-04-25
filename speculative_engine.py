"""
PDC Phase 4 — Speculative Decoding Engine
Implements the draft-target speculative sampling loop described in
Leviathan et al. (2023) "Fast Inference from Transformers via Speculative Decoding".

Architecture:
  - Draft model : TinyLlama-1.1B (fast, small, generates k candidate tokens)
  - Target model: Mistral-7B     (accurate, verifies all k tokens in one pass)

The key insight: the target model processes k+1 tokens in a SINGLE forward pass
(parallel verification), which costs roughly the same as generating 1 token
autoregressively. If most candidates are accepted, we get k tokens for the price
of ~1, improving throughput without changing output distribution.

TTFT impact: speculative decoding primarily improves throughput (tokens/sec),
not TTFT, since the first token still requires one full prefill pass.
This is an important finding to report in the paper.

Usage:
    engine = SpeculativeEngine(
        draft_model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        target_model_path="models/mistral-7b-instruct-v0.2.Q8_0.gguf",
        k=4,
    )
    result = engine.generate(prompt, max_tokens=64)
    print(result.text, result.acceptance_rate)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpecResult:
    text: str = ""
    tokens_generated: int = 0
    tokens_accepted: int = 0       # draft tokens accepted by target
    tokens_rejected: int = 0       # draft tokens rejected (target fallback used)
    acceptance_rate: float = 0.0
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    tpot_ms: float = 0.0           # per output token, post-first-token
    draft_calls: int = 0           # number of draft model forward passes
    target_calls: int = 0          # number of target model forward passes
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def speedup_ratio(self) -> float:
        """
        Theoretical speedup: tokens generated per target forward pass.
        Baseline (no spec decoding) = 1.0
        """
        if self.target_calls == 0:
            return 0.0
        return self.tokens_generated / self.target_calls


# ---------------------------------------------------------------------------
# Speculative Engine
# ---------------------------------------------------------------------------

class SpeculativeEngine:
    """
    Speculative decoding engine.

    Parameters
    ----------
    draft_model_path : path to TinyLlama GGUF
    target_model_path: path to Mistral 7B GGUF
    k                : block size — draft tokens generated per step (experiment: {1, 4, 8})
    n_gpu_layers     : -1 = full GPU offload for both models
    temperature      : sampling temperature (applied at target verification stage)
    """

    def __init__(
        self,
        draft_model_path: str,
        target_model_path: str,
        k: int = 4,
        n_gpu_layers: int = -1,
        n_ctx: int = 2048,
        temperature: float = 0.2,
        verbose: bool = False,
    ):
        from llama_cpp import Llama

        self.k = k
        self.temperature = temperature

        log.info(f"Loading draft model (k={k}): {draft_model_path}")
        self.draft = Llama(
            model_path=draft_model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            logits_all=True,      # needed for token probability access
            verbose=verbose,
        )

        log.info(f"Loading target model: {target_model_path}")
        self.target = Llama(
            model_path=target_model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            logits_all=True,
            verbose=verbose,
        )

        log.info("Speculative engine ready.")

    # ------------------------------------------------------------------
    # Core generation loop
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: Optional[float] = None,
    ) -> SpecResult:
        """
        Generates up to max_tokens using speculative decoding.

        Algorithm per step:
          1. Draft model autoregressively generates k tokens
          2. Target model scores the prompt + k draft tokens in one pass
          3. Accept/reject each draft token using speculative sampling criterion
          4. Append accepted tokens; if any rejected, use target's corrected token
          5. Repeat until max_tokens reached or EOS
        """
        temp = temperature if temperature is not None else self.temperature
        result = SpecResult()
        t_start = time.perf_counter()
        first_token_time: Optional[float] = None

        # Tokenize the prompt
        prompt_tokens = self.target.tokenize(prompt.encode())
        context_tokens: List[int] = list(prompt_tokens)
        generated_tokens: List[int] = []

        try:
            while len(generated_tokens) < max_tokens:
                remaining = max_tokens - len(generated_tokens)
                block_k = min(self.k, remaining)

                # ---- Step 1: Draft generates k tokens ----
                draft_tokens, draft_probs = self._draft_sample(context_tokens, block_k)
                result.draft_calls += 1

                # ---- Step 2: Target scores all k+1 positions in one pass ----
                target_logits = self._target_score(context_tokens, draft_tokens)
                result.target_calls += 1

                # ---- Step 3: Accept / reject ----
                accepted: List[int] = []
                for i, (d_tok, d_prob) in enumerate(zip(draft_tokens, draft_probs)):
                    t_prob = self._softmax_single(target_logits[i], d_tok, temp)

                    # Acceptance criterion: accept if target prob >= draft prob
                    # Otherwise accept with probability t_prob / d_prob
                    if d_prob <= 0:
                        accept = False
                    elif t_prob >= d_prob:
                        accept = True
                    else:
                        accept = np.random.random() < (t_prob / d_prob)

                    if accept:
                        accepted.append(d_tok)
                        result.tokens_accepted += 1
                    else:
                        # Rejection: sample corrected token from adjusted distribution
                        corrected = self._sample_adjusted(target_logits[i], draft_probs[i], d_tok, temp)
                        accepted.append(corrected)
                        result.tokens_rejected += 1
                        break  # Stop at first rejection

                # If all k tokens accepted, also take the target's bonus token
                if len(accepted) == block_k and block_k > 0:
                    bonus = self._sample_from_logits(target_logits[block_k], temp)
                    accepted.append(bonus)

                if not accepted:
                    break

                # ---- Step 4: Extend context ----
                for tok in accepted:
                    if tok == self.target.token_eos():
                        result.tokens_generated = len(generated_tokens)
                        result.e2e_ms = (time.perf_counter() - t_start) * 1000
                        result.text = self._decode(generated_tokens)
                        self._finalise(result, t_start, first_token_time)
                        return result

                    generated_tokens.append(tok)
                    context_tokens.append(tok)

                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        result.ttft_ms = (first_token_time - t_start) * 1000

        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            log.error(f"Speculative generation error: {e}")

        result.tokens_generated = len(generated_tokens)
        result.text = self._decode(generated_tokens)
        self._finalise(result, t_start, first_token_time)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draft_sample(self, context: List[int], k: int):
        """Run draft model autoregressively for k tokens. Returns (tokens, probs)."""
        tokens, probs = [], []
        ctx = list(context)

        for _ in range(k):
            out = self.draft(
                self.draft.detokenize(ctx).decode(errors="replace"),
                max_tokens=1,
                temperature=self.temperature,
                logprobs=1,
                echo=False,
            )
            tok_text = out["choices"][0]["text"]
            logprob   = out["choices"][0].get("logprobs", {})

            # Get the token id and its probability
            tok_ids = self.draft.tokenize(tok_text.encode()) if tok_text else [self.draft.token_eos()]
            tok_id  = tok_ids[0] if tok_ids else self.draft.token_eos()

            # Extract probability from logprobs if available, else use uniform fallback
            top_logprobs = logprob.get("top_logprobs", [{}]) if logprob else [{}]
            tok_prob = np.exp(logprob.get("token_logprobs", [0])[0]) if logprob else 1e-6

            tokens.append(tok_id)
            probs.append(float(tok_prob))
            ctx.append(tok_id)

            if tok_id == self.draft.token_eos():
                break

        return tokens, probs

    def _target_score(self, context: List[int], draft_tokens: List[int]):
        """
        Score context + draft_tokens with target model in ONE forward pass.
        Returns logits for each position in draft_tokens + 1 (bonus position).
        """
        full_sequence = context + draft_tokens
        text = self.target.detokenize(full_sequence).decode(errors="replace")

        out = self.target(
            text,
            max_tokens=1,
            temperature=0.0,
            logprobs=len(draft_tokens) + 1,
            echo=True,
        )

        # Extract per-token logprobs from the response
        # Shape: list of dicts, one per token in the sequence
        logprobs_list = out.get("choices", [{}])[0].get("logprobs", {})
        token_logprobs = logprobs_list.get("token_logprobs", []) if logprobs_list else []

        # Align to draft token positions (last len(draft_tokens)+1 entries)
        n = len(draft_tokens) + 1
        if len(token_logprobs) >= n:
            return token_logprobs[-n:]
        # Fallback: uniform if logprobs unavailable
        return [{}] * n

    def _softmax_single(self, logprob_entry, token_id: int, temperature: float) -> float:
        if not logprob_entry or not isinstance(logprob_entry, dict):
            return 1e-6

        # Find matching token string
        for tok_str, lp in logprob_entry.items():
            tok_ids = self.target.tokenize(tok_str.encode())
            if tok_ids and tok_ids[0] == token_id:
                return float(np.exp(lp))

        return 1e-6

    def _sample_adjusted(self, logprob_entry, draft_prob: float, draft_tok: int, temp: float) -> int:
        """
        Sample from the adjusted distribution max(0, p_target - p_draft) / Z.
        Simplified: sample the highest-probability token from the target distribution.
        """
        return self._sample_from_logits(logprob_entry, temp)

    def _sample_from_logits(self, logprob_entry, temperature: float) -> int:
        """Sample a token id from a logprob dict."""
        if not logprob_entry or not isinstance(logprob_entry, dict):
            return self.target.token_eos()
        # logprob_entry: {token_string: log_prob, ...}
        best = max(logprob_entry, key=logprob_entry.get)
        toks = self.target.tokenize(best.encode())
        return toks[0] if toks else self.target.token_eos()

    def _decode(self, token_ids: List[int]) -> str:
        if not token_ids:
            return ""
        try:
            return self.target.detokenize(token_ids).decode(errors="replace")
        except Exception:
            return ""

    def _finalise(self, result: SpecResult, t_start: float, first_token_time: Optional[float]):
        result.e2e_ms = (time.perf_counter() - t_start) * 1000
        if first_token_time is None:
            result.ttft_ms = result.e2e_ms
        n = result.tokens_generated
        if n > 1 and result.ttft_ms > 0:
            result.tpot_ms = (result.e2e_ms - result.ttft_ms) / (n - 1)
        total = result.tokens_accepted + result.tokens_rejected
        result.acceptance_rate = result.tokens_accepted / total if total > 0 else 0.0
