"""
PDC Phase 4 - Speculative Decoding Engine
Implements draft-target speculative decoding.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass
class SpecResult:
    text: str = ""
    tokens_generated: int = 0
    tokens_accepted: int = 0
    tokens_rejected: int = 0
    acceptance_rate: float = 0.0
    ttft_ms: float = 0.0
    e2e_ms: float = 0.0
    tpot_ms: float = 0.0
    draft_calls: int = 0
    target_calls: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def speedup_ratio(self) -> float:
        if self.target_calls == 0:
            return 0.0
        return self.tokens_generated / self.target_calls


class SpeculativeEngine:
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
            logits_all=True,
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

        probe = b"Spec decode tokenizer probe."
        if list(self.draft.tokenize(probe)) != list(self.target.tokenize(probe)):
            raise ValueError(
                "Draft/target tokenizers are not compatible. "
                "Speculative engine requires identical token ids across both models."
            )

        log.info("Speculative engine ready.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: Optional[float] = None,
    ) -> SpecResult:
        _ = temperature if temperature is not None else self.temperature
        result = SpecResult()
        t_start = time.perf_counter()
        first_token_time: Optional[float] = None

        prompt_tokens = self.target.tokenize(prompt.encode())
        context_tokens: List[int] = list(prompt_tokens)
        generated_tokens: List[int] = []

        try:
            while len(generated_tokens) < max_tokens:
                remaining = max_tokens - len(generated_tokens)
                block_k = min(self.k, remaining)

                draft_tokens = self._draft_sample(context_tokens, block_k)
                result.draft_calls += 1
                if not draft_tokens:
                    break

                target_tokens = self._target_predict(context_tokens, len(draft_tokens) + 1)
                result.target_calls += 1
                if not target_tokens:
                    break

                accepted: List[int] = []
                for i, d_tok in enumerate(draft_tokens):
                    t_tok = target_tokens[i] if i < len(target_tokens) else self.target.token_eos()
                    if t_tok == d_tok:
                        accepted.append(d_tok)
                        result.tokens_accepted += 1
                    else:
                        accepted.append(t_tok)
                        result.tokens_rejected += 1
                        break

                if len(accepted) == len(draft_tokens) and len(target_tokens) > len(draft_tokens):
                    bonus = target_tokens[len(draft_tokens)]
                    accepted.append(bonus)

                if not accepted:
                    break

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

    def _draft_sample(self, context: List[int], k: int) -> List[int]:
        tokens: List[int] = []
        gen = self.draft.generate(
            context,
            reset=True,
            temp=0.0,
            top_p=1.0,
            min_p=0.0,
        )
        for _ in range(k):
            try:
                tok = int(next(gen))
            except StopIteration:
                break
            tokens.append(tok)
            if tok == self.draft.token_eos():
                break
        return tokens

    def _target_predict(self, context: List[int], n_tokens: int) -> List[int]:
        tokens: List[int] = []
        gen = self.target.generate(
            context,
            reset=True,
            temp=0.0,
            top_p=1.0,
            min_p=0.0,
        )
        for _ in range(n_tokens):
            try:
                tok = int(next(gen))
            except StopIteration:
                break
            tokens.append(tok)
            if tok == self.target.token_eos():
                break
        return tokens

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
