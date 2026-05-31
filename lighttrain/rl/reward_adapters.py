"""Judge → RL reward-fn adapters (registry category ``reward_adapter``).

Replaces the hardcoded ``isinstance(judge, VerifierJudge)`` whitelist in the
runtime: a judge declares its ``reward_kind`` (``"pointwise"`` by default) and
the runtime resolves the matching adapter, which wraps the judge + tokenizer
into the ``reward_fn(prompt_ids, response_ids) -> list[float]`` shape the RL
trainers consume. Any registered pointwise judge can now back an RL reward.

Only the ``pointwise`` adapter ships here (bit-identical to the old inline
``_reward_fn``). A ``pairwise`` adapter (turning a pairwise judge into a
pointwise reward, e.g. via within-group win-rate) is a deliberately deferred
*new feature* — the seam is open (register one under ``"pairwise"``) but it
invents a reward scheme with no bit-check baseline, so it is not part of the
audit-fix batch.
"""

from __future__ import annotations

from typing import Any

from ..registry import register


def _decode_batch(tokenizer: Any, ids_batch: Any) -> list[str]:
    return [
        tokenizer.decode(ids.tolist(), skip_special_tokens=True) for ids in ids_batch
    ]


@register("reward_adapter", "pointwise")
class PointwiseRewardAdapter:
    """Wrap a pointwise judge (``score([(prompt, response), ...]) -> [float]``)
    into an RL ``reward_fn``. Bit-identical to the previous inline wrapper."""

    reward_kind = "pointwise"

    def __init__(self, *, judge: Any, tokenizer: Any) -> None:
        self._judge = judge
        self._tok = tokenizer

    def __call__(self, prompt_ids: Any, response_ids: Any) -> list[float]:
        prompts = _decode_batch(self._tok, prompt_ids)
        responses = _decode_batch(self._tok, response_ids)
        return self._judge.score(list(zip(prompts, responses)))


__all__ = ["PointwiseRewardAdapter"]
