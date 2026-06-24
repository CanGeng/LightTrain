"""Judge implementations — verifier / pairwise-LLM.

A Judge scores (prompt, response) pairs. The :class:`JudgeProtocol` is
defined in ``lighttrain.protocols``; implementations here are registered in
the ``"judge"`` registry category.

Multi-model composition is a ``CompositeJudge`` plugin, not part of core.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from typing import Any

from lighttrain.registry import register

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verifier judge — symbolic / regex check
# ---------------------------------------------------------------------------


@register("judge", "verifier")
class VerifierJudge:
    """Code/math verifier judge using a symbolic check function.

    Parameters
    ----------
    verify_fn :
        ``(prompt: str, response: str) -> float`` callable.  Return 1.0 for
        correct, 0.0 for incorrect, or a continuous value in [0, 1].
    answer_pattern :
        If ``verify_fn`` is not provided, a regex pattern is used to extract
        the answer from the response. Patterns like ``r"#### (\\d+)"`` work for
        GSM8K-style answers.
    reference_key :
        Key in the extras dict from which the ground-truth answer is read (when
        using the pattern-based mode).
    """

    reward_kind = "pointwise"  # → PointwiseRewardAdapter when used as an RL reward

    def __init__(
        self,
        *,
        verify_fn: Callable[[str, str], float] | None = None,
        answer_pattern: str | None = r"#### (\S+)",
        verify_pattern: str | None = None,  # alias for answer_pattern (RL recipe form)
        reference_key: str = "answer",
    ) -> None:
        self.verify_fn = verify_fn
        if verify_pattern is not None:
            answer_pattern = verify_pattern
        self.answer_pattern = re.compile(answer_pattern) if answer_pattern else None
        self.reference_key = reference_key

    def score(
        self,
        items: Iterable[Any],
        ctx: Any | None = None,
    ) -> list[float]:
        """Score a list of (prompt, response[, extras]) tuples.

        Each item can be:
        - ``(prompt_str, response_str)``
        - ``(prompt_str, response_str, extras_dict)``
        """
        scores: list[float] = []
        for item in items:
            if len(item) == 2:
                prompt, response = item
                extras: dict[str, Any] = {}
            else:
                prompt, response, extras = item

            if self.verify_fn is not None:
                s = float(self.verify_fn(str(prompt), str(response)))
            elif self.answer_pattern is not None and self.reference_key in extras:
                ref = str(extras[self.reference_key])
                m = self.answer_pattern.search(str(response))
                pred = m.group(1) if m else ""
                ref_m = self.answer_pattern.search(ref)
                ref_val = ref_m.group(1) if ref_m else ref
                s = 1.0 if pred.strip() == ref_val.strip() else 0.0
            elif self.answer_pattern is not None:
                # pure response regex mode — used by RL rollout (no extras/reference)
                s = 1.0 if self.answer_pattern.search(str(response)) else 0.0
            else:
                s = 0.0
            scores.append(s)
        return scores


# ---------------------------------------------------------------------------
# Pairwise LLM judge
# ---------------------------------------------------------------------------


@register("judge", "pairwise_llm")
class PairwiseLLMJudge:
    """Pairwise judge that calls an LLM API to compare two responses.

    Sends a structured prompt to ``judge_model_fn`` and parses the winner.

    Parameters
    ----------
    judge_model_fn :
        ``(prompt: str) -> str`` callable. Typically wraps an API call
        (e.g. ``openai.chat.completions.create``).
    prompt_template :
        Format string with ``{question}``, ``{response_a}``, ``{response_b}``.
    win_pattern :
        Regex to extract the winner label (``"A"`` or ``"B"``) from the judge
        response. Default matches ``"Response A"`` / ``"Response B"``.
    """

    _DEFAULT_TEMPLATE = (
        "Given the following question, compare Response A and Response B.\n\n"
        "Question: {question}\n\n"
        "Response A: {response_a}\n\n"
        "Response B: {response_b}\n\n"
        "Which response is better? Answer with exactly 'Response A' or 'Response B'."
    )

    reward_kind = "pairwise"  # needs a registered pairwise reward_adapter (deferred)

    def __init__(
        self,
        judge_model_fn: Callable[[str], str],
        *,
        prompt_template: str | None = None,
        win_pattern: str = r"Response ([AB])",
    ) -> None:
        self.judge_model_fn = judge_model_fn
        self.prompt_template = prompt_template or self._DEFAULT_TEMPLATE
        self.win_pattern = re.compile(win_pattern, re.IGNORECASE)

    def score(
        self,
        items: Iterable[Any],
        ctx: Any | None = None,
    ) -> list[float]:
        """Score a list of (question, response_a, response_b) tuples.

        Returns 1.0 if response_a wins, 0.0 if response_b wins, 0.5 for tie/parse failure.
        """
        scores: list[float] = []
        for item in items:
            question, resp_a, resp_b = item[:3]
            judge_prompt = self.prompt_template.format(
                question=question, response_a=resp_a, response_b=resp_b
            )
            try:
                output = self.judge_model_fn(judge_prompt)
                m = self.win_pattern.search(output)
                if m:
                    winner = m.group(1).upper()
                    scores.append(1.0 if winner == "A" else 0.0)
                else:
                    scores.append(0.5)
            except Exception:  # noqa: BLE001
                _log.warning("judge: pairwise scoring failed for an item; scoring as tie (0.5)", exc_info=True)
                scores.append(0.5)
        return scores


__all__ = ["PairwiseLLMJudge", "VerifierJudge"]
