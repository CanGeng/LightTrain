"""Evaluation metrics — perplexity / EM / ROUGE / BLEU / lm-eval-harness hook.

All metric functions follow the pattern:
    result = metric_fn(predictions, references, ...)  -> scalar or dict

``perplexity`` takes a model + dataloader.
``lm_eval_harness_hook`` is opt-in (requires the ``lm-eval`` package).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


def perplexity(
    model: Any,
    dataloader: Iterable[Any],
    *,
    device: torch.device | None = None,
    ignore_index: int = -100,
    max_batches: int | None = None,
) -> float:
    """Compute perplexity on a dataloader.

    Parameters
    ----------
    model :
        Model with ``forward(**batch) -> ModelOutput`` returning ``logits``.
    dataloader :
        Yields batches with ``input_ids`` and ``labels``.
    ignore_index :
        Label value excluded from NLL computation.
    max_batches :
        Stop after this many batches (quick smoke-test mode).

    Returns
    -------
    float — perplexity = exp(mean NLL)
    """
    from lighttrain.protocols import ModelOutput

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if max_batches is not None and n_batches >= max_batches:
                break
            if device is not None:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            out = model(**batch)
            if isinstance(out, ModelOutput):
                logits = out.outputs["logits"]
            elif isinstance(out, dict):
                logits = out["logits"]
            else:
                logits = out

            labels = batch.get("labels")
            if labels is None:
                n_batches += 1
                continue

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1).long(),
                ignore_index=ignore_index,
                reduction="sum",
            )
            n_tokens = (shift_labels != ignore_index).sum().item()
            total_nll += float(loss.detach())
            total_tokens += int(n_tokens)
            n_batches += 1

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


# ---------------------------------------------------------------------------
# Exact match
# ---------------------------------------------------------------------------


def exact_match(
    predictions: list[str],
    references: list[str],
    *,
    normalize: bool = True,
) -> float:
    """Exact-match accuracy.

    Parameters
    ----------
    normalize :
        Strip whitespace and lower-case before comparison.
    """
    if not predictions:
        return 0.0

    def _norm(s: str) -> str:
        return s.strip().lower() if normalize else s

    return sum(_norm(p) == _norm(r) for p, r in zip(predictions, references)) / len(predictions)


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (
                dp[i - 1][j - 1] + 1
                if a[i - 1] == b[j - 1]
                else max(dp[i - 1][j], dp[i][j - 1])
            )
    return dp[m][n]


def rouge_score(
    predictions: list[str],
    references: list[str],
    *,
    variant: str = "rougeL",
) -> dict[str, float]:
    """Corpus-level ROUGE-N or ROUGE-L.

    Parameters
    ----------
    variant : ``"rouge1"``, ``"rouge2"``, or ``"rougeL"``

    Returns
    -------
    dict with ``precision``, ``recall``, ``f1`` keys.
    """
    if not predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    variant_key = variant.lower()
    p_sum = r_sum = 0.0

    for pred, ref in zip(predictions, references):
        pt = _tokenize(pred)
        rt = _tokenize(ref)

        if variant_key in ("rouge1", "rouge2"):
            n = 1 if variant_key == "rouge1" else 2
            pred_ng = _ngrams(pt, n)
            ref_ng = _ngrams(rt, n)
            overlap = sum((pred_ng & ref_ng).values())
            p = overlap / max(1, sum(pred_ng.values()))
            r = overlap / max(1, sum(ref_ng.values()))
        elif variant_key == "rougel":
            lcs = _lcs_length(pt, rt)
            p = lcs / max(1, len(pt))
            r = lcs / max(1, len(rt))
        else:
            raise ValueError(
                f"rouge_score: unknown variant {variant!r}. Use rouge1/rouge2/rougeL."
            )

        p_sum += p
        r_sum += r

    n_s = len(predictions)
    precision = p_sum / n_s
    recall = r_sum / n_s
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------


def _modified_precision(pt: list[str], rt: list[str], n: int) -> tuple[int, int]:
    pred_ng = _ngrams(pt, n)
    ref_ng = _ngrams(rt, n)
    return sum((pred_ng & ref_ng).values()), sum(pred_ng.values())


def bleu_score(
    predictions: list[str],
    references: list[str],
    *,
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """Corpus-level BLEU score (up to 4-gram by default)."""
    if not predictions:
        return 0.0

    total_ref_len = total_pred_len = 0
    counts = [0] * max_n
    totals = [0] * max_n

    for pred, ref in zip(predictions, references):
        pt = _tokenize(pred)
        rt = _tokenize(ref)
        total_pred_len += len(pt)
        total_ref_len += len(rt)
        for n in range(1, max_n + 1):
            c, t = _modified_precision(pt, rt, n)
            if smooth:
                c += 1
                t += 1
            counts[n - 1] += c
            totals[n - 1] += t

    bp = (
        1.0
        if total_pred_len >= total_ref_len
        else math.exp(1 - total_ref_len / max(1, total_pred_len))
    )

    log_sum = sum(
        math.log(max(counts[n] / max(1, totals[n]), 1e-10)) / max_n
        for n in range(max_n)
    )
    return bp * math.exp(log_sum)


# ---------------------------------------------------------------------------
# lm-eval-harness hook (opt-in)
# ---------------------------------------------------------------------------


def lm_eval_harness_hook(
    task_name: str,
    model: Any,
    tokenizer: Any,
    *,
    num_fewshot: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run a task via the ``lm_eval`` harness.

    Requires ``pip install lm-eval``.
    """
    try:
        import lm_eval  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "lm_eval_harness_hook requires the lm-eval package. "
            "Install with: pip install lm-eval"
        ) from exc

    results = lm_eval.simple_evaluate(
        model=model,
        tasks=[task_name],
        num_fewshot=num_fewshot,
        limit=limit,
    )
    return results.get("results", {}).get(task_name, {})


__all__ = [
    "bleu_score",
    "exact_match",
    "lm_eval_harness_hook",
    "perplexity",
    "rouge_score",
]
