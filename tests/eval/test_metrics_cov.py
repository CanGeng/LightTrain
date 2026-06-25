"""Additional edge-case tests for ``lighttrain.eval.metrics``.

Pins every previously-uncovered branch:

* perplexity — dict-output model (line 69-70), raw-tensor model (line 72),
  batch without labels → skip (lines 76-77), zero-token → inf (line 93)
* exact_match — empty predictions → 0.0 (line 116)
* _lcs_length — empty sequence → 0 (line 141)
* rouge_score — empty → zero dict (line 170), rouge1 variant (lines 180-185),
  rouge2 variant (line 180, n=2), unknown variant → ValueError (line 191)
* bleu_score — empty → 0.0 (line 226)
* lm_eval_harness_hook — ImportError path (lines 275-278),
  successful call path (lines 283-289)
"""

from __future__ import annotations

import math
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lighttrain.eval.metrics import (
    bleu_score,
    exact_match,
    lm_eval_harness_hook,
    perplexity,
    rouge_score,
)
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# Stub model helpers
# ---------------------------------------------------------------------------

class _DictOutputLM(nn.Module):
    """Model whose forward() returns a plain dict with 'logits' key (line 69-70)."""

    def __init__(self, V: int = 8, D: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        h = self.emb(input_ids)
        return {"logits": self.proj(h)}


class _TensorOutputLM(nn.Module):
    """Model whose forward() returns the logits tensor directly (line 72)."""

    def __init__(self, V: int = 8, D: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        h = self.emb(input_ids)
        return self.proj(h)


class _ModelOutputLM(nn.Module):
    """Model returning ModelOutput (existing path; used for no-labels test)."""

    def __init__(self, V: int = 8, D: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, **batch):
        h = self.emb(batch["input_ids"])
        return ModelOutput(outputs={"logits": self.proj(h)})


def _make_batch(B: int = 2, T: int = 6, V: int = 8):
    ids = torch.randint(0, V, (B, T))
    return {"input_ids": ids, "labels": ids.clone()}


def _make_batch_no_labels(B: int = 2, T: int = 6, V: int = 8):
    ids = torch.randint(0, V, (B, T))
    return {"input_ids": ids}


# ---------------------------------------------------------------------------
# perplexity — dict output model (lines 69-70)
# ---------------------------------------------------------------------------

def test_invariant_perplexity_dict_output_model():
    """perplexity works when the model returns a plain dict with 'logits'."""
    torch.manual_seed(42)
    model = _DictOutputLM()
    loader = [_make_batch() for _ in range(2)]
    ppl = perplexity(model, loader)
    assert math.isfinite(ppl) and ppl > 1.0


# ---------------------------------------------------------------------------
# perplexity — raw tensor output model (line 72)
# ---------------------------------------------------------------------------

def test_invariant_perplexity_tensor_output_model():
    """perplexity works when the model returns a raw logits tensor (line 72)."""
    torch.manual_seed(0)
    model = _TensorOutputLM()
    loader = [_make_batch() for _ in range(2)]
    ppl = perplexity(model, loader)
    assert math.isfinite(ppl) and ppl > 1.0


# ---------------------------------------------------------------------------
# perplexity — batch without labels skipped (lines 76-77)
# ---------------------------------------------------------------------------

def test_invariant_perplexity_skips_batches_without_labels():
    """Batches lacking 'labels' are counted but skipped for NLL; if ALL batches
    are label-free the result is float('inf') (lines 76-77 + 93)."""
    torch.manual_seed(7)
    model = _DictOutputLM()
    loader = [_make_batch_no_labels() for _ in range(3)]
    ppl = perplexity(model, loader)
    assert ppl == float("inf")


def test_invariant_perplexity_mixed_batches_skips_no_label():
    """When some batches have labels and some do not, only labeled ones
    contribute to NLL (lines 76-77); result remains finite."""
    torch.manual_seed(11)
    model = _DictOutputLM()
    labeled = _make_batch()
    unlabeled = _make_batch_no_labels()
    loader = [labeled, unlabeled, labeled]
    ppl = perplexity(model, loader)
    assert math.isfinite(ppl) and ppl > 1.0


# ---------------------------------------------------------------------------
# perplexity — zero tokens → inf (line 93)
# ---------------------------------------------------------------------------

def test_pin_current_behavior_perplexity_zero_tokens_returns_inf():
    """Pin: when every token is the ignore_index (no real tokens counted),
    total_tokens == 0 and perplexity returns float('inf') (line 93).

    Note: this is the designed sentinel, not a bug.
    """
    torch.manual_seed(3)
    model = _DictOutputLM(V=8)

    def _all_ignore_loader():
        ids = torch.zeros(2, 6, dtype=torch.long)
        # labels all -100 → all ignored
        labels = torch.full((2, 6), -100, dtype=torch.long)
        yield {"input_ids": ids, "labels": labels}

    ppl = perplexity(model, _all_ignore_loader(), ignore_index=-100)
    assert ppl == float("inf")


# ---------------------------------------------------------------------------
# exact_match — empty predictions (line 116)
# ---------------------------------------------------------------------------

def test_invariant_exact_match_empty_predictions_returns_zero():
    """exact_match([]) returns 0.0 immediately (line 116)."""
    assert exact_match([], []) == 0.0


def test_invariant_exact_match_empty_predictions_with_refs_returns_zero():
    """exact_match with an empty preds list and non-empty refs still 0.0."""
    assert exact_match([], ["hello", "world"]) == 0.0


# ---------------------------------------------------------------------------
# exact_match — normalize=False branch
# ---------------------------------------------------------------------------

def test_invariant_exact_match_no_normalize_case_sensitive():
    """With normalize=False, case differences produce 0.0."""
    assert exact_match(["Hello"], ["hello"], normalize=False) == 0.0


def test_invariant_exact_match_no_normalize_exact_case():
    """With normalize=False, identical strings match."""
    assert exact_match(["Hello"], ["Hello"], normalize=False) == 1.0


# ---------------------------------------------------------------------------
# _lcs_length — empty sequence (line 141)
# ---------------------------------------------------------------------------

def test_invariant_lcs_length_empty_sequence():
    """_lcs_length returns 0 when either input list is empty (line 141)."""
    from lighttrain.eval.metrics import _lcs_length

    assert _lcs_length([], ["a", "b"]) == 0
    assert _lcs_length(["a", "b"], []) == 0
    assert _lcs_length([], []) == 0


# ---------------------------------------------------------------------------
# rouge_score — empty predictions → zero dict (line 170)
# ---------------------------------------------------------------------------

def test_invariant_rouge_empty_predictions_returns_zero_dict():
    """rouge_score([]) returns {'precision': 0.0, 'recall': 0.0, 'f1': 0.0} (line 170)."""
    out = rouge_score([], [])
    assert out == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# ---------------------------------------------------------------------------
# rouge_score — rouge1 variant (lines 180-185)
# ---------------------------------------------------------------------------

def test_invariant_rouge1_perfect_match():
    """rouge1 on identical strings yields f1 ≈ 1.0 (lines 180-185)."""
    preds = ["the cat sat on the mat"]
    refs = ["the cat sat on the mat"]
    out = rouge_score(preds, refs, variant="rouge1")
    assert out["f1"] == pytest.approx(1.0, abs=1e-6)
    assert out["precision"] == pytest.approx(1.0, abs=1e-6)
    assert out["recall"] == pytest.approx(1.0, abs=1e-6)


def test_invariant_rouge1_no_overlap():
    """rouge1 with no shared unigrams → f1 ≈ 0.0."""
    out = rouge_score(["aaa bbb"], ["xxx yyy"], variant="rouge1")
    assert out["f1"] == pytest.approx(0.0, abs=1e-6)


def test_invariant_rouge1_partial_overlap():
    """rouge1 partial overlap: f1 between 0 and 1."""
    out = rouge_score(["the cat sat"], ["the dog ran"], variant="rouge1")
    assert 0.0 < out["f1"] < 1.0


# ---------------------------------------------------------------------------
# rouge_score — rouge2 variant (line 180, n=2)
# ---------------------------------------------------------------------------

def test_invariant_rouge2_perfect_match():
    """rouge2 on identical strings yields f1 ≈ 1.0."""
    preds = ["the cat sat on the mat"]
    refs = ["the cat sat on the mat"]
    out = rouge_score(preds, refs, variant="rouge2")
    assert out["f1"] == pytest.approx(1.0, abs=1e-6)


def test_invariant_rouge2_no_overlap():
    """rouge2 with no shared bigrams → f1 ≈ 0.0."""
    out = rouge_score(["aaa bbb ccc"], ["xxx yyy zzz"], variant="rouge2")
    assert out["f1"] == pytest.approx(0.0, abs=1e-6)


def test_invariant_rouge2_single_token_no_bigrams():
    """Single-token strings produce 0 bigrams; overlap is 0, counts default to max(1,…),
    so f1 is 0 (line 184: overlap/max(1,0) with overlap=0)."""
    out = rouge_score(["cat"], ["cat"], variant="rouge2")
    # overlap bigrams = 0, precision = 0/max(1,0)=0, recall=0 → f1=0
    assert out["f1"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# rouge_score — unknown variant raises ValueError (line 191)
# ---------------------------------------------------------------------------

def test_invariant_rouge_unknown_variant_raises():
    """rouge_score with an unrecognized variant raises ValueError (line 191)."""
    with pytest.raises(ValueError, match="unknown variant"):
        rouge_score(["hello world"], ["hello world"], variant="rouge3")


@pytest.mark.parametrize("bad_variant", ["rouge-l", "bleu", "", "rouge3", "rougeN"])
def test_invariant_rouge_bad_variant_always_raises(bad_variant):
    """Any variant string not matching rouge1/rouge2/rougeL raises ValueError.
    Note: the source lowercases the variant first, so 'ROUGE1' maps to 'rouge1'
    (valid). We only test strings that remain unrecognized after lowercasing.
    """
    with pytest.raises(ValueError, match="unknown variant"):
        rouge_score(["hello"], ["hello"], variant=bad_variant)


# ---------------------------------------------------------------------------
# bleu_score — empty predictions (line 226)
# ---------------------------------------------------------------------------

def test_invariant_bleu_empty_predictions_returns_zero():
    """bleu_score([]) returns 0.0 immediately (line 226)."""
    assert bleu_score([], []) == 0.0


def test_invariant_bleu_empty_predictions_with_refs_returns_zero():
    """bleu_score with empty preds and non-empty refs still returns 0.0."""
    assert bleu_score([], ["hello world"]) == 0.0


# ---------------------------------------------------------------------------
# bleu_score — no smooth variant
# ---------------------------------------------------------------------------

def test_invariant_bleu_no_smooth():
    """bleu_score with smooth=False and exact match still returns near 1.0."""
    score = bleu_score(
        ["the cat sat on the mat"],
        ["the cat sat on the mat"],
        smooth=False,
    )
    assert score >= 0.99


def test_invariant_bleu_no_smooth_disjoint_in_range():
    """bleu_score with smooth=False on disjoint strings is in [0, 1]."""
    score = bleu_score(["aaa bbb"], ["xxx yyy"], smooth=False)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# lm_eval_harness_hook — ImportError path (lines 275-278)
# ---------------------------------------------------------------------------

def test_invariant_lm_eval_harness_hook_import_error():
    """lm_eval_harness_hook raises ImportError with helpful message when
    lm_eval is not installed (lines 275-278)."""
    # Temporarily make `lm_eval` unimportable
    original = sys.modules.get("lm_eval", None)
    sys.modules["lm_eval"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ImportError, match="lm-eval"):
            lm_eval_harness_hook("hellaswag", model=None, tokenizer=None)
    finally:
        if original is None:
            del sys.modules["lm_eval"]
        else:
            sys.modules["lm_eval"] = original


# ---------------------------------------------------------------------------
# lm_eval_harness_hook — successful call path (lines 283-289)
# ---------------------------------------------------------------------------

def test_invariant_lm_eval_harness_hook_returns_task_results():
    """lm_eval_harness_hook calls lm_eval.simple_evaluate and returns the
    per-task result dict (lines 283-289)."""
    fake_lm_eval = types.ModuleType("lm_eval")
    task_result = {"acc": 0.75, "acc_stderr": 0.01}
    fake_lm_eval.simple_evaluate = MagicMock(
        return_value={"results": {"hellaswag": task_result}}
    )

    original = sys.modules.get("lm_eval", None)
    sys.modules["lm_eval"] = fake_lm_eval
    try:
        result = lm_eval_harness_hook(
            "hellaswag", model=MagicMock(), tokenizer=MagicMock(),
            num_fewshot=5, limit=100,
        )
    finally:
        if original is None:
            sys.modules.pop("lm_eval", None)
        else:
            sys.modules["lm_eval"] = original

    assert result == task_result
    fake_lm_eval.simple_evaluate.assert_called_once_with(
        model=fake_lm_eval.simple_evaluate.call_args.kwargs["model"],
        tasks=["hellaswag"],
        num_fewshot=5,
        limit=100,
    )


def test_invariant_lm_eval_harness_hook_missing_task_key_returns_empty():
    """If the task name is absent from results, hook returns {} (line 289
    .get(task_name, {}))."""
    fake_lm_eval = types.ModuleType("lm_eval")
    fake_lm_eval.simple_evaluate = MagicMock(
        return_value={"results": {}}  # task key absent
    )

    original = sys.modules.get("lm_eval", None)
    sys.modules["lm_eval"] = fake_lm_eval
    try:
        result = lm_eval_harness_hook(
            "winogrande", model=MagicMock(), tokenizer=MagicMock()
        )
    finally:
        if original is None:
            sys.modules.pop("lm_eval", None)
        else:
            sys.modules["lm_eval"] = original

    assert result == {}


def test_invariant_lm_eval_harness_hook_missing_results_key_returns_empty():
    """If simple_evaluate returns a dict without 'results', hook returns {}
    (lines 288-289: .get('results', {}))."""
    fake_lm_eval = types.ModuleType("lm_eval")
    fake_lm_eval.simple_evaluate = MagicMock(
        return_value={}  # no 'results' key at all
    )

    original = sys.modules.get("lm_eval", None)
    sys.modules["lm_eval"] = fake_lm_eval
    try:
        result = lm_eval_harness_hook(
            "hellaswag", model=MagicMock(), tokenizer=MagicMock()
        )
    finally:
        if original is None:
            sys.modules.pop("lm_eval", None)
        else:
            sys.modules["lm_eval"] = original

    assert result == {}


# ---------------------------------------------------------------------------
# perplexity — device argument (line 61-65) and max_batches early stop
# ---------------------------------------------------------------------------

def test_invariant_perplexity_device_cpu_moves_tensors():
    """perplexity with device=torch.device('cpu') still works correctly."""
    torch.manual_seed(5)
    model = _DictOutputLM()
    loader = [_make_batch() for _ in range(2)]
    ppl = perplexity(model, loader, device=torch.device("cpu"))
    assert math.isfinite(ppl) and ppl > 1.0


def test_invariant_perplexity_max_batches_zero_returns_inf():
    """max_batches=0 means the loop never executes → total_tokens=0 → inf."""
    torch.manual_seed(9)
    model = _DictOutputLM()
    loader = [_make_batch() for _ in range(5)]
    ppl = perplexity(model, loader, max_batches=0)
    assert ppl == float("inf")


# ---------------------------------------------------------------------------
# rouge_score — precision/recall/f1 arithmetic edge cases
# ---------------------------------------------------------------------------

def test_invariant_rouge_l_empty_prediction_string():
    """An empty prediction string results in 0 lcs → f1 toward 0.0."""
    out = rouge_score([""], ["hello world"], variant="rougeL")
    assert out["f1"] == pytest.approx(0.0, abs=1e-6)


def test_invariant_rouge1_corpus_level_averaging():
    """rouge1 computes corpus-level averages across multiple sentence pairs.

    Pair 1: 'the cat sat' vs 'the cat sat' → perfect.
    Pair 2: 'the dog ran' vs 'the cat sat' — 'the' overlaps (1 unigram),
    pred=3 tokens, ref=3 tokens → p=r=1/3 → f1=1/3.
    Corpus precision = (1.0 + 1/3)/2 = 2/3;
    corpus recall = (1.0 + 1/3)/2 = 2/3;
    f1 = 2/3 ≈ 0.667. We just assert it is strictly between 0 and 1.
    """
    preds = ["the cat sat", "the dog ran"]
    refs = ["the cat sat", "the cat sat"]
    out = rouge_score(preds, refs, variant="rouge1")
    assert 0.0 < out["f1"] < 1.0


# ---------------------------------------------------------------------------
# bleu_score — custom max_n
# ---------------------------------------------------------------------------

def test_invariant_bleu_max_n_1():
    """bleu_score with max_n=1 computes unigram BLEU only."""
    score = bleu_score(
        ["the cat sat on the mat"],
        ["the cat sat on the mat"],
        max_n=1,
        smooth=False,
    )
    assert score >= 0.99


def test_invariant_bleu_max_n_1_disjoint():
    """bleu_score max_n=1, smooth=False, disjoint → 0.0 or very low."""
    score = bleu_score(
        ["aaa bbb ccc"],
        ["xxx yyy zzz"],
        max_n=1,
        smooth=False,
    )
    # No unigram overlap; with no smooth counts[0]=0 → log(1e-10) driven
    assert 0.0 <= score <= 1.0
