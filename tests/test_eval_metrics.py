"""EvalSuite metrics tests (M6) — perplexity / exact_match / rouge / bleu."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from lighttrain.eval.metrics import (
    bleu_score,
    exact_match,
    perplexity,
    rouge_score,
)
from lighttrain.protocols import ModelOutput


# ---- Tiny model for perplexity -------------------------------------------

class _TinyLM(nn.Module):
    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None):
        h = self.emb(input_ids)
        return ModelOutput(outputs={"logits": self.proj(h)})


def _make_loader(B: int = 2, T: int = 6, V: int = 16, n: int = 3):
    for _ in range(n):
        ids = torch.randint(0, V, (B, T))
        yield {"input_ids": ids, "labels": ids.clone()}


# ---- perplexity -----------------------------------------------------------

def test_perplexity_finite():
    model = _TinyLM()
    ppl = perplexity(model, _make_loader(), max_batches=2)
    assert math.isfinite(ppl) and ppl > 1.0


def test_perplexity_max_batches():
    model = _TinyLM()
    processed = []

    class _CountingLoader:
        def __iter__(self):
            for batch in _make_loader(n=10):
                yield batch
                processed.append(1)  # appended after yield (inside perplexity loop)

    perplexity(model, _CountingLoader(), max_batches=3)
    # processed is appended only after the batch is actually used
    assert sum(processed) == 3


# ---- exact_match ---------------------------------------------------------

def test_exact_match_perfect():
    preds = ["hello world", "foo bar"]
    refs  = ["hello world", "foo bar"]
    assert exact_match(preds, refs) == 1.0


def test_exact_match_none():
    assert exact_match(["abc"], ["xyz"]) == 0.0


def test_exact_match_normalize():
    assert exact_match(["Hello World"], ["hello world"], normalize=True) == 1.0


# ---- rouge ---------------------------------------------------------------

def test_rouge_l_perfect():
    preds = ["the cat sat on the mat"]
    refs  = ["the cat sat on the mat"]
    out = rouge_score(preds, refs, variant="rougeL")
    f1 = out.get("f1", out.get("rougeL", out.get("f", 0.0)))
    assert f1 >= 0.99


def test_rouge_zero_when_no_overlap():
    preds = ["aaaa bbbb"]
    refs  = ["xxxx yyyy"]
    out = rouge_score(preds, refs)
    f1 = out.get("f1", out.get("f", max(out.values())))
    assert f1 < 0.01


# ---- bleu ----------------------------------------------------------------

def test_bleu_perfect():
    score = bleu_score(["the cat sat on the mat"], ["the cat sat on the mat"])
    assert score >= 0.99


def test_bleu_between_zero_and_one():
    score = bleu_score(["a b c"], ["x y z"])
    assert 0.0 <= score <= 1.0
