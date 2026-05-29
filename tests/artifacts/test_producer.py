"""Adversarial tests for ``ModelForwardProducer``.

Targets that the legacy ``tests/test_artifact_producer.py`` misses:
  * Produced tensor values **match** the model's forward output (not just
    "some tensor of the right shape")
  * Outputs have ``requires_grad=False`` — no graph leak
  * ``produce`` skips already-stored sample_ids and returns ``{}``
  * ``derive_sample_id`` fallback is deterministic across two calls
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from lighttrain.artifacts import ModelForwardProducer, SafetensorsShardStore
from lighttrain.artifacts.producer import _coerce_model_output
from lighttrain.protocols import ModelOutput


# --------------------------------------------------------------------------- #
# Deterministic tiny model                                                    #
# --------------------------------------------------------------------------- #


class _TinyLM(nn.Module):
    """A deterministic tiny LM that returns logits = embedding(input_ids) @ W^T.

    We seed `W` so produced tensor values are predictable; tests assert
    ``produce(sample)["logits"]`` matches a hand-computed reference via
    ``torch.testing.assert_close``.
    """

    def __init__(self, vocab: int = 8, dim: int = 4) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, **_kw) -> ModelOutput:
        h = self.emb(input_ids)
        return ModelOutput(outputs={"logits": self.head(h)})


def _make_producer(tmp_path: Path) -> tuple[ModelForwardProducer, _TinyLM]:
    model = _TinyLM()
    store = SafetensorsShardStore(tmp_path / "art")
    return (
        ModelForwardProducer(
            model=model, store=store, collect_outputs=["logits"]
        ),
        model,
    )


# --------------------------------------------------------------------------- #
# Tensor value parity with model forward                                      #
# --------------------------------------------------------------------------- #


def test_producer_tensor_values_match_forward_output(tmp_path: Path) -> None:
    """Produced ``logits`` are bit-equivalent (within rtol/atol) to the model's
    forward output, after the producer's batch-of-1 squeeze.

    Input: a single sample with deterministic ``input_ids``.

    Analytical: ``ModelForwardProducer.produce`` constructs a batch-of-1
    via ``_as_batch``, runs the model under ``no_grad``, and squeezes the
    leading batch dim out of every output tensor (producer.py:226-227).
    The expected value is therefore ``model(input_ids.unsqueeze(0)).outputs["logits"][0]``.
    """
    torch.manual_seed(42)
    producer, model = _make_producer(tmp_path)
    producer.prepare()

    ids = torch.tensor([1, 3, 5, 7])
    sample = {"id": "s1", "input_ids": ids}
    out = producer.produce(sample)

    # Compute the analytical reference: batch-of-1 forward, then squeeze.
    with torch.no_grad():
        ref = model(input_ids=ids.unsqueeze(0)).outputs["logits"].squeeze(0)

    assert "logits" in out
    torch.testing.assert_close(out["logits"], ref, atol=1e-5, rtol=1e-4)


def test_invariant_producer_strips_requires_grad(tmp_path: Path) -> None:
    """Output tensors have ``requires_grad=False``.

    Invariant: prevents a graph leak — captured tensors must not carry
    autograd state that pins the model graph alive in callers.
    """
    producer, _ = _make_producer(tmp_path)
    producer.prepare()
    out = producer.produce({"id": "s1", "input_ids": torch.tensor([0, 1, 2])})
    for k, v in out.items():
        assert isinstance(v, torch.Tensor)
        assert v.requires_grad is False, f"{k} leaked requires_grad=True"


# --------------------------------------------------------------------------- #
# Sample-id skipping (resume-safety)                                          #
# --------------------------------------------------------------------------- #


def test_producer_skips_existing_sample_id(tmp_path: Path) -> None:
    """If the store already contains ``sid``, ``produce`` returns ``{}`` and
    does not call the model.

    Input: prefill the store with sample "s1" via direct ``put``; ensure
    ``ModelForwardProducer.produce({"id": "s1", ...})`` returns ``{}``.

    Invariant (resume-safety): producer is a forward over a partly-completed
    artifact dir; samples already written must not be re-forwarded.
    """
    model = _TinyLM()
    store = SafetensorsShardStore(tmp_path / "art")
    # Prefill + flush so contains("s1") returns True.
    store.put("s1", {"logits": torch.zeros(3, 8)})
    store._flush_shard()

    producer = ModelForwardProducer(
        model=model, store=store, collect_outputs=["logits"]
    )
    producer.prepare()

    # Counting forward invocations.
    n_calls = {"n": 0}
    real_forward = model.forward

    def _counting_forward(*a, **kw):
        n_calls["n"] += 1
        return real_forward(*a, **kw)

    model.forward = _counting_forward  # type: ignore[assignment]

    out = producer.produce({"id": "s1", "input_ids": torch.tensor([0, 1, 2])})
    assert out == {}
    assert n_calls["n"] == 0, "forward was called for an already-stored sample"


# --------------------------------------------------------------------------- #
# Sample-id fallback determinism                                              #
# --------------------------------------------------------------------------- #


def test_producer_derive_sample_id_deterministic(tmp_path: Path) -> None:
    """Without an explicit ``id``, ``derive_sample_id`` produces a deterministic
    key — calling produce twice with the same content lands on the same sid.

    Input: a sample dict with no ``id``; submit twice. Analytical:
    ``produce`` falls back to ``derive_sample_id(sample)``; the same input
    content yields the same id; the second call hits the skip branch.

    Note: ``derive_sample_id`` JSON-serializes the head-64 of ``input_ids``
    after a plain ``list(...)``, so it requires a JSON-friendly element type
    (Python ints, not 0-d tensors). The model in ``_as_batch`` accepts a
    list and turns it into a long tensor.

    Pin: after both puts, ``iter_keys`` reports exactly one stored sample.
    """
    producer, _ = _make_producer(tmp_path)
    producer.prepare()
    sample = {"input_ids": [2, 4, 6]}
    producer.produce(sample)
    producer.produce(sample)
    producer.finalize()
    keys = list(producer.store.iter_keys())
    assert len(keys) == 1


# --------------------------------------------------------------------------- #
# finalize writes manifest + count + idempotence                              #
# --------------------------------------------------------------------------- #


def test_producer_finalize_writes_manifest_complete(tmp_path: Path) -> None:
    """After ``finalize``, ``MANIFEST_COMPLETE.json`` exists with the right count.

    Input: produce 3 deterministic samples; finalize; inspect manifest.
    """
    producer, _ = _make_producer(tmp_path)
    producer.prepare()
    for sid, ids in (
        ("s1", torch.tensor([0, 1])),
        ("s2", torch.tensor([2, 3])),
        ("s3", torch.tensor([4, 5])),
    ):
        producer.produce({"id": sid, "input_ids": ids})
    producer.finalize()

    manifest_path = tmp_path / "art" / "MANIFEST_COMPLETE.json"
    assert manifest_path.exists()
    body = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert body["count"] == 3
    assert body["backend"] == "safetensors-shards"


def test_producer_coerce_model_output_passthrough() -> None:
    """``_coerce_model_output`` returns the input unchanged when it is already a
    ``ModelOutput``.

    Pin (lightweight): the coercer's identity path. Defends against a
    refactor where someone always wraps even pre-coerced outputs and
    silently doubles the squeeze.
    """
    mo = ModelOutput(outputs={"logits": torch.tensor([1.0])})
    assert _coerce_model_output(mo) is mo
