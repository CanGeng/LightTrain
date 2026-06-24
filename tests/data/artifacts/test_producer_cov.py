"""Edge-case coverage for ``builtin_plugins/data/artifacts/producer.py``.

Companion to ``test_producer.py``; drives the branches that file leaves
uncovered. What this module pins:

  * ``_as_batch`` carries ``modality_inputs`` through (producer.py:57) and
    moves nested batches onto a device.
  * ``_to_device`` recurses into list/tuple (preserving the concrete type)
    and passes non-tensor scalars through unchanged (68-71).
  * ``_resolve_store`` accepts a Mapping config (root/path/name/backend),
    raises on a config missing both ``root`` and ``path`` (85), opens an
    already-finalized store from a path string (90-91), and falls back to a
    fresh ``SafetensorsShardStore`` with a warning when the path is not yet
    finalized (92-99).
  * ``collect_attentions`` injects ``output_attentions=True`` (165).
  * ``prepare`` defaults the device to CPU for a param-less model (188-189)
    and reads ``artifact_version`` out of cfg (198).
  * ``produce`` coerces a non-``ModelOutput`` return (217), stores a
    non-Mapping hook payload (223), and skips an output filtered out by
    ``collect_outputs`` (230).
  * ``finalize`` writes a lineage ``produced_by`` edge via the most-recent
    run-node fallback with a warning (281-289), and warns (not raises) when
    the lineage write blows up (301-305).
  * ``_coerce_model_output`` handles Mapping / ``.logits`` duck-type / bare
    Tensor / unknown-object inputs (312-321).
  * ``run_artifact_production`` drives prepare→produce→finalize (331-334).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.data.artifacts import (
    ModelForwardProducer,
    SafetensorsShardStore,
    open_artifact_store,
)
from lighttrain.builtin_plugins.data.artifacts.producer import (
    _as_batch,
    _coerce_model_output,
    _resolve_store,
    _to_device,
    run_artifact_production,
)
from lighttrain.protocols import ModelOutput

# --------------------------------------------------------------------------- #
# Deterministic stubs                                                         #
# --------------------------------------------------------------------------- #


class _TinyLM(nn.Module):
    """Deterministic embed→head LM returning a ``ModelOutput``."""

    def __init__(self, vocab: int = 8, dim: int = 4) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, **_kw) -> ModelOutput:
        h = self.emb(input_ids)
        return ModelOutput(outputs={"logits": self.head(h)})


class _DictReturningModel(nn.Module):
    """Returns a plain ``dict`` (not a ``ModelOutput``) so ``produce`` must
    route through ``_coerce_model_output``."""

    def __init__(self, vocab: int = 8, dim: int = 4) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, **_kw) -> dict[str, torch.Tensor]:
        h = self.emb(input_ids)
        return {"logits": self.head(h)}


class _ParamlessModel(nn.Module):
    """A module with no parameters; ``next(parameters())`` raises StopIteration."""

    def forward(self, input_ids, **_kw) -> ModelOutput:
        # int input_ids -> emit a float tensor of matching length.
        return ModelOutput(outputs={"logits": input_ids.float().unsqueeze(-1)})


class _LogitsOnlyOutput:
    """Duck-typed HF-style output: has ``.logits`` but is not a Mapping."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeLineageStore:
    """Minimal lineage stand-in. Records calls; the run-node fallback reads
    ``iter_nodes('run')`` and the edge lands in ``edges``."""

    def __init__(self, run_rows: list[dict[str, Any]] | None = None) -> None:
        self._run_rows = run_rows or []
        self.edges: list[tuple[int, int, str, dict[str, Any]]] = []
        self.upserts: list[dict[str, Any]] = []

    def upsert_node(self, **kwargs: Any) -> int:
        self.upserts.append(kwargs)
        return 777  # deterministic artifact node id

    def iter_nodes(self, *, kind: str | None = None):
        if kind in (None, "run"):
            yield from self._run_rows

    def add_edge(self, src: int, dst: int, kind: str, payload: dict[str, Any]) -> None:
        self.edges.append((src, dst, kind, payload))


class _ExplodingLineageStore:
    """``upsert_node`` raises, exercising finalize's warn-not-raise guard."""

    def upsert_node(self, **_kwargs: Any) -> int:
        raise RuntimeError("boom: lineage backend offline")


def _make_producer(tmp_path: Path, **kw: Any) -> ModelForwardProducer:
    return ModelForwardProducer(
        model=_TinyLM(),
        store=SafetensorsShardStore(tmp_path / "art"),
        **kw,
    )


# --------------------------------------------------------------------------- #
# _as_batch / _to_device                                                      #
# --------------------------------------------------------------------------- #


def test_invariant_as_batch_carries_modality_inputs() -> None:
    """``_as_batch`` copies ``modality_inputs`` straight through (line 57)."""
    mod = {"pixel_values": torch.zeros(2, 2)}
    batch = _as_batch({"input_ids": torch.tensor([1, 2]), "modality_inputs": mod})
    assert batch["modality_inputs"] is mod


def test_invariant_as_batch_moves_to_device_when_given() -> None:
    """A non-None device routes the batch through ``_to_device`` (line 58-59).

    CPU is the only device guaranteed present, so we assert the tensors stay
    on CPU and remain equal rather than asserting a device change.
    """
    dev = torch.device("cpu")
    batch = _as_batch({"input_ids": [1, 2, 3]}, device=dev)
    assert batch["input_ids"].device == dev
    torch.testing.assert_close(batch["input_ids"], torch.tensor([[1, 2, 3]]))


def test_invariant_to_device_recurses_into_list_and_tuple() -> None:
    """``_to_device`` recurses through list/tuple and preserves the concrete
    container type (lines 68-70)."""
    dev = torch.device("cpu")
    as_list = _to_device([torch.tensor([1]), torch.tensor([2])], dev)
    as_tuple = _to_device((torch.tensor([1]),), dev)
    assert isinstance(as_list, list)
    assert isinstance(as_tuple, tuple)
    assert as_list[0].device == dev and as_tuple[0].device == dev


def test_invariant_to_device_passes_scalars_through() -> None:
    """Non-tensor, non-container values fall through unchanged (line 71)."""
    assert _to_device("hello", torch.device("cpu")) == "hello"
    assert _to_device(42, torch.device("cpu")) == 42


# --------------------------------------------------------------------------- #
# _resolve_store                                                              #
# --------------------------------------------------------------------------- #


def test_invariant_resolve_store_from_mapping_config(tmp_path: Path) -> None:
    """A ``{name, root, shard_size}`` mapping resolves to the registered backend
    with extra kwargs forwarded (lines 80-88)."""
    store = _resolve_store(
        {"name": "safetensors-shards", "root": str(tmp_path / "art"), "shard_size": 7}
    )
    assert isinstance(store, SafetensorsShardStore)
    assert store.shard_size == 7
    assert store.root == tmp_path / "art"


def test_invariant_resolve_store_mapping_missing_root_raises(tmp_path: Path) -> None:
    """A mapping config without ``root`` or ``path`` is a config error (line 85)."""
    with pytest.raises(ValueError, match=r"needs `root` or `path`"):
        _resolve_store({"name": "safetensors-shards"})


def test_invariant_resolve_store_opens_finalized_path(tmp_path: Path) -> None:
    """A path string pointing at a *finalized* store opens it for read
    (lines 89-91), not a fresh shard store."""
    root = tmp_path / "art"
    seed = SafetensorsShardStore(root)
    seed.put("s1", {"logits": torch.zeros(2, 4)})
    seed.finalize()

    resolved = _resolve_store(str(root))
    assert isinstance(resolved, SafetensorsShardStore)
    assert resolved._finalized is True
    assert "s1" in list(resolved.iter_keys())


def test_pin_current_behavior_resolve_store_unfinalized_path_falls_back(
    tmp_path: Path,
) -> None:
    """A path with no ``MANIFEST_COMPLETE.json`` cannot be opened, so
    ``_resolve_store`` warns and creates a fresh writable shard store
    (lines 92-99).

    Pins current behavior: the fallback swallows the open error (logged via
    ``_log.warning``) and silently returns a brand-new store — callers in
    *produce* mode rely on this resume path.
    """
    root = tmp_path / "fresh"
    store = _resolve_store(str(root))
    assert isinstance(store, SafetensorsShardStore)
    assert store._finalized is False
    # Writable: a fresh store accepts a put.
    store.put("x", {"t": torch.zeros(1)})
    assert store.contains("x")


def test_invariant_resolve_store_passes_through_instance(tmp_path: Path) -> None:
    """An already-built store instance is returned as-is (line 100)."""
    built = SafetensorsShardStore(tmp_path / "art")
    assert _resolve_store(built) is built


# --------------------------------------------------------------------------- #
# __init__ / prepare                                                          #
# --------------------------------------------------------------------------- #


def test_invariant_collect_attentions_sets_forward_kwarg(tmp_path: Path) -> None:
    """``collect_attentions=True`` injects ``output_attentions=True`` into the
    forward kwargs (line 165)."""
    producer = _make_producer(tmp_path, collect_attentions=True)
    assert producer.forward_kwargs["output_attentions"] is True


def test_invariant_prepare_defaults_device_for_paramless_model(tmp_path: Path) -> None:
    """A model with no parameters drives the StopIteration branch and pins the
    device to CPU (lines 187-189)."""
    producer = ModelForwardProducer(
        model=_ParamlessModel(), store=SafetensorsShardStore(tmp_path / "art")
    )
    producer.prepare()
    assert producer._device == torch.device("cpu")


def test_invariant_prepare_reads_artifact_version_from_cfg(tmp_path: Path) -> None:
    """``prepare(cfg)`` picks up ``artifact_version`` from the cfg dict (line 198)."""
    producer = _make_producer(tmp_path)
    producer.prepare({"artifact_version": "v9", "artifact_name": "named"})
    assert producer.artifact_version == "v9"
    assert producer.artifact_name == "named"


# --------------------------------------------------------------------------- #
# produce: coercion / hook payload / collect_outputs filter                   #
# --------------------------------------------------------------------------- #


def test_invariant_produce_coerces_dict_model_output(tmp_path: Path) -> None:
    """A model returning a plain ``dict`` is coerced via ``_coerce_model_output``
    before flattening (line 216-217), and the tensor still lands in the store."""
    producer = ModelForwardProducer(
        model=_DictReturningModel(),
        store=SafetensorsShardStore(tmp_path / "art"),
        collect_outputs=["logits"],
    )
    producer.prepare()
    out = producer.produce({"id": "s1", "input_ids": torch.tensor([0, 1, 2])})
    assert "logits" in out
    assert out["logits"].requires_grad is False


def test_invariant_produce_stores_non_mapping_hook_payload(tmp_path: Path) -> None:
    """When a hook yields a bare Tensor (not a Mapping) ``produce`` assigns it
    straight onto ``extras`` (line 222-223).

    A ``layer``-transform extra returns a single tensor (one indexed layer),
    so the non-Mapping branch fires; a topk extra would hit the Mapping branch.
    """
    from lighttrain.models.extras import ExtraOutputSpec

    torch.manual_seed(0)
    model = _TinyLM()
    producer = ModelForwardProducer(
        model=model,
        store=SafetensorsShardStore(tmp_path / "art"),
        # Capture the head's output tensor verbatim (no transform => Tensor).
        extras=[ExtraOutputSpec(name="head_out", source="head")],
        collect_outputs=["logits"],
    )
    producer.prepare()
    out = producer.produce({"id": "s1", "input_ids": torch.tensor([1, 2, 3])})
    assert "head_out" in out
    assert isinstance(out["head_out"], torch.Tensor)


def test_invariant_collect_outputs_filters_unlisted_output(tmp_path: Path) -> None:
    """An output key absent from ``collect_outputs`` is dropped (line 227-230)."""

    class _TwoHeadLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.emb = nn.Embedding(8, 4)
            self.head = nn.Linear(4, 8, bias=False)

        def forward(self, input_ids, **_kw) -> ModelOutput:
            h = self.emb(input_ids)
            logits = self.head(h)
            return ModelOutput(outputs={"logits": logits, "aux": logits * 2})

    producer = ModelForwardProducer(
        model=_TwoHeadLM(),
        store=SafetensorsShardStore(tmp_path / "art"),
        collect_outputs=["logits"],  # 'aux' must be filtered out
    )
    producer.prepare()
    out = producer.produce({"id": "s1", "input_ids": torch.tensor([0, 1])})
    assert "logits" in out
    assert "aux" not in out


# --------------------------------------------------------------------------- #
# finalize: lineage fallback + warn-not-raise                                 #
# --------------------------------------------------------------------------- #


def test_pin_current_behavior_finalize_falls_back_to_recent_run_node(
    tmp_path: Path,
) -> None:
    """With no explicit ``run_node_id``, finalize sorts run nodes by
    ``(ts, id)`` descending and links the most recent, emitting a warning
    (lines 281-293).

    Pins current behavior: the fallback is order-stable but a *warned*
    best-guess — the docstring in source flags it as removable guesswork.
    """
    run_rows = [
        {"id": 1, "run_id": "old", "ts": 1.0},
        {"id": 2, "run_id": "new", "ts": 5.0},  # most recent -> chosen
        {"id": 3, "run_id": "mid", "ts": 3.0},
    ]
    fake = _FakeLineageStore(run_rows=run_rows)
    producer = _make_producer(tmp_path, artifact_name="art", artifact_version="v1")
    producer.prepare({"lineage_store": fake})
    producer.produce({"id": "s1", "input_ids": torch.tensor([0, 1])})

    with pytest.warns(UserWarning, match="no explicit run_node_id"):
        producer.finalize()

    assert len(fake.edges) == 1
    src, dst, kind, payload = fake.edges[0]
    assert src == 2  # the (ts=5.0) winner
    assert dst == 777
    assert kind == "produced_by"
    assert "elapsed_s" in payload


def test_invariant_finalize_no_run_candidates_writes_no_edge(tmp_path: Path) -> None:
    """When no run node carries a ``run_id``, the fallback finds no candidate
    and no edge is written (the ``if candidates`` guard at 284 is False)."""
    # run rows present but all lack a run_id -> filtered out at line 281-283.
    fake = _FakeLineageStore(run_rows=[{"id": 1, "run_id": None, "ts": 1.0}])
    producer = _make_producer(tmp_path, artifact_name="art")
    producer.prepare({"lineage_store": fake})
    producer.produce({"id": "s1", "input_ids": torch.tensor([0, 1])})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning expected on this path
        producer.finalize()
    assert fake.edges == []
    assert len(fake.upserts) == 1  # artifact node still upserted


def test_pin_current_behavior_finalize_warns_on_lineage_failure(
    tmp_path: Path,
) -> None:
    """A lineage write that raises is swallowed into a ``warnings.warn`` rather
    than propagating (lines 301-305).

    Pins current behavior: finalize must still return the manifest path even
    when lineage bookkeeping fails.
    """
    producer = _make_producer(tmp_path, artifact_name="art")
    producer.prepare({"lineage_store": _ExplodingLineageStore()})
    producer.produce({"id": "s1", "input_ids": torch.tensor([0, 1])})

    with pytest.warns(UserWarning, match="lineage write failed: boom"):
        manifest = producer.finalize()
    assert manifest.exists()


# --------------------------------------------------------------------------- #
# _coerce_model_output                                                        #
# --------------------------------------------------------------------------- #


def test_invariant_coerce_mapping_keeps_only_tensors() -> None:
    """A Mapping input keeps tensor values and drops non-tensors (line 312-313)."""
    out = _coerce_model_output({"logits": torch.zeros(2), "meta": "skip", "n": 3})
    assert set(out.outputs) == {"logits"}
    assert isinstance(out.outputs["logits"], torch.Tensor)


def test_invariant_coerce_logits_duck_type() -> None:
    """An object exposing ``.logits`` (HF-style) is mapped, pulling optional
    loss/hidden_states/attentions when present (lines 314-318)."""
    obj = _LogitsOnlyOutput(torch.ones(3))
    out = _coerce_model_output(obj)
    assert torch.equal(out.outputs["logits"], torch.ones(3))
    assert out.loss is None
    assert out.hidden_states is None  # empty tuple -> None
    assert out.attentions is None


def test_invariant_coerce_bare_tensor() -> None:
    """A bare Tensor becomes ``outputs={'output': t}`` (lines 319-320)."""
    t = torch.arange(4)
    out = _coerce_model_output(t)
    assert torch.equal(out.outputs["output"], t)


def test_invariant_coerce_unknown_object_empty_outputs() -> None:
    """An object that matches none of the branches yields empty outputs
    (line 321)."""
    out = _coerce_model_output(object())
    assert out.outputs == {}


# --------------------------------------------------------------------------- #
# run_artifact_production driver                                              #
# --------------------------------------------------------------------------- #


def test_invariant_run_artifact_production_drives_end_to_end(tmp_path: Path) -> None:
    """``run_artifact_production`` runs prepare→produce(*)→finalize and returns a
    manifest path covering every sample (lines 331-334)."""
    producer = _make_producer(tmp_path, collect_outputs=["logits"])
    dataset = [
        {"id": "a", "input_ids": torch.tensor([0, 1])},
        {"id": "b", "input_ids": torch.tensor([2, 3])},
    ]
    manifest = run_artifact_production(dataset, producer)
    assert manifest.exists()

    store = open_artifact_store(tmp_path / "art")
    assert sorted(store.iter_keys()) == ["a", "b"]
