"""Edge-case unit tests for ``lighttrain.builtin_plugins.data.core._prep_module``.

Pins the public surface of :class:`PrepGraphDataModule` plus its module-level
helpers, driving every reachable branch toward 100% coverage.

What we pin:

* ``_normalize_terminal``: ``"prep_graph:<t>"`` prefix, ``{prep_graph: <t>}``
  mapping, ``None``, and a plain string (returns ``None``).
* ``__init__`` end-to-end against a *real* ``PrepGraph`` + ``PrepRunner`` built
  from a tiny fake :class:`PrepNode` (registered via ``clean_registry``):
  default ByteTokenizer + CausalLMCollator, explicit tokenizer/collator
  objects, mapping-resolved collator/sampler, ``run_on_init`` semantics.
* ``_validate_terminals``: bad train terminal and bad val terminal both raise
  ``ValueError``.
* ``_dataset_for``: on-disk memmap (header) vs rows (shards) views, the
  ``store`` / ``rows`` in-memory fallbacks, and both ``RuntimeError`` guards
  (no result; neither rows nor store).
* DataLoader surface: ``train_loader`` with/without a sampler, ``val_loader``
  None vs DataLoader, ``predict_loader`` always None.
* ``state_dict`` / ``load_state_dict`` round-trip through a stateful sampler,
  plus the no-sampler and missing-key no-ops.
* ``_maybe_resolve`` / ``_maybe_resolve_sampler``: None, passthrough instance,
  and mapping resolution (with ``default_kwargs`` / injected ``dataset``).
"""

from __future__ import annotations

from typing import Any

import pytest

from lighttrain.builtin_plugins.data.core._prep_module import (
    PrepGraphDataModule,
    _maybe_resolve,
    _maybe_resolve_sampler,
    _normalize_terminal,
)
from lighttrain.data.cache._memmap import MemmapDataset, write_memmap
from lighttrain.data.cache._rows import _RowsDataset
from lighttrain.data.cache._shards import ShardWriter
from lighttrain.data.prepgraph.node import NodeResult, PrepNode
from lighttrain.registry import register

# ---------------------------------------------------------------------------
# Fake prep nodes — registered under non-conflicting kinds so a real PrepGraph
# + PrepRunner can build, run, and commit them to disk. Each writes a tiny
# committed cache the DataModule mounts via ``final_dir``.
# ---------------------------------------------------------------------------


class _RowsNode(PrepNode):
    """Materializes a handful of variable-length rows as JSONL shards."""

    kind = "_fakerows"
    schema_kind = "rows"

    def run(self, ctx) -> NodeResult:
        writer = ShardWriter(ctx.store_root)
        n = int(self.config.get("n", 6))
        for i in range(n):
            writer.write({"input_ids": [i, i + 1]})
        writer.finalize()
        return NodeResult(fingerprint="x")


class _MemmapNode(PrepNode):
    """Materializes a fixed-shape memmap (header.json present)."""

    kind = "_fakemem"
    schema_kind = "packed"

    def run(self, ctx) -> NodeResult:
        n = int(self.config.get("n", 3))
        write_memmap(
            ctx.store_root,
            [{"input_ids": [1, 2, 3, 4]} for _ in range(n)],
            seq_len=4,
        )
        return NodeResult(fingerprint="x")


class _Tokenizer:
    """Minimal tokenizer stand-in carrying a custom ``pad_id``."""

    pad_id = 42


class _Collator:
    """Identity-ish collator: returns the batch size so we can assert on it."""

    def __init__(self, pad_id: int = 0) -> None:
        self.pad_id = pad_id

    def __call__(self, samples: list[Any]) -> dict[str, int]:
        return {"n": len(samples)}


class _StatelessSampler:
    """A sampler object with no ``state_dict`` / ``load_state_dict``."""

    def __init__(self, dataset: Any) -> None:
        self._n = len(dataset)

    def __iter__(self):
        return iter(range(self._n))

    def __len__(self) -> int:
        return self._n


@pytest.fixture
def rows_nodes(clean_registry):
    """Register the fake nodes for the duration of a test (auto-restored)."""
    register("prep_node", "_fakerows", _RowsNode, force=True)
    register("prep_node", "_fakemem", _MemmapNode, force=True)
    return clean_registry


def _rows_spec(*names: str, n: int = 6) -> dict[str, Any]:
    nodes = [{"name": nm, "kind": "_fakerows", "n": n, "salt": nm} for nm in names]
    return {"nodes": nodes, "terminals": list(names)}


# ---------------------------------------------------------------------------
# _normalize_terminal
# ---------------------------------------------------------------------------


def test_invariant_normalize_terminal_strips_prefix():
    """``"prep_graph:<t>"`` returns the bare terminal name (lines 32-33)."""
    assert _normalize_terminal("prep_graph:train") == "train"


def test_invariant_normalize_terminal_reads_mapping():
    """A ``{prep_graph: <t>}`` mapping returns ``str(<t>)`` (lines 34-35)."""
    assert _normalize_terminal({"prep_graph": "val"}) == "val"
    assert _normalize_terminal({"prep_graph": 7}) == "7"


def test_invariant_normalize_terminal_none_returns_none():
    """``None`` short-circuits to ``None`` (lines 30-31)."""
    assert _normalize_terminal(None) is None


def test_invariant_normalize_terminal_plain_string_returns_none():
    """A bare string without the ``prep_graph:`` prefix returns ``None`` (line 36)."""
    assert _normalize_terminal("train") is None
    assert _normalize_terminal({"other": "x"}) is None


# ---------------------------------------------------------------------------
# __init__ — defaults
# ---------------------------------------------------------------------------


def test_invariant_init_defaults_byte_tokenizer_and_causal_collator(
    rows_nodes, tmp_path
):
    """No tokenizer/collator → ByteTokenizer + CausalLMCollator (lines 86-90, 117-118)."""
    from lighttrain.builtin_plugins.data.collators.text import CausalLMCollator
    from lighttrain.builtin_plugins.data.core.tokenizers import PAD_ID, ByteTokenizer

    dm = PrepGraphDataModule(
        prep_graph=_rows_spec("train"),
        train="train",
        store_root=tmp_path / "store",
    )
    assert isinstance(dm.tokenizer, ByteTokenizer)
    assert isinstance(dm.collator, CausalLMCollator)
    # The collator inherits the tokenizer's pad_id (ByteTokenizer.pad_id == PAD_ID).
    assert dm.collator.pad_id == PAD_ID
    assert len(dm.dataset) == 6
    assert dm.val_dataset is None
    assert dm._train_sampler is None


def test_invariant_init_runs_runner_and_mounts_rows_dataset(rows_nodes, tmp_path):
    """``run_on_init=True`` executes the graph; terminal mounts as a rows view."""
    dm = PrepGraphDataModule(
        prep_graph=_rows_spec("train"),
        train="train",
        store_root=tmp_path / "store",
    )
    assert isinstance(dm.dataset, _RowsDataset)
    assert dm.dataset[0] == {"input_ids": [0, 1]}


def test_pin_current_behavior_run_on_init_false_raises_missing_result(
    rows_nodes, tmp_path
):
    """Pin: ``run_on_init=False`` leaves ``_results`` empty, so ``__init__``'s own
    ``_dataset_for(train)`` raises ``RuntimeError`` (lines 100-103 + 138-140).

    Debatable: the flag exists to *defer* the run, yet construction eagerly
    mounts datasets and therefore cannot complete without results. We pin the
    current contract rather than asserting an (absent) lazy path.
    """
    with pytest.raises(RuntimeError, match="has no result"):
        PrepGraphDataModule(
            prep_graph=_rows_spec("train"),
            train="train",
            store_root=tmp_path / "store",
            run_on_init=False,
        )


# ---------------------------------------------------------------------------
# __init__ — explicit / mapping-resolved components
# ---------------------------------------------------------------------------


def test_invariant_init_accepts_explicit_tokenizer_and_collator(rows_nodes, tmp_path):
    """Pre-built tokenizer/collator instances pass straight through (lines 85, 112-118)."""
    tok = _Tokenizer()
    coll = _Collator(pad_id=99)
    dm = PrepGraphDataModule(
        prep_graph=_rows_spec("train"),
        train="train",
        store_root=tmp_path / "store",
        tokenizer=tok,
        collator=coll,
    )
    assert dm.tokenizer is tok
    assert dm.collator is coll


def test_invariant_init_resolves_collator_mapping_with_pad_id_default(
    rows_nodes, tmp_path
):
    """A ``{name: causal_lm}`` collator mapping resolves with the tokenizer pad_id
    injected as a default (lines 112-116, _maybe_resolve 219-223)."""
    from lighttrain.builtin_plugins.data.collators.text import CausalLMCollator

    dm = PrepGraphDataModule(
        prep_graph=_rows_spec("train"),
        train="train",
        store_root=tmp_path / "store",
        tokenizer=_Tokenizer(),  # pad_id == 42
        collator={"name": "causal_lm"},
    )
    assert isinstance(dm.collator, CausalLMCollator)
    assert dm.collator.pad_id == 42


def test_invariant_init_resolves_sampler_mapping_and_injects_dataset(
    rows_nodes, tmp_path
):
    """A ``{name: shuffle}`` sampler mapping resolves with ``dataset`` injected
    (line 121, _maybe_resolve_sampler 231-234)."""
    from lighttrain.builtin_plugins.data.core.samplers import ShuffleSampler

    dm = PrepGraphDataModule(
        prep_graph=_rows_spec("train"),
        train="train",
        store_root=tmp_path / "store",
        sampler={"name": "shuffle", "seed": 0},
    )
    assert isinstance(dm._train_sampler, ShuffleSampler)


def test_invariant_init_mounts_val_terminal(rows_nodes, tmp_path):
    """A ``val`` terminal is validated and mounted as a second dataset (lines 82, 107-109)."""
    dm = PrepGraphDataModule(
        prep_graph=_rows_spec("train", "val"),
        train="train",
        val="val",
        store_root=tmp_path / "store",
    )
    assert dm.val_terminal == "val"
    assert len(dm.val_dataset) == 6


def test_invariant_init_scalar_kwargs_are_coerced(rows_nodes, tmp_path):
    """Numeric/bool kwargs are coerced to their declared types (lines 77-81)."""
    dm = PrepGraphDataModule(
        prep_graph=_rows_spec("train"),
        train="train",
        store_root=tmp_path / "store",
        batch_size="4",
        num_workers="0",
        pin_memory=1,
        drop_last=0,
    )
    assert dm.batch_size == 4 and isinstance(dm.batch_size, int)
    assert dm.num_workers == 0
    assert dm.pin_memory is True
    assert dm.drop_last is False


# ---------------------------------------------------------------------------
# _validate_terminals
# ---------------------------------------------------------------------------


def test_invariant_unknown_train_terminal_raises(rows_nodes, tmp_path):
    """An unknown train terminal raises ``ValueError`` (lines 126-130)."""
    with pytest.raises(ValueError, match="train terminal 'nope' not found"):
        PrepGraphDataModule(
            prep_graph=_rows_spec("train"),
            train="nope",
            store_root=tmp_path / "store",
        )


def test_invariant_unknown_val_terminal_raises(rows_nodes, tmp_path):
    """An unknown val terminal raises ``ValueError`` (lines 131-135)."""
    with pytest.raises(ValueError, match="val terminal 'nope' not found"):
        PrepGraphDataModule(
            prep_graph=_rows_spec("train"),
            train="train",
            val="nope",
            store_root=tmp_path / "store",
        )


# ---------------------------------------------------------------------------
# _dataset_for — on-disk views + fallbacks
# ---------------------------------------------------------------------------


def test_invariant_dataset_for_memmap_view(rows_nodes, tmp_path):
    """A memmap final_dir (header present) mounts as ``MemmapDataset`` (lines 150-151)."""
    spec = {"nodes": [{"name": "train", "kind": "_fakemem"}], "terminals": ["train"]}
    dm = PrepGraphDataModule(
        prep_graph=spec, train="train", store_root=tmp_path / "store"
    )
    assert isinstance(dm.dataset, MemmapDataset)
    assert len(dm.dataset) == 3
    assert dm.dataset[0]["input_ids"] == [1, 2, 3, 4]


def _bare_module() -> PrepGraphDataModule:
    """A PrepGraphDataModule with __init__ bypassed (only ``_results`` set)."""
    return object.__new__(PrepGraphDataModule)


def test_invariant_dataset_for_store_fallback_when_no_final_dir():
    """No ``final_dir`` but a ``store`` handle → returns the store (lines 153-154)."""
    obj = _bare_module()
    store = [{"input_ids": [9]}]
    obj._results = {"a": NodeResult(fingerprint="x", final_dir=None, store=store)}
    assert obj._dataset_for("a") is store


def test_invariant_dataset_for_rows_fallback_lists_rows():
    """No ``final_dir`` / ``store`` but ``rows`` present → ``list(rows)`` (lines 155-156)."""
    obj = _bare_module()
    rows = iter([{"input_ids": [7]}, {"input_ids": [8]}])
    obj._results = {"b": NodeResult(fingerprint="x", final_dir=None, rows=rows)}
    out = obj._dataset_for("b")
    assert out == [{"input_ids": [7]}, {"input_ids": [8]}]
    assert isinstance(out, list)


def test_invariant_dataset_for_missing_result_raises():
    """A terminal with no result entry raises ``RuntimeError`` (lines 138-140)."""
    obj = _bare_module()
    obj._results = {}
    with pytest.raises(RuntimeError, match="terminal 'gone' has no result"):
        obj._dataset_for("gone")


def test_invariant_dataset_for_neither_rows_nor_store_raises():
    """A result with no final_dir, store, or rows raises ``RuntimeError`` (lines 157-160)."""
    obj = _bare_module()
    obj._results = {"c": NodeResult(fingerprint="x", final_dir=None)}
    with pytest.raises(RuntimeError, match="neither rows nor a store"):
        obj._dataset_for("c")


def test_pin_current_behavior_stale_final_dir_falls_through_to_store():
    """Pin: a ``final_dir`` that does not exist on disk is skipped, falling back
    to the in-memory ``store`` (line 146 false branch → 153-154)."""
    obj = _bare_module()
    store = [{"input_ids": [5]}]
    obj._results = {
        "d": NodeResult(
            fingerprint="x", final_dir=_tmp_missing(), store=store, rows=[{"x": 1}]
        )
    }
    # store takes precedence over rows when final_dir is missing.
    assert obj._dataset_for("d") is store


def _tmp_missing():
    from pathlib import Path

    return Path("/nonexistent/prepgraph/final/dir/that/does/not/exist")


# ---------------------------------------------------------------------------
# DataLoader surface
# ---------------------------------------------------------------------------


def test_invariant_train_loader_without_sampler_shuffles(rows_nodes, tmp_path):
    """No sampler → DataLoader with ``shuffle=True`` and no sampler (lines 165-175)."""
    dm = PrepGraphDataModule(
        prep_graph=_rows_spec("train"),
        train="train",
        store_root=tmp_path / "store",
        batch_size=2,
        collator=_Collator(),
    )
    loader = dm.train_loader()
    assert loader.sampler is not None  # torch fills a RandomSampler when shuffle=True
    batch = next(iter(loader))
    assert batch == {"n": 2}


def test_invariant_train_loader_with_sampler_disables_shuffle(rows_nodes, tmp_path):
    """An explicit sampler is forwarded and ``shuffle`` is forced False (lines 165-175)."""
    dm = PrepGraphDataModule(
        prep_graph=_rows_spec("train"),
        train="train",
        store_root=tmp_path / "store",
        batch_size=3,
        sampler={"name": "shuffle", "seed": 0},
        collator=_Collator(),
    )
    loader = dm.train_loader()
    assert loader.sampler is dm._train_sampler
    batch = next(iter(loader))
    assert batch == {"n": 3}


def test_invariant_val_loader_none_when_no_val_dataset(rows_nodes, tmp_path):
    """``val_loader`` returns None when there is no validation dataset (lines 178-179)."""
    dm = PrepGraphDataModule(
        prep_graph=_rows_spec("train"),
        train="train",
        store_root=tmp_path / "store",
    )
    assert dm.val_loader() is None


def test_invariant_val_loader_builds_dataloader(rows_nodes, tmp_path):
    """With a val dataset, ``val_loader`` returns a non-shuffling DataLoader (lines 180-187)."""
    from torch.utils.data import DataLoader

    dm = PrepGraphDataModule(
        prep_graph=_rows_spec("train", "val"),
        train="train",
        val="val",
        store_root=tmp_path / "store",
        batch_size=2,
        collator=_Collator(),
    )
    loader = dm.val_loader()
    assert isinstance(loader, DataLoader)
    assert next(iter(loader)) == {"n": 2}


def test_invariant_predict_loader_is_none(rows_nodes, tmp_path):
    """``predict_loader`` is always None (line 190)."""
    dm = PrepGraphDataModule(
        prep_graph=_rows_spec("train"),
        train="train",
        store_root=tmp_path / "store",
    )
    assert dm.predict_loader() is None


# ---------------------------------------------------------------------------
# state_dict / load_state_dict
# ---------------------------------------------------------------------------


def test_invariant_state_dict_roundtrips_stateful_sampler(rows_nodes, tmp_path):
    """A stateful sampler's state is captured and restored (lines 193-206)."""
    dm = PrepGraphDataModule(
        prep_graph=_rows_spec("train"),
        train="train",
        store_root=tmp_path / "store",
        sampler={"name": "shuffle", "seed": 0},
    )
    sd = dm.state_dict()
    assert "sampler" in sd
    # Round-trips without error and re-applies to the sampler.
    dm.load_state_dict(sd)


def test_invariant_state_dict_empty_without_sampler(rows_nodes, tmp_path):
    """No sampler → empty state_dict, and load_state_dict is a no-op (lines 193-194, 201-206)."""
    dm = PrepGraphDataModule(
        prep_graph=_rows_spec("train"),
        train="train",
        store_root=tmp_path / "store",
    )
    assert dm.state_dict() == {}
    dm.load_state_dict({"sampler": {"anything": 1}})  # no sampler → ignored


def test_invariant_state_dict_skips_stateless_sampler():
    """A sampler lacking ``state_dict`` yields ``{}`` (line 194 guard short-circuits)."""
    obj = _bare_module()
    obj._train_sampler = _StatelessSampler([0, 1, 2])
    assert obj.state_dict() == {}
    # load_state_dict also no-ops because the sampler lacks load_state_dict.
    obj.load_state_dict({"sampler": {"x": 1}})


def test_invariant_load_state_dict_missing_key_is_noop():
    """``load_state_dict`` without a ``sampler`` key does nothing (line 201 guard)."""
    from lighttrain.builtin_plugins.data.core.samplers import ShuffleSampler

    obj = _bare_module()
    obj._train_sampler = ShuffleSampler([0, 1, 2], seed=0)
    before = obj._train_sampler.state_dict()
    obj.load_state_dict({})  # no "sampler" key → untouched
    assert obj._train_sampler.state_dict() == before


# ---------------------------------------------------------------------------
# _maybe_resolve / _maybe_resolve_sampler
# ---------------------------------------------------------------------------


def test_invariant_maybe_resolve_none_returns_none():
    """``_maybe_resolve(None, ...)`` returns None (lines 215-216)."""
    assert _maybe_resolve(None, "collator") is None


def test_invariant_maybe_resolve_passes_through_instance():
    """A non-mapping instance is returned unchanged (lines 217-218)."""
    sentinel = _Collator(pad_id=3)
    assert _maybe_resolve(sentinel, "collator") is sentinel


def test_invariant_maybe_resolve_mapping_applies_default_kwargs():
    """A mapping resolves via the registry, with ``default_kwargs`` set-defaulted (lines 219-223)."""
    from lighttrain.builtin_plugins.data.collators.text import CausalLMCollator

    coll = _maybe_resolve(
        {"name": "causal_lm"}, "collator", default_kwargs={"pad_id": 11}
    )
    assert isinstance(coll, CausalLMCollator)
    assert coll.pad_id == 11


def test_invariant_maybe_resolve_explicit_kwarg_overrides_default():
    """An explicit pad_id in the spec wins over the default_kwargs (setdefault, line 222)."""
    from lighttrain.builtin_plugins.data.collators.text import CausalLMCollator

    coll = _maybe_resolve(
        {"name": "causal_lm", "pad_id": 5}, "collator", default_kwargs={"pad_id": 11}
    )
    assert isinstance(coll, CausalLMCollator)
    assert coll.pad_id == 5


def test_invariant_maybe_resolve_sampler_none_returns_none():
    """``_maybe_resolve_sampler(None, ...)`` returns None (lines 229-230)."""
    assert _maybe_resolve_sampler(None, dataset=[1, 2]) is None


def test_invariant_maybe_resolve_sampler_passes_through_instance():
    """A non-mapping sampler instance is returned unchanged (line 235)."""
    sampler = _StatelessSampler([0, 1])
    assert _maybe_resolve_sampler(sampler, dataset=[0, 1]) is sampler


def test_invariant_maybe_resolve_sampler_mapping_injects_dataset():
    """A sampler mapping resolves with ``dataset`` injected (lines 231-234)."""
    from lighttrain.builtin_plugins.data.core.samplers import SequentialSampler

    dataset = [0, 1, 2, 3]
    sampler = _maybe_resolve_sampler({"name": "sequential"}, dataset=dataset)
    assert isinstance(sampler, SequentialSampler)
    assert list(sampler) == [0, 1, 2, 3]
