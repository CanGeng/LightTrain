"""Edge-case unit tests for ``lighttrain.builtin_plugins.data.core._module``.

What we pin:

* ``SimpleDataModule.__init__``:
  - default ByteTokenizer + CausalLMCollator (lines 57-60, uncovered branches)
  - explicit tokenizer/collator/sampler passthrough
  - mapping-resolved collator and sampler
  - scalar kwargs coerced to declared types
  - val_dataset wired vs absent

* ``train_loader``: sampler-present (shuffle=False) and sampler-absent (shuffle=True)
* ``val_loader`` (line 81): absent → None; present → DataLoader
* ``predict_loader`` (line 91): always None
* ``seek`` (line 103): no sampler → early return; no .seek → early return;
  sampler.seek called with batch_size multiplication
* ``state_dict`` / ``load_state_dict``: stateful round-trip, no-sampler empty, missing key
* ``_maybe_resolve`` (line 130): None, passthrough non-mapping, mapping+default_kwargs
* ``_resolve_dataset`` (lines 139-145): non-mapping passthrough, mapping with/without
  name → resolve or ValueError
* ``_maybe_resolve_sampler`` (line 157): None, non-mapping passthrough, mapping resolve
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lighttrain.builtin_plugins.data.core._module import (
    SimpleDataModule,
    _maybe_resolve,
    _maybe_resolve_sampler,
    _resolve_dataset,
)

# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------


class _Tokenizer:
    """Minimal tokenizer stand-in carrying a custom pad_id."""

    pad_id = 42


class _Collator:
    """Count-returning collator so we can assert it was wired."""

    def __init__(self, pad_id: int = 0) -> None:
        self.pad_id = pad_id

    def __call__(self, samples: list[Any]) -> dict[str, int]:
        return {"n": len(samples)}


class _SamplerWithSeek:
    """Sampler that records seek() calls and supports state_dict."""

    def __init__(self, dataset: Any) -> None:
        self._n = len(dataset)
        self.seek_calls: list[tuple[int, int]] = []
        self._state: dict[str, Any] = {}

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        return iter(range(self._n))

    def seek(self, epoch: int, consumed_indices: int) -> None:
        self.seek_calls.append((epoch, consumed_indices))

    def state_dict(self) -> dict[str, Any]:
        return dict(self._state)

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._state = dict(sd)


class _SamplerNoSeek:
    """Sampler without seek() or state_dict()."""

    def __init__(self, dataset: Any) -> None:
        self._n = len(dataset)

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        return iter(range(self._n))


# ---------------------------------------------------------------------------
# Fixture: a tiny in-memory list dataset that satisfies len()
# ---------------------------------------------------------------------------


_TINY_DATASET = [{"input_ids": [i]} for i in range(4)]


def _make_dm(**overrides: Any) -> SimpleDataModule:
    """Build a SimpleDataModule with a tiny list dataset and sensible defaults."""
    kwargs: dict[str, Any] = dict(
        dataset=_TINY_DATASET,
        collator=_Collator(),
    )
    kwargs.update(overrides)
    return SimpleDataModule(**kwargs)


# ===========================================================================
# __init__ — default tokenizer and collator (covers lines 57-60)
# ===========================================================================


def test_invariant_default_byte_tokenizer_created_when_none_given():
    """No tokenizer → ``ByteTokenizer`` instance is built (lines 42-45)."""
    from lighttrain.builtin_plugins.data.core.tokenizers import ByteTokenizer

    dm = _make_dm()
    assert isinstance(dm.tokenizer, ByteTokenizer)


def test_invariant_default_causal_lm_collator_created_when_none_given():
    """No collator → ``CausalLMCollator`` is built with the tokenizer's pad_id
    (lines 57-60).
    """
    from lighttrain.builtin_plugins.data.collators.text import CausalLMCollator
    from lighttrain.builtin_plugins.data.core.tokenizers import PAD_ID

    dm = SimpleDataModule(dataset=_TINY_DATASET)
    assert isinstance(dm.collator, CausalLMCollator)
    assert dm.collator.pad_id == PAD_ID


def test_invariant_default_collator_inherits_custom_tokenizer_pad_id():
    """When a tokenizer with a custom pad_id is given but no collator, the
    CausalLMCollator is built using that custom pad_id (line 55, 58-60).
    """
    from lighttrain.builtin_plugins.data.collators.text import CausalLMCollator

    dm = SimpleDataModule(dataset=_TINY_DATASET, tokenizer=_Tokenizer())
    assert isinstance(dm.collator, CausalLMCollator)
    assert dm.collator.pad_id == 42


def test_invariant_explicit_tokenizer_passed_through():
    """An explicit tokenizer instance passes straight through (lines 41-45)."""
    tok = _Tokenizer()
    dm = _make_dm(tokenizer=tok)
    assert dm.tokenizer is tok


def test_invariant_explicit_collator_passed_through():
    """An explicit collator instance is stored without wrapping (lines 54-60)."""
    coll = _Collator(pad_id=99)
    dm = _make_dm(collator=coll)
    assert dm.collator is coll


def test_invariant_scalar_kwargs_coerced():
    """Numeric and bool kwargs are coerced to the declared types (lines 36-39)."""
    dm = _make_dm(
        batch_size="3",
        num_workers="0",
        pin_memory=1,
        drop_last=0,
    )
    assert dm.batch_size == 3 and isinstance(dm.batch_size, int)
    assert dm.num_workers == 0 and isinstance(dm.num_workers, int)
    assert dm.pin_memory is True
    assert dm.drop_last is False


def test_invariant_val_dataset_wired_when_provided():
    """A val_dataset spec that is already a list is stored directly (lines 48-52)."""
    val_ds = [{"input_ids": [10, 11]}]
    dm = _make_dm(val_dataset=val_ds)
    assert dm.val_dataset is val_ds


def test_invariant_val_dataset_none_when_not_provided():
    """Without a val_dataset argument, val_dataset is None (lines 48-52)."""
    dm = _make_dm()
    assert dm.val_dataset is None


def test_invariant_mapping_collator_resolved():
    """A ``{name: causal_lm}`` collator mapping is resolved via the registry."""
    from lighttrain.builtin_plugins.data.collators.text import CausalLMCollator

    dm = SimpleDataModule(
        dataset=_TINY_DATASET,
        tokenizer=_Tokenizer(),  # pad_id = 42
        collator={"name": "causal_lm"},
    )
    assert isinstance(dm.collator, CausalLMCollator)
    assert dm.collator.pad_id == 42


def test_invariant_mapping_sampler_resolved():
    """A ``{name: sequential}`` sampler mapping is resolved with the dataset injected."""
    from lighttrain.builtin_plugins.data.core.samplers import SequentialSampler

    dm = _make_dm(sampler={"name": "sequential"})
    assert isinstance(dm._train_sampler, SequentialSampler)


def test_invariant_explicit_sampler_instance_passed_through():
    """A pre-built sampler object is stored as-is (line 157 of _maybe_resolve_sampler)."""
    sampler = _SamplerNoSeek(_TINY_DATASET)
    dm = _make_dm(sampler=sampler)
    assert dm._train_sampler is sampler


# ===========================================================================
# train_loader
# ===========================================================================


def test_invariant_train_loader_shuffle_true_when_no_sampler():
    """Without a sampler DataLoader uses shuffle=True (no explicit sampler set)."""
    from torch.utils.data import DataLoader

    dm = _make_dm()
    loader = dm.train_loader()
    assert isinstance(loader, DataLoader)
    # When shuffle=True, torch inserts a RandomSampler; the explicit sampler is None.
    assert loader.sampler is not None


def test_invariant_train_loader_uses_sampler_when_provided():
    """A pre-built sampler is forwarded to the DataLoader with shuffle=False."""
    sampler = _SamplerNoSeek(_TINY_DATASET)
    dm = _make_dm(sampler=sampler)
    loader = dm.train_loader()
    assert loader.sampler is sampler


def test_invariant_train_loader_respects_batch_size():
    """``batch_size`` flows to the DataLoader."""
    dm = _make_dm(batch_size=2)
    loader = dm.train_loader()
    assert loader.batch_size == 2


# ===========================================================================
# val_loader (line 81)
# ===========================================================================


def test_invariant_val_loader_returns_none_when_no_val_dataset():
    """``val_loader()`` returns None when there is no validation dataset (line 80)."""
    dm = _make_dm()
    assert dm.val_loader() is None


def test_invariant_val_loader_returns_dataloader_with_val_dataset():
    """``val_loader()`` returns a DataLoader over val_dataset (line 81)."""
    from torch.utils.data import DataLoader

    val_ds = [{"input_ids": [1, 2]}, {"input_ids": [3, 4]}]
    dm = _make_dm(val_dataset=val_ds, batch_size=2)
    loader = dm.val_loader()
    assert isinstance(loader, DataLoader)
    # Validation loader must not shuffle.
    assert loader.batch_size == 2


# ===========================================================================
# predict_loader (line 91)
# ===========================================================================


def test_invariant_predict_loader_always_returns_none():
    """``predict_loader()`` always returns None (line 91)."""
    dm = _make_dm()
    assert dm.predict_loader() is None


# ===========================================================================
# seek (line 103)
# ===========================================================================


def test_invariant_seek_noop_when_no_sampler():
    """``seek`` returns early when ``_train_sampler`` is None (line 102-103)."""
    dm = _make_dm()
    assert dm._train_sampler is None
    # Must not raise; nothing happens.
    dm.seek(epoch=0, consumed_batches=5)


def test_invariant_seek_noop_when_sampler_has_no_seek():
    """``seek`` returns early when the sampler has no ``.seek`` method (line 102-103)."""
    sampler = _SamplerNoSeek(_TINY_DATASET)
    dm = _make_dm(sampler=sampler)
    assert not hasattr(dm._train_sampler, "seek")
    dm.seek(epoch=1, consumed_batches=2)  # must not raise


def test_invariant_seek_delegates_to_sampler_with_batch_multiplication():
    """``seek`` calls ``sampler.seek(epoch, consumed_batches * batch_size)`` (line 104)."""
    sampler = _SamplerWithSeek(_TINY_DATASET)
    dm = _make_dm(sampler=sampler, batch_size=4)
    dm._train_sampler = sampler  # wire it in directly

    dm.seek(epoch=2, consumed_batches=3)

    assert sampler.seek_calls == [(2, 12)]  # 3 * batch_size(4) = 12


def test_invariant_seek_coerces_args_to_int():
    """``seek`` casts its arguments to int before delegating (line 104)."""
    sampler = _SamplerWithSeek(_TINY_DATASET)
    dm = _make_dm(sampler=sampler, batch_size=2)
    dm._train_sampler = sampler

    dm.seek(epoch="1", consumed_batches="3")  # type: ignore[arg-type]
    assert sampler.seek_calls == [(1, 6)]


# ===========================================================================
# state_dict / load_state_dict
# ===========================================================================


def test_invariant_state_dict_empty_without_sampler():
    """No sampler → ``state_dict()`` is an empty dict."""
    dm = _make_dm()
    assert dm.state_dict() == {}


def test_invariant_state_dict_empty_when_sampler_has_no_state_dict():
    """A sampler lacking ``state_dict`` → empty state dict."""
    sampler = _SamplerNoSeek(_TINY_DATASET)
    dm = _make_dm(sampler=sampler)
    assert dm.state_dict() == {}


def test_invariant_state_dict_roundtrip_with_stateful_sampler():
    """A stateful sampler's state is captured and restored without error."""
    sampler = _SamplerWithSeek(_TINY_DATASET)
    sampler._state = {"epoch": 3, "skip": 7}
    dm = _make_dm(sampler=sampler)
    dm._train_sampler = sampler

    sd = dm.state_dict()
    assert "sampler" in sd
    assert sd["sampler"] == {"epoch": 3, "skip": 7}

    # Round-trip
    dm.load_state_dict({"sampler": {"epoch": 5, "skip": 2}})
    assert sampler._state == {"epoch": 5, "skip": 2}


def test_invariant_load_state_dict_noop_when_sampler_key_missing():
    """``load_state_dict`` with no ``sampler`` key leaves a stateful sampler untouched."""
    sampler = _SamplerWithSeek(_TINY_DATASET)
    sampler._state = {"epoch": 1, "skip": 0}
    dm = _make_dm(sampler=sampler)
    dm._train_sampler = sampler

    before = sampler.state_dict()
    dm.load_state_dict({})
    assert sampler.state_dict() == before


def test_invariant_load_state_dict_noop_when_no_sampler():
    """``load_state_dict`` with no train sampler is a silent no-op."""
    dm = _make_dm()
    dm.load_state_dict({"sampler": {"epoch": 0}})  # must not raise


# ===========================================================================
# _maybe_resolve (line 130)
# ===========================================================================


def test_invariant_maybe_resolve_none_returns_none():
    """``_maybe_resolve(None, ...)`` returns None."""
    assert _maybe_resolve(None, "collator") is None


def test_invariant_maybe_resolve_passthrough_non_mapping():
    """A non-mapping instance passes through unchanged (line 130)."""
    sentinel = _Collator(pad_id=7)
    result = _maybe_resolve(sentinel, "collator")
    assert result is sentinel


def test_invariant_maybe_resolve_mapping_resolves_via_registry():
    """A mapping is resolved through the registry (line 135)."""
    from lighttrain.builtin_plugins.data.collators.text import CausalLMCollator

    result = _maybe_resolve({"name": "causal_lm", "pad_id": 5}, "collator")
    assert isinstance(result, CausalLMCollator)
    assert result.pad_id == 5


def test_invariant_maybe_resolve_default_kwargs_applied_to_mapping():
    """``default_kwargs`` are set-defaulted into the spec before resolution."""
    from lighttrain.builtin_plugins.data.collators.text import CausalLMCollator

    result = _maybe_resolve(
        {"name": "causal_lm"},
        "collator",
        default_kwargs={"pad_id": 17},
    )
    assert isinstance(result, CausalLMCollator)
    assert result.pad_id == 17


def test_invariant_maybe_resolve_explicit_kwarg_wins_over_default():
    """An explicit value in the spec is preserved over ``default_kwargs``."""
    from lighttrain.builtin_plugins.data.collators.text import CausalLMCollator

    result = _maybe_resolve(
        {"name": "causal_lm", "pad_id": 3},
        "collator",
        default_kwargs={"pad_id": 99},
    )
    assert isinstance(result, CausalLMCollator)
    assert result.pad_id == 3


# ===========================================================================
# _resolve_dataset (lines 139-145)
# ===========================================================================


def test_invariant_resolve_dataset_passthrough_non_mapping():
    """A non-mapping dataset spec is returned as-is (line 140)."""
    ds = [{"input_ids": [1, 2]}]
    result = _resolve_dataset(ds, tokenizer=_Tokenizer())
    assert result is ds


def test_invariant_resolve_dataset_mapping_with_name_resolves(tmp_path: Path):
    """A mapping with ``name`` key resolves via the registry (line 144)."""
    # Write a tiny text file so LineFileTextDataset can read it.
    txt = tmp_path / "tiny.txt"
    txt.write_text("hello world\n", encoding="utf-8")

    _Tokenizer()
    # We need a real tokenizer here because LineFileTextDataset calls encode()
    from lighttrain.builtin_plugins.data.core.tokenizers import ByteTokenizer

    tok_real = ByteTokenizer()
    ds = _resolve_dataset(
        {"name": "line_file_text", "path": str(txt)},
        tokenizer=tok_real,
    )
    assert len(ds) == 1


def test_invariant_resolve_dataset_mapping_with_target_resolves(tmp_path: Path):
    """A mapping with ``_target_`` key resolves via import (line 144)."""
    txt = tmp_path / "data.txt"
    txt.write_text("foo bar\nbaz\n", encoding="utf-8")

    from lighttrain.builtin_plugins.data.core.tokenizers import ByteTokenizer

    tok = ByteTokenizer()
    ds = _resolve_dataset(
        {
            "_target_": "lighttrain.builtin_plugins.data.core.datasets.LineFileTextDataset",
            "path": str(txt),
        },
        tokenizer=tok,
    )
    assert len(ds) == 2


def test_invariant_resolve_dataset_mapping_without_name_or_target_raises():
    """A mapping without ``name`` or ``_target_`` raises ``ValueError`` (line 145)."""
    with pytest.raises(ValueError, match="needs `name` or `_target_`"):
        _resolve_dataset({"path": "/some/file.txt"}, tokenizer=_Tokenizer())


def test_invariant_resolve_dataset_injects_tokenizer_as_default(tmp_path: Path):
    """``tokenizer`` is setdefault-injected into the mapping before resolution.

    Use LineFileTextDataset (registered under ``line_file_text``) so the
    tokenizer is consumed via ``tokenizer.encode``; verify dataset builds OK
    which proves the injected tokenizer reached the constructor.
    """
    from lighttrain.builtin_plugins.data.core.tokenizers import ByteTokenizer

    txt = tmp_path / "inj.txt"
    txt.write_text("hello\n", encoding="utf-8")
    tok = ByteTokenizer()

    # Spec does NOT include 'tokenizer'; it must be injected by _resolve_dataset.
    ds = _resolve_dataset(
        {"name": "line_file_text", "path": str(txt)},
        tokenizer=tok,
    )
    # Dataset built successfully and has one sample.
    assert len(ds) == 1


# ===========================================================================
# _maybe_resolve_sampler (line 157)
# ===========================================================================


def test_invariant_maybe_resolve_sampler_none_returns_none():
    """``_maybe_resolve_sampler(None, ...)`` returns None."""
    assert _maybe_resolve_sampler(None, dataset=_TINY_DATASET) is None


def test_invariant_maybe_resolve_sampler_passthrough_non_mapping():
    """A non-mapping sampler instance is returned unchanged (line 157)."""
    sampler = _SamplerNoSeek(_TINY_DATASET)
    result = _maybe_resolve_sampler(sampler, dataset=_TINY_DATASET)
    assert result is sampler


def test_invariant_maybe_resolve_sampler_mapping_resolves_and_injects_dataset():
    """A mapping is resolved with dataset injected via setdefault (lines 154-156)."""
    from lighttrain.builtin_plugins.data.core.samplers import SequentialSampler

    result = _maybe_resolve_sampler({"name": "sequential"}, dataset=_TINY_DATASET)
    assert isinstance(result, SequentialSampler)
    assert list(result) == list(range(len(_TINY_DATASET)))


def test_invariant_maybe_resolve_sampler_dataset_default_not_overridden():
    """An explicit ``dataset`` key in the spec is not replaced by setdefault."""
    from lighttrain.builtin_plugins.data.core.samplers import SequentialSampler

    explicit_ds = [0, 1, 2]
    result = _maybe_resolve_sampler(
        {"name": "sequential", "dataset": explicit_ds},
        dataset=_TINY_DATASET,  # different from explicit_ds
    )
    assert isinstance(result, SequentialSampler)
    # The injected dataset must be the explicit one (len 3), not _TINY_DATASET (len 4).
    assert len(result) == 3


# ===========================================================================
# End-to-end: SimpleDataModule with a real file dataset
# ===========================================================================


@pytest.mark.parametrize("batch_size", [1, 2])
def test_invariant_train_loader_yields_batches_end_to_end(
    tmp_path: Path, batch_size: int
):
    """Full pipeline: file dataset → SimpleDataModule → train_loader → batch."""

    txt = tmp_path / "corpus.txt"
    txt.write_text("\n".join(f"line{i}" for i in range(4)), encoding="utf-8")

    dm = SimpleDataModule(
        dataset={"name": "line_file_text", "path": str(txt)},
        batch_size=batch_size,
    )
    loader = dm.train_loader()
    batch = next(iter(loader))
    # CausalLMCollator returns input_ids, attention_mask, labels tensors.
    assert "input_ids" in batch
    assert batch["input_ids"].shape[0] == batch_size
