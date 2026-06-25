"""Edge-case + error-path tests for
``lighttrain.observability.diagnostics.frozen_step``.

Complements the happy-path coverage in ``test_frozen_step_bundle.py``,
``test_frozen_step_peft.py`` and ``test_replay_step.py`` by driving the
best-effort swallow branches that those leave untouched:

* **snapshot** — model ``state_dict`` copy / optimizer ``deepcopy`` / RNG
  capture each fail independently → that slice is ``None`` and a WARNING with
  ``exc_info`` is logged (122-128, 133-139, 142-148);
* **restore_snapshot** — no snapshot is a no-op; model / optimizer
  ``load_state_dict`` and RNG restore failures are swallowed-with-warning
  (176-205);
* **commit** — no snapshot → warns + ``None``; an unknown ``reason`` is
  normalised to ``"scheduled"``; a write failure that also removes the temp
  zip exercises the ``tmp.unlink()`` FileNotFoundError swallow (216-222,
  308-309);
* **commit lineage** — the ``add_edge`` "produced_by" edge is written when a
  ``run_node_id`` is set, and a lineage exception is non-fatal (327-332);
* **replay_step_bundle** — adapter-import failure warns; a bundle with no
  ``model_spec`` raises; a vanished temp safetensors file is swallowed; an
  RNG-restore failure warns (399-417, 423-424);
* **_infer_model_spec** — a peft ``ImportError`` falls through to the
  short-name / ``_target_`` paths (492-493).
"""

from __future__ import annotations

import json
import logging
import sys
import zipfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.models.text.tiny_lm import TinyCausalLM
from lighttrain.engine._context import StepContext
from lighttrain.observability.diagnostics import frozen_step as fs_mod
from lighttrain.observability.diagnostics.frozen_step import (
    FrozenStepBundle,
    FrozenStepWriter,
    _infer_model_spec,
    read_frozen_step_bundle,
    replay_step_bundle,
)

_FS_LOGGER = "lighttrain.observability.diagnostics.frozen_step"


# ---------------------------------------------------------------------------
# Stubs / helpers
# ---------------------------------------------------------------------------


class _Ctx:
    """Minimal step-context double exposing only ``epoch``."""

    def __init__(self, epoch: int = 0) -> None:
        self.epoch = epoch


class _BadStateModel(nn.Module):
    """``state_dict()`` raises → snapshot model-copy branch."""

    def state_dict(self, *a, **k):  # noqa: D401, ANN001
        raise RuntimeError("state_dict unavailable")

    def forward(self, **batch):  # pragma: no cover — never called
        return {}


class _BadOptimizer:
    """``state_dict()`` raises → snapshot optimizer-copy branch."""

    def state_dict(self):  # noqa: D401
        raise RuntimeError("opt state_dict unavailable")


class _RaisingLoadModel(nn.Module):
    """``load_state_dict`` raises → restore model branch swallow."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 2)

    def load_state_dict(self, *a, **k):  # noqa: ANN001
        raise RuntimeError("cannot load model state")


class _RaisingLoadOptimizer:
    """Has ``load_state_dict`` but it raises → restore optimizer swallow."""

    def load_state_dict(self, *a, **k):  # noqa: ANN001
        raise RuntimeError("cannot load opt state")


class _RecordingLineageStore:
    """Lineage double: records upsert/edge calls; optionally raises."""

    def __init__(self, *, node_id: int = 99, raise_on: str | None = None) -> None:
        self._node_id = node_id
        self._raise_on = raise_on
        self.upserts: list[dict] = []
        self.edges: list[tuple] = []

    def upsert_node(self, **kw):  # noqa: ANN003
        if self._raise_on == "upsert":
            raise RuntimeError("lineage upsert exploded")
        self.upserts.append(kw)
        return self._node_id

    def add_edge(self, src, dst, kind, payload=None):  # noqa: ANN001
        if self._raise_on == "edge":
            raise RuntimeError("lineage edge exploded")
        self.edges.append((src, dst, kind, payload))


def _tiny() -> TinyCausalLM:
    return TinyCausalLM(
        vocab_size=32, d_model=8, n_layers=1, n_heads=2, max_seq_len=8
    )


def _ids_batch() -> dict:
    return {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }


def _snapshotted_writer(tmp_path: Path, **wkw) -> FrozenStepWriter:
    """Writer with a committed-ready snapshot of a tiny model."""
    torch.manual_seed(0)
    model = _tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    writer = FrozenStepWriter(tmp_path, **wkw)
    writer.snapshot(
        step=5, ctx=StepContext(step=5, epoch=0), batch=_ids_batch(),
        model=model, optimizer=opt, config_resolved_yaml="x: 1\n",
    )
    return writer


# ---------------------------------------------------------------------------
# snapshot — best-effort swallow branches
# ---------------------------------------------------------------------------


def test_invariant_snapshot_model_copy_failure_is_swallowed(tmp_path, caplog):
    """A model whose ``state_dict()`` raises leaves ``model_state=None`` + warns."""
    writer = FrozenStepWriter(tmp_path)
    with caplog.at_level(logging.WARNING, logger=_FS_LOGGER):
        writer.snapshot(
            step=1, ctx=_Ctx(), batch=_ids_batch(),
            model=_BadStateModel(), optimizer=None,
        )
    assert writer._snapshot is not None
    assert writer._snapshot["model_state"] is None
    recs = [r for r in caplog.records if "model state_dict copy failed" in r.getMessage()]
    assert recs, caplog.text
    assert recs[0].exc_info is not None


def test_invariant_snapshot_optimizer_copy_failure_is_swallowed(tmp_path, caplog):
    """An optimizer whose ``state_dict()`` raises leaves ``optimizer_state=None``."""
    writer = FrozenStepWriter(tmp_path)
    with caplog.at_level(logging.WARNING, logger=_FS_LOGGER):
        writer.snapshot(
            step=1, ctx=_Ctx(), batch=_ids_batch(),
            model=_tiny(), optimizer=_BadOptimizer(),
        )
    assert writer._snapshot["optimizer_state"] is None
    recs = [
        r for r in caplog.records
        if "optimizer state_dict copy failed" in r.getMessage()
    ]
    assert recs, caplog.text
    assert recs[0].exc_info is not None


def test_invariant_snapshot_rng_capture_failure_is_swallowed(
    tmp_path, monkeypatch, caplog
):
    """If ``rng_state()`` raises, the snapshot keeps ``rng_state=None`` + warns."""
    def _boom():
        raise RuntimeError("rng capture broke")

    monkeypatch.setattr(fs_mod, "rng_state", _boom)
    writer = FrozenStepWriter(tmp_path)
    with caplog.at_level(logging.WARNING, logger=_FS_LOGGER):
        writer.snapshot(
            step=1, ctx=_Ctx(), batch=_ids_batch(),
            model=_tiny(), optimizer=None,
        )
    assert writer._snapshot["rng_state"] is None
    recs = [r for r in caplog.records if "RNG state capture failed" in r.getMessage()]
    assert recs, caplog.text
    assert recs[0].exc_info is not None


def test_invariant_snapshot_optimizer_without_state_dict_is_none(tmp_path):
    """An optimizer with no ``state_dict`` attribute snapshots as ``None``."""
    writer = FrozenStepWriter(tmp_path)
    writer.snapshot(
        step=1, ctx=_Ctx(), batch=_ids_batch(),
        model=_tiny(), optimizer=object(),
    )
    assert writer._snapshot["optimizer_state"] is None


# ---------------------------------------------------------------------------
# restore_snapshot
# ---------------------------------------------------------------------------


def test_invariant_restore_without_snapshot_is_noop(tmp_path):
    """``restore_snapshot`` returns immediately when nothing was captured."""
    writer = FrozenStepWriter(tmp_path)
    assert writer._snapshot is None
    # Must not raise even though no model/optimizer state exists.
    writer.restore_snapshot(model=_tiny(), optimizer=None)


def test_invariant_restore_round_trips_model_params(tmp_path):
    """A snapshotted model's params are restored over a mutated model."""
    torch.manual_seed(1)
    model = _tiny()
    writer = FrozenStepWriter(tmp_path)
    writer.snapshot(
        step=1, ctx=_Ctx(), batch=_ids_batch(), model=model, optimizer=None
    )
    before = {k: v.clone() for k, v in model.state_dict().items()}
    # Mutate every parameter in place, then restore.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    writer.restore_snapshot(model=model, optimizer=None)
    after = model.state_dict()
    for k, v in before.items():
        assert torch.allclose(after[k], v)


def test_invariant_restore_model_load_failure_is_swallowed(tmp_path, caplog):
    """A model whose ``load_state_dict`` raises is swallowed-with-warning."""
    writer = FrozenStepWriter(tmp_path)
    writer.snapshot(
        step=1, ctx=_Ctx(), batch=_ids_batch(), model=_tiny(), optimizer=None
    )
    with caplog.at_level(logging.WARNING, logger=_FS_LOGGER):
        writer.restore_snapshot(model=_RaisingLoadModel(), optimizer=None)
    recs = [
        r for r in caplog.records
        if "model load_state_dict failed" in r.getMessage()
    ]
    assert recs, caplog.text
    assert recs[0].exc_info is not None


def test_invariant_restore_optimizer_load_failure_is_swallowed(tmp_path, caplog):
    """An optimizer whose ``load_state_dict`` raises is swallowed-with-warning."""
    model = _tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Take an optimizer step so optimizer_state is non-empty / non-None.
    out = model(input_ids=torch.zeros(1, 2, dtype=torch.long))
    out.outputs["logits"].mean().backward()
    opt.step()
    writer = FrozenStepWriter(tmp_path)
    writer.snapshot(
        step=1, ctx=_Ctx(), batch=_ids_batch(), model=model, optimizer=opt
    )
    assert writer._snapshot["optimizer_state"] is not None
    with caplog.at_level(logging.WARNING, logger=_FS_LOGGER):
        writer.restore_snapshot(model=None, optimizer=_RaisingLoadOptimizer())
    recs = [
        r for r in caplog.records
        if "optimizer load_state_dict failed" in r.getMessage()
    ]
    assert recs, caplog.text
    assert recs[0].exc_info is not None


def test_invariant_restore_rng_failure_is_swallowed(tmp_path, monkeypatch, caplog):
    """An ``restore_rng_state`` failure during restore is swallowed-with-warning."""
    writer = FrozenStepWriter(tmp_path)
    writer.snapshot(
        step=1, ctx=_Ctx(), batch=_ids_batch(), model=_tiny(), optimizer=None
    )
    assert writer._snapshot["rng_state"] is not None

    def _boom(_state):
        raise RuntimeError("rng restore broke")

    monkeypatch.setattr(fs_mod, "restore_rng_state", _boom)
    with caplog.at_level(logging.WARNING, logger=_FS_LOGGER):
        writer.restore_snapshot(model=None, optimizer=None)
    recs = [r for r in caplog.records if "RNG restore failed" in r.getMessage()]
    assert recs, caplog.text
    assert recs[0].exc_info is not None


# ---------------------------------------------------------------------------
# commit — guards + reason normalisation + temp cleanup
# ---------------------------------------------------------------------------


def test_invariant_commit_without_snapshot_warns_and_returns_none(tmp_path, caplog):
    """Committing before any ``snapshot()`` warns and writes nothing."""
    writer = FrozenStepWriter(tmp_path)
    with caplog.at_level(logging.WARNING, logger=_FS_LOGGER):
        result = writer.commit(reason="scheduled")
    assert result is None
    assert not list((tmp_path / "frozen_steps").glob("*.zip"))
    recs = [r for r in caplog.records if "no snapshot captured" in r.getMessage()]
    assert recs, caplog.text


def test_invariant_commit_unknown_reason_normalises_to_scheduled(tmp_path):
    """An out-of-vocab ``reason`` is rewritten to ``scheduled`` on disk + bundle."""
    writer = _snapshotted_writer(tmp_path)
    out = writer.commit(reason="totally-bogus")
    assert out is not None
    assert out.name == "step_5_scheduled.zip"
    bundle = read_frozen_step_bundle(out)
    assert bundle.reason == "scheduled"


def test_invariant_commit_tmp_unlink_missing_is_swallowed(
    tmp_path, monkeypatch, caplog
):
    """A commit failure that also removes the temp zip swallows FileNotFoundError."""
    writer = _snapshotted_writer(tmp_path)
    tmp_zip = tmp_path / "frozen_steps" / "step_5_scheduled.zip.tmp"
    real_save = fs_mod.torch.save

    def _save_then_vanish(obj, buf, *a, **k):
        # First torch.save call (batch.pt): delete the half-written temp zip
        # so the except-handler's tmp.unlink() raises FileNotFoundError.
        if tmp_zip.exists():
            tmp_zip.unlink()
        raise RuntimeError("disk full mid-write")

    monkeypatch.setattr(fs_mod.torch, "save", _save_then_vanish)
    with caplog.at_level(logging.WARNING, logger=_FS_LOGGER):
        result = writer.commit(reason="scheduled")
    monkeypatch.setattr(fs_mod.torch, "save", real_save)

    assert result is None
    assert not (tmp_path / "frozen_steps" / "step_5_scheduled.zip").exists()
    recs = [r for r in caplog.records if "frozen_step commit failed" in r.getMessage()]
    assert recs, caplog.text


def test_invariant_commit_uses_code_snapshot_dir_when_present(tmp_path):
    """When ``<run_dir>/code.snapshot`` exists, the pointer targets it."""
    (tmp_path / "code.snapshot").mkdir()
    writer = _snapshotted_writer(tmp_path)
    out = writer.commit(reason="scheduled")
    with zipfile.ZipFile(out) as zf:
        pointer = zf.read("code_snapshot_pointer.txt").decode("utf-8").strip()
    assert pointer.endswith("code.snapshot")


# ---------------------------------------------------------------------------
# commit — lineage edge + non-fatal failure
# ---------------------------------------------------------------------------


def test_invariant_commit_writes_lineage_edge_with_run_node_id(tmp_path):
    """With a ``run_node_id`` set, commit writes a ``produced_by`` lineage edge."""
    store = _RecordingLineageStore(node_id=99)
    writer = _snapshotted_writer(
        tmp_path, lineage_store=store, run_id="run-x", run_node_id=7
    )
    out = writer.commit(reason="retry")
    assert out is not None
    assert len(store.upserts) == 1
    assert store.upserts[0]["kind"] == "frozen_step"
    assert store.edges == [(7, 99, "produced_by", {"reason": "retry", "step": 5})]


def test_invariant_commit_no_edge_without_run_node_id(tmp_path):
    """A node is upserted but no edge is added when ``run_node_id`` is None."""
    store = _RecordingLineageStore()
    writer = _snapshotted_writer(tmp_path, lineage_store=store, run_id="run-y")
    out = writer.commit(reason="scheduled")
    assert out is not None
    assert len(store.upserts) == 1
    assert store.edges == []


def test_invariant_commit_lineage_failure_is_non_fatal(tmp_path, caplog):
    """A lineage write that raises is swallowed; the on-disk bundle is unaffected."""
    store = _RecordingLineageStore(raise_on="upsert")
    writer = _snapshotted_writer(
        tmp_path, lineage_store=store, run_id="run-z", run_node_id=3
    )
    with caplog.at_level(logging.WARNING, logger=_FS_LOGGER):
        out = writer.commit(reason="scheduled")
    assert out is not None and out.exists()
    recs = [
        r for r in caplog.records
        if "lineage node write failed" in r.getMessage()
    ]
    assert recs, caplog.text
    assert recs[0].exc_info is not None


def test_invariant_commit_lineage_edge_failure_is_non_fatal(tmp_path, caplog):
    """An ``add_edge`` failure after a successful upsert is swallowed."""
    store = _RecordingLineageStore(node_id=42, raise_on="edge")
    writer = _snapshotted_writer(
        tmp_path, lineage_store=store, run_id="run-q", run_node_id=11
    )
    with caplog.at_level(logging.WARNING, logger=_FS_LOGGER):
        out = writer.commit(reason="scheduled")
    assert out is not None and out.exists()
    assert len(store.upserts) == 1
    recs = [
        r for r in caplog.records
        if "lineage node write failed" in r.getMessage()
    ]
    assert recs, caplog.text


# ---------------------------------------------------------------------------
# replay_step_bundle — error paths
# ---------------------------------------------------------------------------


def _make_real_bundle(tmp_path: Path) -> Path:
    torch.manual_seed(3)
    model = _tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    writer = FrozenStepWriter(tmp_path, run_id="r")
    writer.snapshot(
        step=5, ctx=StepContext(step=5, epoch=0),
        batch={
            "input_ids": torch.randint(0, 32, (2, 4)),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "labels": torch.randint(0, 32, (2, 4)),
        },
        model=model, optimizer=opt,
    )
    return writer.commit(reason="scheduled")  # type: ignore[return-value]


def test_invariant_replay_no_model_spec_raises():
    """A bundle whose ``model_spec`` lacks ``name`` and ``_target_`` raises."""
    bundle = FrozenStepBundle(
        step=1, reason="cli", batch={"input_ids": torch.zeros(1, 2, dtype=torch.long)},
        model_spec={},  # no name, no _target_
        model_state_bytes=b"", optimizer_state=None, rng_state=None,
        config_resolved_yaml="",
    )
    with pytest.raises(RuntimeError, match="no model spec"):
        replay_step_bundle(bundle, do_backward=False)


def test_pin_current_behavior_replay_adapter_import_failure_warns(
    tmp_path, monkeypatch, caplog
):
    """Pin: if the text-adapter import fails, replay warns then proceeds.

    Pins the current best-effort behavior — short-name resolution may still
    succeed from the already-populated registry, so replay does not abort.
    """
    path = _make_real_bundle(tmp_path)
    # Force `import lighttrain.builtin_plugins.models.text` to raise ImportError.
    monkeypatch.setitem(sys.modules, "lighttrain.builtin_plugins.models.text", None)
    with caplog.at_level(logging.WARNING, logger=_FS_LOGGER):
        result = replay_step_bundle(path, do_backward=False)
    assert result["step"] == 5
    recs = [
        r for r in caplog.records
        if "model adapter import failed" in r.getMessage()
    ]
    assert recs, caplog.text
    assert recs[0].exc_info is not None


def test_invariant_replay_missing_temp_state_file_is_swallowed(
    tmp_path, monkeypatch
):
    """If the temp safetensors file vanishes before unlink, replay still returns."""
    path = _make_real_bundle(tmp_path)
    real_load = fs_mod.load_state

    def _load_then_vanish(model, p, *a, **k):
        out = real_load(model, p, *a, **k)
        Path(p).unlink()  # delete so the finally-clause unlink raises
        return out

    monkeypatch.setattr(fs_mod, "load_state", _load_then_vanish)
    result = replay_step_bundle(path, do_backward=False)
    assert result["step"] == 5
    assert result["logits_shape"] is not None


def test_invariant_replay_rng_restore_failure_warns(tmp_path, monkeypatch, caplog):
    """An RNG-restore failure during replay is swallowed-with-warning."""
    path = _make_real_bundle(tmp_path)

    def _boom(_state):
        raise RuntimeError("replay rng restore broke")

    monkeypatch.setattr(fs_mod, "restore_rng_state", _boom)
    with caplog.at_level(logging.WARNING, logger=_FS_LOGGER):
        result = replay_step_bundle(path, do_backward=False)
    assert result["step"] == 5
    recs = [r for r in caplog.records if "RNG restore failed" in r.getMessage()]
    assert recs, caplog.text
    assert recs[0].exc_info is not None


def test_invariant_replay_accepts_prebuilt_bundle_object(tmp_path):
    """Passing a ``FrozenStepBundle`` (not a path) skips the read step."""
    path = _make_real_bundle(tmp_path)
    bundle = read_frozen_step_bundle(path)
    result = replay_step_bundle(bundle, do_backward=False)
    assert result["step"] == 5
    assert result["reason"] == "scheduled"


# ---------------------------------------------------------------------------
# _infer_model_spec — peft ImportError fallthrough
# ---------------------------------------------------------------------------


def test_invariant_infer_spec_peft_import_error_falls_through(monkeypatch):
    """A peft ``ImportError`` is caught; inference falls back to the short name."""
    # Block the peft module so the inner `from ...peft import ...` raises.
    monkeypatch.setitem(sys.modules, "lighttrain.builtin_plugins.models.peft", None)
    spec = _infer_model_spec(_tiny())
    assert spec["name"] == "tiny_lm"
    assert spec["params"]["vocab_size"] == 32


def test_invariant_infer_spec_unregistered_module_uses_target(monkeypatch):
    """A model not in the ``model`` registry falls back to a ``_target_`` spec."""
    monkeypatch.setitem(sys.modules, "lighttrain.builtin_plugins.models.peft", None)
    spec = _infer_model_spec(nn.Linear(2, 2))
    assert "_target_" in spec
    assert spec["_target_"].endswith(":Linear")
    assert spec["params"] == {}


# ---------------------------------------------------------------------------
# FrozenStepBundle dataclass
# ---------------------------------------------------------------------------


def test_invariant_bundle_metadata_defaults_to_empty_dict():
    """``FrozenStepBundle.metadata`` defaults to a fresh empty dict per instance."""
    b1 = FrozenStepBundle(
        step=0, reason="cli", batch={}, model_spec={}, model_state_bytes=b"",
        optimizer_state=None, rng_state=None, config_resolved_yaml="",
    )
    b2 = FrozenStepBundle(
        step=1, reason="cli", batch={}, model_spec={}, model_state_bytes=b"",
        optimizer_state=None, rng_state=None, config_resolved_yaml="",
    )
    assert b1.metadata == {}
    b1.metadata["k"] = "v"
    assert b2.metadata == {}  # not shared


def test_invariant_read_bundle_round_trips_metadata(tmp_path):
    """A committed bundle's metadata block survives the read round-trip."""
    path = _make_real_bundle(tmp_path)
    bundle = read_frozen_step_bundle(path)
    assert bundle.metadata["step"] == 5
    assert bundle.metadata["reason"] == "scheduled"
    assert bundle.metadata["model_spec"]["name"] == "tiny_lm"
    # step_metadata.json is internally consistent with the bundle fields.
    with zipfile.ZipFile(path) as zf:
        meta = json.loads(zf.read("step_metadata.json"))
    assert meta["epoch"] == 0
