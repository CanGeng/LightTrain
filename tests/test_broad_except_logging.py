"""Regression: broad best-effort swallows are no longer *silent*.

The refactor that added these tests left control flow untouched — best-effort
handlers still swallow and the run does not crash — but the swallowed exception
must now surface as a ``logging.WARNING`` with a traceback (``exc_info``), so a
failure like the frozen_step CI flake can be diagnosed from logs instead of an
opaque empty-bundle assertion.
"""

from __future__ import annotations

import logging

import torch
from omegaconf import OmegaConf

from lighttrain.builtin_plugins.models.adapters.tiny_lm import TinyCausalLM
from lighttrain.config._loader import _leaf_exists
from lighttrain.diagnostics import frozen_step as fs_mod
from lighttrain.diagnostics.frozen_step import FrozenStepWriter


class _Ctx:
    epoch = 0


def test_frozen_step_commit_failure_logs_warning(tmp_path, monkeypatch, caplog):
    model = TinyCausalLM(
        vocab_size=260, d_model=16, n_layers=1, n_heads=2, max_seq_len=32
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "input_ids": torch.randint(0, 260, (2, 32)),
        "attention_mask": torch.ones(2, 32, dtype=torch.long),
    }
    writer = FrozenStepWriter(tmp_path, mode="lab")
    writer.snapshot(
        step=1,
        ctx=_Ctx(),
        batch=batch,
        model=model,
        optimizer=opt,
        config_resolved_yaml="x: 1\n",
    )

    # Force the commit's safetensors write to blow up mid-bundle.
    def _boom(*_a, **_k):
        raise RuntimeError("boom-disk-full")

    monkeypatch.setattr(fs_mod, "_save_model", _boom)

    with caplog.at_level(
        logging.WARNING, logger="lighttrain.diagnostics.frozen_step"
    ):
        result = writer.commit(reason="scheduled")

    # Behavior unchanged: swallowed, returns None, no bundle left on disk.
    assert result is None
    assert not list((tmp_path / "frozen_steps").glob("*.zip"))

    # No longer silent: the swallowed exception is logged *with* a traceback.
    recs = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "frozen_step commit failed" in r.getMessage()
    ]
    assert recs, caplog.text
    assert recs[0].exc_info is not None


def test_config_leaf_descend_failure_logs_and_returns_false(caplog):
    # ``${nonexistent}`` raises on resolution when descending into "b".
    cfg = OmegaConf.create({"a": {"b": "${nonexistent}"}})

    with caplog.at_level(logging.WARNING, logger="lighttrain.config._loader"):
        result = _leaf_exists(cfg, ["a", "b", "c"])

    # Behavior unchanged: descend failure is treated as "leaf absent".
    assert result is False

    recs = [r for r in caplog.records if "descend into key" in r.getMessage()]
    assert recs, caplog.text
    assert recs[0].exc_info is not None
