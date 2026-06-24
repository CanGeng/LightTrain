"""Seam + error-path tests for ``lighttrain.trainers.base.Trainer``.

Complements tests/trainers/test_base.py (constructor / step-normalize /
state_dict) by exercising the loop seams it didn't reach:

* custom ``forward_loss`` → ``apply_update`` path + ``_split_forward_result``;
* ``eval`` (model/loader/loss_fn guards + ModelOutput-wrap + logger);
* ``predict`` (guards + ModelOutput-wrap + CPU outputs);
* periodic hooks ``_maybe_log`` / ``_maybe_eval`` / ``_maybe_save``;
* ``_save_with_events`` (no-manager + manifest re-read failure);
* ``_collect_state`` (data_module state + best-effort failures);
* ``load_checkpoint`` (no-manager guard + restore + best-effort failures);
* ``_write_crash_bundle`` (rank guard + run-dir guard + write + OOM).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lighttrain.protocols import StepOutput
from lighttrain.trainers.base import Trainer

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _FakeEngine:
    def step(self, batch, ctx):
        return {"loss": torch.tensor(0.1), "ppl": 2.0}


class _DM:
    """Configurable data_module: each loader is whatever was passed in."""

    def __init__(self, *, val=None, predict=None, train=None):
        self._val, self._predict, self._train = val, predict, train

    def val_loader(self):
        return self._val

    def predict_loader(self):
        return self._predict

    def train_loader(self):
        return self._train


class _DictModel(nn.Module):
    """forward() returns a plain dict (not a ModelOutput) → wrap branch."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 2)

    def forward(self, **batch):
        return {"logits": torch.zeros(1, 2)}


class _NonMainPctx:
    is_main_process = False


def _kwargs(**over):
    base = dict(engine=_FakeEngine(), data_module=_DM(), optimizer=MagicMock(), max_steps=3)
    base.update(over)
    return base


def _loss_fn(out, batch, ctx):
    return {"loss": torch.tensor(0.5)}


# ===========================================================================
# custom forward_loss path + _split_forward_result
# ===========================================================================

@pytest.mark.parametrize(
    "result, exp_loss, exp_metric_key",
    [
        ((torch.tensor(1.0), {"acc": 0.9}), 1.0, "acc"),   # 2-tuple (loss, metrics)
        ({"loss": torch.tensor(2.0), "x": 1}, 2.0, "x"),   # Mapping
        (torch.tensor(3.0), 3.0, "loss"),                  # bare loss
    ],
)
def test_split_forward_result_shapes(result, exp_loss, exp_metric_key):
    """``_split_forward_result`` normalizes tuple / Mapping / bare-loss."""
    loss, metrics = Trainer._split_forward_result(result)
    assert float(loss) == exp_loss
    assert exp_metric_key in metrics


def test_custom_forward_loss_routes_through_apply_update():
    """A trainer whose ``forward_loss`` returns a real loss drives the shared
    ``apply_update`` backward half and reports ``grad_norm``."""
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    class _Custom(Trainer):
        def forward_loss(self, batch):
            return self.model(batch["x"]).sum()

    trainer = _Custom(**_kwargs(model=model, optimizer=opt))
    out = trainer.train_step({"x": torch.ones(1, 2)})
    assert isinstance(out, StepOutput)
    assert "grad_norm" in out.metrics


def test_before_step_default_is_noop():
    """The base ``before_step`` hook is a no-op returning None."""
    trainer = Trainer(**_kwargs())
    assert trainer.before_step({}) is None


def test_constructor_empty_models_and_optimizers_when_none():
    """With no model and a falsy optimizer, the named sets are empty dicts."""
    trainer = Trainer(**_kwargs(model=None, optimizer=None))
    assert trainer.models == {}
    assert trainer.optimizers == {}


# ===========================================================================
# eval
# ===========================================================================

def test_eval_raises_when_model_is_none():
    trainer = Trainer(**_kwargs(model=None))
    with pytest.raises(RuntimeError, match="model is not set"):
        trainer.eval()


def test_eval_returns_empty_when_no_val_loader():
    trainer = Trainer(**_kwargs(model=_DictModel(), data_module=_DM(val=None)))
    assert trainer.eval() == {}


def test_eval_returns_empty_when_no_loss_fn():
    trainer = Trainer(**_kwargs(model=_DictModel(), data_module=_DM(val=[{"a": 1}])))
    # ctx.loss_fn defaults to None
    assert trainer.eval() == {}


def test_eval_full_path_wraps_output_and_logs():
    """A loader + loss_fn produces val_loss; a non-ModelOutput forward result is
    wrapped; the logger receives the metric."""
    logger = MagicMock()
    trainer = Trainer(**_kwargs(model=_DictModel(), data_module=_DM(val=[{"a": 1}, {"a": 2}]), logger=logger))
    trainer.ctx.loss_fn = _loss_fn
    metrics = trainer.eval()
    assert metrics["val_loss"] == pytest.approx(0.5)
    logger.log_dict.assert_called_once()


# ===========================================================================
# predict
# ===========================================================================

def test_predict_raises_when_model_is_none():
    trainer = Trainer(**_kwargs(model=None))
    with pytest.raises(RuntimeError, match="model is not set"):
        trainer.predict()


def test_predict_raises_when_no_loader_and_no_data_module():
    trainer = Trainer(**_kwargs(model=_DictModel(), data_module=None))
    with pytest.raises(RuntimeError, match="no loader and no data_module"):
        trainer.predict()


def test_predict_raises_when_predict_loader_returns_none():
    trainer = Trainer(**_kwargs(model=_DictModel(), data_module=_DM(predict=None)))
    with pytest.raises(RuntimeError, match="returned None"):
        trainer.predict()


def test_predict_full_path_wraps_and_returns_cpu_outputs():
    """predict wraps a non-ModelOutput result and returns CPU output dicts."""
    trainer = Trainer(**_kwargs(model=_DictModel(), data_module=_DM(predict=[{"a": 1}])))
    results = trainer.predict()
    assert len(results) == 1
    assert "logits" in results[0]
    assert results[0]["logits"].device.type == "cpu"


# ===========================================================================
# periodic hooks
# ===========================================================================

def test_maybe_log_returns_early_for_non_main_rank():
    trainer = Trainer(**_kwargs(logger=MagicMock()))
    trainer.ctx.parallel_ctx = _NonMainPctx()
    trainer.ctx.step = trainer.log_every  # would log if main
    trainer._maybe_log({"loss": 0.1})
    trainer.logger.log_dict.assert_not_called()


def test_maybe_log_filters_to_finite_scalars():
    logger = MagicMock()
    trainer = Trainer(**_kwargs(logger=logger, log_every=1))
    trainer.ctx.step = 1
    trainer._maybe_log({"loss": 0.1, "flag": True, "bad": float("nan"), "name": "x"})
    logged = logger.log_dict.call_args.args[0]
    assert logged == {"loss": 0.1}  # bool / nan / str dropped


def test_maybe_eval_skips_when_val_every_nonpositive():
    trainer = Trainer(**_kwargs(model=_DictModel(), val_every=0))
    trainer.eval = MagicMock()  # type: ignore[method-assign]
    trainer._maybe_eval()
    trainer.eval.assert_not_called()


def test_maybe_eval_runs_on_schedule():
    trainer = Trainer(**_kwargs(model=_DictModel(), val_every=2))
    trainer.eval = MagicMock()  # type: ignore[method-assign]
    trainer.ctx.step = 4
    trainer._maybe_eval()
    trainer.eval.assert_called_once()


# ===========================================================================
# save
# ===========================================================================

class _CkptMgr:
    def __init__(self, tmp, *, bad_manifest=False):
        self.tmp = tmp
        self.bad_manifest = bad_manifest
        self.saved: list = []
        self.to_load: dict = {}

    def save(self, *, step, state, kind, extras, parallel_ctx):
        d = self.tmp / f"ckpt_{step}_{kind}"
        d.mkdir(parents=True, exist_ok=True)
        if self.bad_manifest:
            (d / "manifest.json").write_text("{ not json", encoding="utf-8")
        self.saved.append((step, kind))
        return d

    def load(self, path):
        return dict(self.to_load)


def test_save_with_events_noop_without_manager():
    trainer = Trainer(**_kwargs(ckpt_manager=None))
    assert trainer._save_with_events(kind="step") is None


def test_maybe_save_writes_on_schedule(tmp_path):
    mgr = _CkptMgr(tmp_path)
    trainer = Trainer(**_kwargs(model=_DictModel(), ckpt_manager=mgr, ckpt_every=2))
    trainer.ctx.step = 2
    trainer._maybe_save({"loss": 0.1})
    assert mgr.saved == [(2, "step")]


def test_maybe_save_skips_off_schedule(tmp_path):
    mgr = _CkptMgr(tmp_path)
    trainer = Trainer(**_kwargs(model=_DictModel(), ckpt_manager=mgr, ckpt_every=5))
    trainer.ctx.step = 3
    trainer._maybe_save({"loss": 0.1})
    assert mgr.saved == []


def test_save_with_events_survives_bad_manifest(tmp_path):
    """A corrupt manifest.json after save is caught (best-effort re-read)."""
    mgr = _CkptMgr(tmp_path, bad_manifest=True)
    trainer = Trainer(**_kwargs(model=_DictModel(), ckpt_manager=mgr))
    path = trainer._save_with_events(kind="step")
    assert path is not None  # save still returns the path despite manifest error


# ===========================================================================
# _collect_state
# ===========================================================================

def test_collect_state_includes_model_and_data_module(tmp_path):
    class _DMState(_DM):
        def state_dict(self):
            return {"cursor": 7}

    trainer = Trainer(**_kwargs(model=_DictModel(), data_module=_DMState()))
    state = trainer._collect_state()
    assert "model" in state and "trainer" in state and "rng" in state
    assert state["data_module"] == {"cursor": 7}


def test_collect_state_swallows_data_module_failure():
    class _DMBoom(_DM):
        def state_dict(self):
            raise RuntimeError("boom")

    trainer = Trainer(**_kwargs(model=_DictModel(), data_module=_DMBoom()))
    state = trainer._collect_state()  # must not raise
    assert "data_module" not in state


def test_collect_state_swallows_rng_capture_failure(monkeypatch):
    import lighttrain.trainers.base as base_mod

    monkeypatch.setattr(base_mod, "rng_state", lambda: (_ for _ in ()).throw(RuntimeError("no rng")))
    trainer = Trainer(**_kwargs(model=_DictModel()))
    state = trainer._collect_state()  # warning logged, no raise
    assert "rng" not in state


# ===========================================================================
# load_checkpoint
# ===========================================================================

def test_load_checkpoint_raises_without_manager():
    trainer = Trainer(**_kwargs(ckpt_manager=None))
    with pytest.raises(RuntimeError, match="no ckpt_manager"):
        trainer.load_checkpoint("anywhere")


def test_load_checkpoint_restores_trainer_state(tmp_path):
    mgr = _CkptMgr(tmp_path)
    mgr.to_load = {"trainer": {"step": 9, "epoch": 1, "global_step": 9, "batch_in_epoch": 0}}
    trainer = Trainer(**_kwargs(model=_DictModel(), ckpt_manager=mgr, max_steps=20))
    trainer.load_checkpoint("p")
    assert trainer.ctx.step == 9


def test_load_checkpoint_swallows_data_module_and_seek_and_rng_failures(tmp_path, monkeypatch):
    """The three best-effort restores (data_module load, sampler seek, RNG)
    each swallow exceptions so resume never crashes."""

    class _DMBoom(_DM):
        def load_state_dict(self, sd):
            raise RuntimeError("dm load boom")

        def seek(self, epoch, batch):
            raise RuntimeError("seek boom")

    mgr = _CkptMgr(tmp_path)
    mgr.to_load = {"data_module": {"x": 1}, "rng": {"bad": "state"}}
    import lighttrain.trainers.base as base_mod
    monkeypatch.setattr(
        base_mod, "restore_rng_state",
        lambda rng: (_ for _ in ()).throw(RuntimeError("rng boom")),
    )
    trainer = Trainer(**_kwargs(model=_DictModel(), data_module=_DMBoom(), ckpt_manager=mgr))
    trainer.load_checkpoint("p")  # must not raise despite all three failing


# ===========================================================================
# _write_crash_bundle
# ===========================================================================

def test_write_crash_bundle_returns_early_for_non_main_rank():
    trainer = Trainer(**_kwargs(model=_DictModel()))
    trainer.ctx.parallel_ctx = _NonMainPctx()
    trainer._run_dir = "/nonexistent"  # would be used if main
    # Non-main → returns before touching run_dir; no raise.
    trainer._write_crash_bundle(RuntimeError("x"), {"a": 1}, {"loss": 0.1})


def test_write_crash_bundle_returns_early_when_no_run_dir():
    trainer = Trainer(**_kwargs(model=_DictModel()))
    # No _run_dir attribute → returns silently.
    trainer._write_crash_bundle(RuntimeError("x"), {"a": 1}, {"loss": 0.1})


def test_write_crash_bundle_writes_bundle_and_oom_report(tmp_path):
    """With a run_dir on the main rank, a crash bundle is written; an
    OOM-looking exception also triggers the OOM report path."""
    trainer = Trainer(**_kwargs(model=_DictModel()))
    trainer._run_dir = tmp_path
    exc = RuntimeError("CUDA out of memory. Tried to allocate ...")
    trainer._write_crash_bundle(exc, {"input_ids": torch.zeros(1, 2, dtype=torch.long)}, {"loss": 0.1})
    # A diagnostics dir should now exist under the run dir.
    assert (tmp_path / "diagnostics").exists()


def test_write_crash_bundle_swallows_inner_write_failures(tmp_path, monkeypatch):
    """If the bundle / OOM writers themselves raise, both are swallowed."""
    monkeypatch.setattr(
        "lighttrain.observability.diagnostics.crash_bundle.write_crash_bundle",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bundle boom")),
    )
    monkeypatch.setattr(
        "lighttrain.observability.diagnostics.oom_report.write_oom_report",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("oom boom")),
    )
    trainer = Trainer(**_kwargs(model=_DictModel()))
    trainer._run_dir = tmp_path
    exc = RuntimeError("CUDA out of memory")  # is_oom_exception → True
    trainer._write_crash_bundle(exc, {"a": 1}, {"loss": 0.1})  # must not raise


# ===========================================================================
# remaining periodic-hook + save/load branches
# ===========================================================================

def test_maybe_log_skips_off_schedule():
    logger = MagicMock()
    trainer = Trainer(**_kwargs(logger=logger, log_every=2))
    trainer.ctx.step = 1  # 1 % 2 != 0
    trainer._maybe_log({"loss": 0.1})
    logger.log_dict.assert_not_called()


def test_maybe_eval_skips_off_schedule():
    trainer = Trainer(**_kwargs(model=_DictModel(), val_every=3))
    trainer.eval = MagicMock()  # type: ignore[method-assign]
    trainer.ctx.step = 2  # 2 % 3 != 0
    trainer._maybe_eval()
    trainer.eval.assert_not_called()


def test_maybe_save_best_writes_for_flagged_callback(tmp_path):
    """A callback with ``should_save=True`` triggers a 'best' checkpoint and the
    flag is reset."""
    mgr = _CkptMgr(tmp_path)

    class _BestCB:
        should_save = True
        monitor = "val_loss"
        last_value = 0.3

    cb = _BestCB()
    trainer = Trainer(**_kwargs(model=_DictModel(), ckpt_manager=mgr, callbacks=[cb]))
    trainer._maybe_save_best({"loss": 0.1})
    assert mgr.saved == [(0, "best")]
    assert cb.should_save is False


def test_final_save_writes_when_off_ckpt_schedule(tmp_path):
    mgr = _CkptMgr(tmp_path)
    trainer = Trainer(**_kwargs(model=_DictModel(), ckpt_manager=mgr, ckpt_every=5))
    trainer.ctx.step = 3  # not a multiple → a final step ckpt is warranted
    trainer._final_save({"loss": 0.1})
    assert mgr.saved == [(3, "step")]


def test_final_save_skips_when_on_ckpt_schedule(tmp_path):
    mgr = _CkptMgr(tmp_path)
    trainer = Trainer(**_kwargs(model=_DictModel(), ckpt_manager=mgr, ckpt_every=2))
    trainer.ctx.step = 2  # already saved by _maybe_save → skip
    trainer._final_save({"loss": 0.1})
    assert mgr.saved == []


def test_collect_state_includes_scheduler():
    sched = MagicMock()
    sched.state_dict.return_value = {"lr": 0.01}
    trainer = Trainer(**_kwargs(model=_DictModel(), scheduler=sched))
    state = trainer._collect_state()
    assert state["scheduler"] == {"lr": 0.01}


def test_load_checkpoint_restores_model_optimizer_scheduler(tmp_path):
    model = _DictModel()
    opt, sched = MagicMock(), MagicMock()
    mgr = _CkptMgr(tmp_path)
    mgr.to_load = {
        "model": model.state_dict(),
        "optimizer": {"o": 1},
        "scheduler": {"s": 1},
    }
    trainer = Trainer(**_kwargs(model=model, optimizer=opt, scheduler=sched, ckpt_manager=mgr))
    trainer.load_checkpoint("p")
    opt.load_state_dict.assert_called_once_with({"o": 1})
    sched.load_state_dict.assert_called_once_with({"s": 1})
