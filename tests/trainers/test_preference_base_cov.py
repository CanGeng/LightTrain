"""Coverage-completion tests for ``lighttrain.builtin_plugins.trainers._preference_base``.

Complements tests/trainers/test_preference_base.py (lifecycle / loss-seam math)
by exercising the seams and error paths it didn't reach:

* ``_seq_logps_and_nll`` model-output normalization — Mapping (dict-of-logits)
  and bare-tensor (raw logits) branches (the ModelOutput branch is already
  covered elsewhere);
* ``fit`` pre-flight guards — model is None / optimizer is None / no
  preference-loss-configured;
* ``fit`` epoch-rollover ``StopIteration`` path (loader shorter than steps:
  on_epoch_end → epoch++ → on_epoch_begin → re-iterate);
* ``fit`` exception path — ``on_exception`` dispatch + re-raise, the secondary
  on_exception suppression, the on_train_end-dispatch suppression, and the
  logger.flush suppression;
* ``_preference_step`` no-loss guard (second guard, distinct from fit's);
* trivial ``eval`` / ``predict`` overrides;
* periodic ``_maybe_log`` / ``_maybe_eval`` / ``_maybe_save`` branches;
* ``state_dict`` ``ref_namespace`` round-trip.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.losses.preference import DPOLoss
from lighttrain.builtin_plugins.trainers._preference_base import (
    PreferenceTrainer,
    _seq_logps_and_nll,
)
from lighttrain.exceptions import BatchValidationError
from lighttrain.protocols import ModelOutput

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _TinyLM(nn.Module):
    """Default model: returns a ModelOutput wrapping logits."""

    def __init__(self, V: int = 16, D: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.proj = nn.Linear(D, V, bias=False)

    def _logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.emb(input_ids))

    def forward(self, input_ids, attention_mask=None, **_):
        return ModelOutput(outputs={"logits": self._logits(input_ids)})


class _DictLM(_TinyLM):
    """forward() returns a plain dict (Mapping branch: ``out['logits']``)."""

    def forward(self, input_ids, attention_mask=None, **_):
        return {"logits": self._logits(input_ids)}


class _BareLM(_TinyLM):
    """forward() returns the raw logits tensor (else branch: ``logits = out``)."""

    def forward(self, input_ids, attention_mask=None, **_):
        return self._logits(input_ids)


class _FakeEngine:
    pass


class _NonMainPctx:
    is_main_process = False


def _pref_batch(V: int = 16, T: int = 5, B: int = 2) -> dict:
    """A well-formed preference batch with reference log-probs."""
    torch.manual_seed(0)
    return {
        "chosen_input_ids": torch.randint(0, V, (B, T)),
        "chosen_attention_mask": torch.ones(B, T, dtype=torch.long),
        "chosen_labels": torch.randint(0, V, (B, T)),
        "rejected_input_ids": torch.randint(0, V, (B, T)),
        "rejected_attention_mask": torch.ones(B, T, dtype=torch.long),
        "rejected_labels": torch.randint(0, V, (B, T)),
        "aux.ref.chosen_logprobs": torch.randn(B) - 1.0,
        "aux.ref.rejected_logprobs": torch.randn(B) - 2.0,
    }


class _ListDM:
    """DataModule whose train_loader returns a re-iterable list of batches."""

    def __init__(self, batches: list) -> None:
        self._batches = batches

    def train_loader(self):
        return list(self._batches)


_DEFAULT = object()


def _make_trainer(
    *,
    model=_DEFAULT,
    optimizer="auto",
    data_module=None,
    callbacks=None,
    logger=None,
    ckpt_manager=None,
    max_steps: int = 1,
    set_loss: bool = True,
    **kw,
) -> PreferenceTrainer:
    """Construct a PreferenceTrainer with sensible defaults for these tests.

    ``model=None`` is honored verbatim (to exercise the no-model guard); the
    default sentinel builds a fresh ``_TinyLM``.
    """
    if model is _DEFAULT:
        model = _TinyLM()
    if optimizer == "auto":
        optimizer = (
            torch.optim.AdamW(model.parameters(), lr=1e-3)
            if model is not None
            else None
        )
    if data_module is None:
        data_module = _ListDM([_pref_batch() for _ in range(max_steps + 1)])
    trainer = PreferenceTrainer(
        engine=_FakeEngine(),
        data_module=data_module,
        optimizer=optimizer,
        model=model,
        callbacks=callbacks,
        logger=logger,
        ckpt_manager=ckpt_manager,
        max_steps=max_steps,
        **kw,
    )
    if set_loss:
        trainer.ctx.loss_fn = DPOLoss(beta=0.1)
    return trainer


# ===========================================================================
# _seq_logps_and_nll — model-output normalization branches (lines 67-70)
# ===========================================================================


@pytest.mark.parametrize("model_cls", [_DictLM, _BareLM])
def test_invariant_seq_logps_handles_mapping_and_bare_logits(model_cls):
    """``_seq_logps_and_nll`` accepts a dict-of-logits (Mapping) or a raw logits
    tensor identically to a ModelOutput, yielding finite (B,) logps + nonneg nll."""
    torch.manual_seed(0)
    B, T, V = 2, 5, 16
    model = model_cls(V=V)
    ids = torch.randint(0, V, (B, T))
    logps, nll = _seq_logps_and_nll(model, ids, None, ids.clone())
    assert logps.shape == (B,)
    assert nll.shape == (B,)
    assert torch.isfinite(logps).all()
    assert (nll >= 0).all()


def test_invariant_seq_logps_mapping_matches_modeloutput():
    """A dict-returning and a ModelOutput-returning model with identical weights
    produce identical logps/nll (the branch is purely an unwrap)."""
    torch.manual_seed(0)
    B, T, V = 2, 4, 16
    mo_model = _TinyLM(V=V)
    dict_model = _DictLM(V=V)
    dict_model.load_state_dict(mo_model.state_dict())
    ids = torch.randint(0, V, (B, T))

    lp_a, nll_a = _seq_logps_and_nll(mo_model, ids, None, ids.clone())
    lp_b, nll_b = _seq_logps_and_nll(dict_model, ids, None, ids.clone())
    assert torch.allclose(lp_a, lp_b)
    assert torch.allclose(nll_a, nll_b)


# ===========================================================================
# fit — pre-flight guards (lines 159, 161, 166)
# ===========================================================================


def test_invariant_fit_raises_when_model_is_none():
    """fit() refuses to run with no model set."""
    trainer = _make_trainer(model=None, optimizer=MagicMock())
    with pytest.raises(RuntimeError, match="model is not set"):
        trainer.fit()


def test_invariant_fit_raises_when_optimizer_is_none():
    """fit() refuses to run with a model but no optimizer."""
    trainer = _make_trainer(optimizer=None)
    with pytest.raises(RuntimeError, match="optimizer is not set"):
        trainer.fit()


def test_invariant_fit_raises_when_no_preference_loss_configured():
    """fit() refuses to run when consumes_objective and ctx.loss_fn is None."""
    trainer = _make_trainer(set_loss=False)
    assert trainer.consumes_objective is True
    assert trainer.ctx.loss_fn is None
    with pytest.raises(RuntimeError, match="no preference loss configured"):
        trainer.fit()


# ===========================================================================
# fit — epoch rollover via StopIteration (lines 184-189)
# ===========================================================================


def test_invariant_fit_rolls_over_epoch_when_loader_exhausts():
    """With max_steps exceeding the loader length, fit re-iterates: on_epoch_end
    fires, ctx.epoch increments, on_epoch_begin fires again, and training
    continues to the step target."""
    events: list[str] = []

    class _Rec:
        def on_epoch_begin(self, **_):
            events.append("begin")

        def on_epoch_end(self, **_):
            events.append("end")

    # Loader yields only 2 batches but we ask for 3 steps → one rollover.
    dm = _ListDM([_pref_batch(), _pref_batch()])
    trainer = _make_trainer(data_module=dm, callbacks=[_Rec()], max_steps=3)
    trainer.fit()

    assert trainer.ctx.step == 3
    assert trainer.ctx.epoch == 1
    # begin, (2 steps), end, begin, (1 step) → at least one end between begins.
    assert "end" in events
    assert events.count("begin") == 2


# ===========================================================================
# fit — exception path (lines 222-230, 236-237, 239-242)
# ===========================================================================


def test_invariant_fit_dispatches_on_exception_then_reraises():
    """A step-level error (malformed batch → BatchValidationError) is caught:
    on_exception is dispatched with the exception, then re-raised out of fit."""
    seen: dict = {}

    class _ExcCB:
        def on_exception(self, exception=None, step=None, **_):
            seen["exc"] = exception
            seen["step"] = step

    # Batch missing required preference keys → validate_batch raises.
    dm = _ListDM([{"not_a_pref_key": torch.zeros(1)}])
    trainer = _make_trainer(data_module=dm, callbacks=[_ExcCB()], max_steps=1)
    with pytest.raises(BatchValidationError):
        trainer.fit()
    assert isinstance(seen["exc"], BatchValidationError)
    assert seen["step"] == 0


def test_invariant_fit_suppresses_secondary_on_exception_failure(caplog):
    """If the on_exception dispatch itself raises (critical callback), the
    secondary failure is logged-and-swallowed and the ORIGINAL error re-raises."""

    class _BadExcCB:
        critical = True  # so its raise propagates out of dispatch

        def on_exception(self, **_):
            raise ValueError("on_exception boom")

    dm = _ListDM([{"bad": torch.zeros(1)}])
    trainer = _make_trainer(data_module=dm, callbacks=[_BadExcCB()], max_steps=1)
    with caplog.at_level("WARNING"):
        with pytest.raises(BatchValidationError):  # original, not the secondary
            trainer.fit()
    assert any("on_exception" in r.message for r in caplog.records)


def test_invariant_fit_suppresses_on_train_end_dispatch_failure(caplog):
    """A critical callback that raises in on_train_end has its failure swallowed
    in the finally-block (training still completes / re-raises normally)."""

    class _BadEndCB:
        critical = True

        def on_train_end(self, **_):
            raise ValueError("train_end boom")

    trainer = _make_trainer(callbacks=[_BadEndCB()], max_steps=1)
    with caplog.at_level("WARNING"):
        trainer.fit()  # completes despite the on_train_end failure
    assert any("on_train_end" in r.message for r in caplog.records)
    assert trainer.ctx.step == 1


def test_invariant_fit_suppresses_logger_flush_failure(caplog):
    """A logger whose flush() raises is swallowed in the finally block."""
    logger = MagicMock()
    logger.flush.side_effect = RuntimeError("flush boom")
    trainer = _make_trainer(logger=logger, max_steps=1)
    with caplog.at_level("WARNING"):
        trainer.fit()
    logger.flush.assert_called_once()
    assert any("logger.flush" in r.message for r in caplog.records)


def test_invariant_fit_flushes_logger_on_success():
    """On a clean run the logger is flushed exactly once (finally block)."""
    logger = MagicMock()
    trainer = _make_trainer(logger=logger, max_steps=1)
    trainer.fit()
    logger.flush.assert_called_once()


# ===========================================================================
# _preference_step — second no-loss guard (line 292)
# ===========================================================================


def test_invariant_preference_step_raises_without_loss_fn():
    """Calling _preference_step directly with no ctx.loss_fn trips the second
    guard (distinct from fit's pre-flight check)."""
    trainer = _make_trainer(set_loss=False)
    assert trainer.ctx.loss_fn is None
    with pytest.raises(RuntimeError, match="no preference loss configured"):
        trainer._preference_step(_pref_batch())


# ===========================================================================
# eval / predict trivial overrides (lines 313, 316)
# ===========================================================================


def test_invariant_eval_returns_empty_dict():
    """PreferenceTrainer.eval is a no-op returning {}."""
    trainer = _make_trainer()
    assert trainer.eval() == {}
    assert trainer.eval("ignored", kw=1) == {}


def test_invariant_predict_returns_empty_list():
    """PreferenceTrainer.predict is a no-op returning []."""
    trainer = _make_trainer()
    assert trainer.predict() == []
    assert trainer.predict("ignored", kw=1) == []


# ===========================================================================
# _maybe_log branches (lines 320, 323-325, 329-330)
# ===========================================================================


def test_invariant_maybe_log_returns_early_for_non_main_rank():
    """Non-main rank: _maybe_log returns before touching the logger."""
    logger = MagicMock()
    trainer = _make_trainer(logger=logger, log_every=1)
    trainer.ctx.parallel_ctx = _NonMainPctx()  # type: ignore[assignment]
    trainer.ctx.step = 1
    trainer._maybe_log({"loss": 0.1})
    logger.log_dict.assert_not_called()


def test_invariant_maybe_log_noop_without_logger_or_metrics():
    """No logger → silent; empty metrics → silent (no crash)."""
    trainer_no_logger = _make_trainer(logger=None, log_every=1)
    trainer_no_logger.ctx.step = 1
    trainer_no_logger._maybe_log({"loss": 0.1})  # logger is None → returns

    logger = MagicMock()
    trainer_empty = _make_trainer(logger=logger, log_every=1)
    trainer_empty.ctx.step = 1
    trainer_empty._maybe_log({})  # empty metrics → returns
    logger.log_dict.assert_not_called()


def test_invariant_maybe_log_skips_off_schedule():
    """Off the log_every cadence, nothing is logged."""
    logger = MagicMock()
    trainer = _make_trainer(logger=logger, log_every=2)
    trainer.ctx.step = 1  # 1 % 2 != 0
    trainer._maybe_log({"loss": 0.1})
    logger.log_dict.assert_not_called()


def test_invariant_maybe_log_filters_to_finite_scalars():
    """On schedule, only finite int/float (not bool/nan/str) scalars are logged."""
    logger = MagicMock()
    trainer = _make_trainer(logger=logger, log_every=1)
    trainer.ctx.step = 1
    trainer._maybe_log(
        {"loss": 0.25, "flag": True, "bad": float("nan"), "name": "x", "n": 3}
    )
    logged = logger.log_dict.call_args.args[0]
    assert logged == {"loss": pytest.approx(0.25), "n": pytest.approx(3.0)}
    assert logger.log_dict.call_args.kwargs["step"] == 1


def test_invariant_maybe_log_skips_when_all_scalars_filtered():
    """If every metric is non-scalar/non-finite, log_dict is never called."""
    logger = MagicMock()
    trainer = _make_trainer(logger=logger, log_every=1)
    trainer.ctx.step = 1
    trainer._maybe_log({"flag": True, "bad": float("inf"), "name": "x"})
    logger.log_dict.assert_not_called()


# ===========================================================================
# _maybe_eval branches (lines 335-337)
# ===========================================================================


def test_invariant_maybe_eval_skips_when_val_every_nonpositive():
    """val_every <= 0 disables eval entirely."""
    trainer = _make_trainer(val_every=0)
    trainer.eval = MagicMock()  # type: ignore[method-assign]
    trainer._maybe_eval()
    trainer.eval.assert_not_called()


def test_invariant_maybe_eval_skips_off_schedule():
    """Off the val_every cadence, eval is not invoked."""
    trainer = _make_trainer(val_every=3)
    trainer.eval = MagicMock()  # type: ignore[method-assign]
    trainer.ctx.step = 2  # 2 % 3 != 0
    trainer._maybe_eval()
    trainer.eval.assert_not_called()


def test_invariant_maybe_eval_runs_on_schedule():
    """On the val_every cadence, eval is invoked once."""
    trainer = _make_trainer(val_every=2)
    trainer.eval = MagicMock()  # type: ignore[method-assign]
    trainer.ctx.step = 4
    trainer._maybe_eval()
    trainer.eval.assert_called_once()


# ===========================================================================
# _maybe_save branches (lines 342-344)
# ===========================================================================


class _CkptMgr:
    def __init__(self) -> None:
        self.saved: list = []

    def save(self, *, step, state, kind, extras, parallel_ctx):
        self.saved.append((step, kind))
        return None


def test_invariant_maybe_save_noop_without_manager_or_cadence():
    """No ckpt_manager → no-op; ckpt_every <= 0 → no-op."""
    trainer = _make_trainer(ckpt_manager=None, ckpt_every=2)
    trainer.ctx.step = 2
    trainer._maybe_save({"loss": 0.1})  # no manager → returns, no crash

    mgr = _CkptMgr()
    trainer2 = _make_trainer(ckpt_manager=mgr, ckpt_every=0)
    trainer2.ctx.step = 2
    trainer2._maybe_save({"loss": 0.1})
    assert mgr.saved == []


def test_invariant_maybe_save_skips_off_schedule():
    """Off the ckpt_every cadence, nothing is saved."""
    mgr = _CkptMgr()
    trainer = _make_trainer(ckpt_manager=mgr, ckpt_every=5)
    trainer.ctx.step = 3  # 3 % 5 != 0
    trainer._maybe_save({"loss": 0.1})
    assert mgr.saved == []


def test_invariant_maybe_save_writes_on_schedule():
    """On the ckpt_every cadence, a 'step' checkpoint is written with metrics."""
    mgr = _CkptMgr()
    trainer = _make_trainer(ckpt_manager=mgr, ckpt_every=2)
    trainer.ctx.step = 2  # 2 % 2 == 0
    trainer._maybe_save({"loss": 0.1})
    assert mgr.saved == [(2, "step")]


# ===========================================================================
# state_dict — ref_namespace round-trip (lines 353-355)
# ===========================================================================


def test_invariant_state_dict_includes_ref_namespace():
    """state_dict augments the base trainer state with ref_namespace."""
    trainer = _make_trainer(ref_namespace="myref")
    sd = trainer.state_dict()
    assert sd["ref_namespace"] == "myref"
    # Base trainer keys still present (delegation, not replacement).
    assert "step" in sd
