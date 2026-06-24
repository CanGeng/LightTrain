"""Adversarial tests for ``lighttrain.builtin_plugins.diagnostics.nan_hunter.NanHunterCallback``.

Layered on top of ``tests/test_nan_hunter.py``. New coverage:

* **``critical = True`` class-attribute pin** — silently flipping this to
  False would let NaN swallow the run instead of crashing it.
* **Hook attach count on train_start** > 0; detach count = 0 after
  train_end / on_exception.
* **``_fired`` prevents multiple raises in a single step** even with
  multiple NaN modules.
* **``on_step_begin`` resets ``_fired``** so a new step can detect again.
* **NaN in inputs detected** (``check_inputs=True`` branch).
* **Inf in outputs detected** (``check_outputs=True`` branch).
* **Finite forward does NOT trigger** (false-positive guard).
* **``raise_on_hit=False`` records dump but does not raise**.
* **``_flatten_tensors`` handles dict / list / tuple / nested recursively**.
* **``_safe_name`` escapes dots and slashes**.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.diagnostics.nan_hunter import (
    NanHunterCallback,
    _flatten_tensors,
    _safe_name,
)
from tests._diagnostics import expect_count, expect_exists, expect_nonempty


class _Trainer:
    """Minimal trainer surface for the callback's on_train_start handshake."""

    def __init__(self, model: nn.Module, run_dir: Path) -> None:
        self.model = model
        self._run_dir = run_dir


class _ToyModel(nn.Module):
    """Two-layer linear stack. The hooks attach to ``a`` and ``b``."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 2)

    def forward(self, x):
        return self.b(self.a(x))


# ---------------------------------------------------------------------------
# critical = True pin
# ---------------------------------------------------------------------------

def test_pin_nan_hunter_is_critical_by_class_attribute():
    """Pin: NanHunterCallback declares ``critical = True`` so the EventBus
    re-raises immediately on its exceptions (rather than swallowing them).

    Silently flipping to False would mean a NaN-induced RuntimeError gets
    quarantined and the run continues with corrupted weights.
    """
    assert NanHunterCallback.critical is True


def test_pin_nan_hunter_registered_under_callback_nan_hunter():
    """Pin: registered as ``('callback', 'nan_hunter')``."""
    from lighttrain.registry import get
    cls = get("callback", "nan_hunter")
    assert cls is NanHunterCallback


# ---------------------------------------------------------------------------
# Hook lifecycle
# ---------------------------------------------------------------------------

def test_invariant_hooks_attached_on_train_start(tmp_path):
    """After ``on_train_start``, the callback's ``_handles`` list has one
    entry per named submodule (not counting the model itself).

    Setup: _ToyModel has 2 named submodules (a, b).
    Expected: 2 handles registered.
    """
    model = _ToyModel()
    cb = NanHunterCallback()
    cb.on_train_start(trainer=_Trainer(model, tmp_path))
    # _ToyModel has named submodules `a` and `b` (model itself is excluded)
    assert len(cb._handles) == 2


def test_invariant_hooks_detached_on_train_end(tmp_path):
    """``on_train_end`` removes every hook (``_handles`` is empty after)."""
    model = _ToyModel()
    cb = NanHunterCallback()
    cb.on_train_start(trainer=_Trainer(model, tmp_path))
    cb.on_train_end()
    assert cb._handles == []


def test_invariant_hooks_detached_on_exception(tmp_path):
    """``on_exception`` also removes hooks — important so hooks don't leak
    across a crash bundle.
    """
    model = _ToyModel()
    cb = NanHunterCallback()
    cb.on_train_start(trainer=_Trainer(model, tmp_path))
    cb.on_exception()
    assert cb._handles == []


def test_on_train_start_no_model_is_safe(tmp_path):
    """When the trainer has no ``model``, ``on_train_start`` is a no-op
    (line 58-59 of source).
    """
    class _NoModelTrainer:
        _run_dir = None
        model = None

    cb = NanHunterCallback()
    cb.on_train_start(trainer=_NoModelTrainer())
    assert cb._handles == []


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def test_finite_forward_does_not_trigger(tmp_path):
    """Pin: a forward pass with all-finite tensors does NOT raise.

    Goal: false-positive guard — production training with finite values
    must not be disturbed by the callback.
    """
    torch.manual_seed(0)
    model = _ToyModel()
    cb = NanHunterCallback(raise_on_hit=True)
    cb.on_train_start(trainer=_Trainer(model, tmp_path))
    try:
        x = torch.randn(2, 4)
        out = model(x)
        assert torch.isfinite(out).all()
    finally:
        cb.on_train_end()


def test_invariant_nan_in_input_triggers_runtime_error(tmp_path):
    """NaN in input tensor triggers the hook and raises RuntimeError when
    ``raise_on_hit=True`` (default).

    Setup: model with finite weights; input contains NaN.
    Expected: forward raises RuntimeError; message names the offending module.
    """
    model = _ToyModel()
    cb = NanHunterCallback(check_inputs=True, check_outputs=False, raise_on_hit=True)
    cb.on_train_start(trainer=_Trainer(model, tmp_path))
    cb.on_step_begin(step=1, batch={"x": torch.tensor([float("nan")])})

    bad_input = torch.tensor([[float("nan"), 0.0, 0.0, 0.0]])
    try:
        with pytest.raises(RuntimeError) as exc:
            model(bad_input)
        msg = str(exc.value)
        assert "NaN" in msg or "Inf" in msg
    finally:
        cb.on_train_end()


def test_invariant_inf_in_output_triggers_runtime_error(tmp_path):
    """Inf in a hook'd module's output triggers RuntimeError.

    Setup: forcibly poison ``model.a``'s weights so its forward outputs Inf
    given a normal input.
    """
    model = _ToyModel()
    # Make weight enormous so output overflows to inf for any non-zero input
    with torch.no_grad():
        model.a.weight.fill_(1e38)
        model.a.bias.fill_(1e38)
    cb = NanHunterCallback(check_inputs=False, check_outputs=True, raise_on_hit=True)
    cb.on_train_start(trainer=_Trainer(model, tmp_path))
    cb.on_step_begin(step=1, batch={"x": torch.tensor([1.0])})

    try:
        with pytest.raises(RuntimeError):
            model(torch.full((1, 4), 1e30))
    finally:
        cb.on_train_end()


def test_invariant_fired_flag_prevents_double_dump_in_same_step(tmp_path):
    """Once a module fires, subsequent module hits in the SAME step are
    silently skipped (line 94-95 of source: ``if self._fired: return``).

    Setup: poison both modules so both would fire.
    Expected: only ONE module dump on disk after the first raise.
    """
    model = _ToyModel()
    with torch.no_grad():
        model.a.weight.fill_(float("nan"))
        model.b.weight.fill_(float("nan"))
    cb = NanHunterCallback(check_outputs=True, raise_on_hit=False)
    cb.on_train_start(trainer=_Trainer(model, tmp_path))
    cb.on_step_begin(step=1, batch={"x": torch.tensor([1.0])})

    try:
        model(torch.ones(1, 4))
    finally:
        cb.on_train_end()

    dump_dir = tmp_path / "diagnostics" / "nan_dumps" / "step_1"
    dumps = list(dump_dir.glob("*.pt"))
    # The first hit set _fired; the second module is skipped.
    expect_count(dumps, 1, dump_dir.parent, what="step_1 module dump (*.pt)")


def test_invariant_on_step_begin_resets_fired_for_next_step(tmp_path):
    """``on_step_begin`` zeros ``_fired`` (line 80) so a new step can
    detect again.

    Setup: poison weights; run step 1 (fires); call on_step_begin(step=2);
    run forward again (must fire again).
    Expected: two dump directories — step_1 and step_2 — each with one .pt.
    """
    model = _ToyModel()
    with torch.no_grad():
        model.a.weight.fill_(float("nan"))
    cb = NanHunterCallback(check_outputs=True, raise_on_hit=False)
    cb.on_train_start(trainer=_Trainer(model, tmp_path))

    cb.on_step_begin(step=1, batch={"x": torch.tensor([1.0])})
    model(torch.ones(1, 4))
    cb.on_step_begin(step=2, batch={"x": torch.tensor([1.0])})
    model(torch.ones(1, 4))
    cb.on_train_end()

    nan_dumps = tmp_path / "diagnostics" / "nan_dumps"
    expect_exists(nan_dumps / "step_1", nan_dumps, what="step_1 dump dir")
    expect_exists(nan_dumps / "step_2", nan_dumps, what="step_2 dump dir")


def test_raise_on_hit_false_records_dump_without_raising(tmp_path):
    """With ``raise_on_hit=False`` the hook records the dump file but does
    NOT raise (line 146-149 of source).
    """
    model = _ToyModel()
    with torch.no_grad():
        model.a.weight.fill_(float("nan"))
    cb = NanHunterCallback(check_outputs=True, raise_on_hit=False)
    cb.on_train_start(trainer=_Trainer(model, tmp_path))
    cb.on_step_begin(step=5, batch={"x": torch.tensor([1.0])})

    # Should NOT raise
    model(torch.ones(1, 4))
    cb.on_train_end()
    step5 = tmp_path / "diagnostics" / "nan_dumps" / "step_5"
    dumps = list(step5.glob("*.pt"))
    expect_nonempty(dumps, step5.parent, what="a step_5 module dump (*.pt)")


# ---------------------------------------------------------------------------
# _flatten_tensors
# ---------------------------------------------------------------------------

def test_flatten_tensors_handles_bare_tensor():
    """``_flatten_tensors`` yields the tensor itself when given a bare tensor."""
    t = torch.zeros(3)
    out = list(_flatten_tensors(t))
    assert len(out) == 1
    assert out[0] is t


def test_flatten_tensors_handles_list_of_tensors():
    """``_flatten_tensors`` yields each tensor in a list in order."""
    t1, t2 = torch.zeros(2), torch.ones(2)
    out = list(_flatten_tensors([t1, t2]))
    assert out == [t1, t2]


def test_flatten_tensors_handles_tuple_of_tensors():
    """Tuples are handled the same as lists."""
    t1, t2 = torch.zeros(2), torch.ones(2)
    out = list(_flatten_tensors((t1, t2)))
    assert out == [t1, t2]


def test_flatten_tensors_handles_dict_yields_values():
    """Dict input: yields values (NOT keys), in iteration order."""
    t1, t2 = torch.zeros(2), torch.ones(2)
    out = list(_flatten_tensors({"a": t1, "b": t2}))
    assert set(id(x) for x in out) == {id(t1), id(t2)}


def test_flatten_tensors_handles_nested_recursive():
    """Nested list-of-dicts-of-tuples are flattened recursively.

    Setup: ``[{"a": t1, "b": (t2, t3)}, t4]``.
    Expected: all four tensors yielded.
    """
    t1, t2, t3, t4 = (torch.zeros(2) for _ in range(4))
    nested = [{"a": t1, "b": (t2, t3)}, t4]
    out = list(_flatten_tensors(nested))
    assert len(out) == 4
    assert set(id(x) for x in out) == {id(t1), id(t2), id(t3), id(t4)}


def test_flatten_tensors_skips_non_tensor_values():
    """Non-tensor values (int / str / None) are silently skipped."""
    t1 = torch.zeros(2)
    nested = {"a": t1, "b": 42, "c": "hello", "d": None}
    out = list(_flatten_tensors(nested))
    assert len(out) == 1
    assert out[0] is t1


# ---------------------------------------------------------------------------
# _safe_name escaping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("model.layers.0.attn", "model_layers_0_attn"),
        ("a/b/c", "a_b_c"),
        ("a.b/c.d", "a_b_c_d"),
        ("", "root"),  # empty input → "root"
        ("plain_name", "plain_name"),
    ],
)
def test_invariant_safe_name_escapes_dots_and_slashes(raw, expected):
    """``_safe_name`` replaces ``.`` and ``/`` with ``_``; empty input →
    ``"root"`` (line 167-168 of source).

    Goal: pin the filename-safety contract — these characters would create
    spurious subdirectories or hidden files on disk.
    """
    assert _safe_name(raw) == expected


# ---------------------------------------------------------------------------
# Repro-kit emission (DESIGN §18.3 / §18.7)
#
# These exercise the full callback path against a real ``TinyCausalLM`` (the
# _ToyModel cases above only cover the hook/flatten/escaping units). When the
# callback fires it must drop a self-contained repro kit alongside the module
# dump.
# ---------------------------------------------------------------------------


class _LMTrainer:
    def __init__(self, model, run_dir: Path) -> None:
        self.model = model
        self._run_dir = run_dir


def _tiny_lm():
    from lighttrain.builtin_plugins.models.adapters.tiny_lm import TinyCausalLM

    return TinyCausalLM(
        vocab_size=16, d_model=8, n_layers=1, n_heads=2, max_seq_len=8
    )


def test_invariant_repro_kit_written_on_nan_fire(tmp_path):
    """When a NaN forward fires, the callback writes one ``repro_nan_*`` kit
    containing ``repro.py``, ``batch.pt`` and ``model_state.safetensors``, plus
    at least one module dump under ``nan_dumps``.
    """
    from lighttrain.engine._context import StepContext

    model = _tiny_lm()
    with torch.no_grad():
        model.tok_emb.weight[0].fill_(float("nan"))
    cb = NanHunterCallback()
    ctx = StepContext(run_dir=tmp_path)
    cb.on_train_start(trainer=_LMTrainer(model, tmp_path), ctx=ctx)
    cb.on_step_begin(
        step=1,
        batch={
            "input_ids": torch.zeros(1, 4, dtype=torch.long),  # row 0 ⇒ NaN
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        },
    )
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        model(input_ids=torch.zeros(1, 4, dtype=torch.long))
    cb.on_train_end()

    diag = tmp_path / "diagnostics"
    repros = sorted(diag.glob("repro_nan_*"))
    expect_count(repros, 1, diag, what="repro_nan_* kit")
    expect_exists(repros[0] / "repro.py", repros[0], what="repro.py")
    expect_exists(repros[0] / "batch.pt", repros[0], what="batch.pt")
    expect_exists(repros[0] / "model_state.safetensors", repros[0], what="model_state.safetensors")
    pt_dumps = sorted((diag / "nan_dumps").rglob("*.pt"))
    expect_nonempty(pt_dumps, diag, what="a module dump (*.pt) under nan_dumps")


def test_invariant_repro_py_is_under_80_lines(tmp_path):
    """The emitted ``repro.py`` stays ≤80 lines (DESIGN §18.3)."""
    from lighttrain.engine._context import StepContext

    model = _tiny_lm()
    with torch.no_grad():
        model.tok_emb.weight[0].fill_(float("inf"))
    cb = NanHunterCallback()
    ctx = StepContext(run_dir=tmp_path)
    cb.on_train_start(trainer=_LMTrainer(model, tmp_path), ctx=ctx)
    cb.on_step_begin(
        step=1,
        batch={
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        },
    )
    with pytest.raises(RuntimeError):
        model(input_ids=torch.zeros(1, 4, dtype=torch.long))
    cb.on_train_end()

    repros = sorted((tmp_path / "diagnostics").glob("repro_nan_*"))
    expect_nonempty(repros, tmp_path / "diagnostics", what="a repro_nan_* kit")
    lines = (repros[0] / "repro.py").read_text(encoding="utf-8").splitlines()
    assert len(lines) <= 80, f"repro.py is {len(lines)} lines, DESIGN §18.3 says ≤80"
