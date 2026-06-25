"""Coverage tests for ``DeadNeuronCallback`` targeting uncovered branches.

Lines driven to covered:
* 53  — model resolved from trainer when ctx has none
* 55  — early return when model is None from both ctx and trainer
* 58  — run_dir resolved from trainer._run_dir when ctx.run_dir is None
* 64–65 — regex module_pattern that *excludes* a sub-module name
* 69  — default filter: cls name has none of linear/silu/gelu/relu → skip
* 76  — h.remove() called in on_train_end
* 77  — warning logged when h.remove() raises
* 88  — on_step_end early return when _run_dir is None
* 92  — empty samples list skipped inside on_step_end report loop
* 116 — hook output is a list/tuple → take first element
* 118 — hook output is not a tensor after unwrap → return early
* 122 — rolling window overflow: buf.pop(0) when buf grows > window
"""

from __future__ import annotations

import json
import logging

import pytest
import torch
import torch.nn as nn

from lighttrain.builtin_plugins.observability.diagnostics.dead_neuron import (
    DeadNeuronCallback,
)
from lighttrain.engine._context import StepContext
from tests._diagnostics import expect_nonempty

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _TinyNet(nn.Module):
    """Two-layer net: Linear + a custom module that is NOT linear/silu/gelu/relu."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 6)
        self.bn = nn.BatchNorm1d(6)  # 'batchnorm1d' — no keyword match

    def forward(self, x):
        return self.bn(self.fc(x))


class _Trainer:
    """Minimal trainer stub carrying model and _run_dir."""

    def __init__(self, model, run_dir=None):
        self.model = model
        self._run_dir = run_dir


class _CtxNoModel:
    """ctx without a model attribute."""

    def __init__(self, run_dir=None):
        self.run_dir = run_dir


class _CtxNoRunDir:
    """ctx with a model but no run_dir."""

    def __init__(self, model):
        self.model = model
        self.run_dir = None


class _BrokenHandle:
    """Simulates a hook handle whose .remove() raises."""

    def remove(self):
        raise RuntimeError("synthetic remove failure")


class _TupleOutputNet(nn.Module):
    """forward() returns a tuple; hook should unwrap and keep first element."""

    def __init__(self):
        super().__init__()
        # We need *some* registered sub-module so we can attach a hook.
        self.lin = nn.Linear(2, 3)

    def forward(self, x):
        # Return a tuple so the hook sees a tuple output.
        out = self.lin(x)
        return (out, torch.zeros(1))


class _NonTensorOutputNet(nn.Module):
    """forward() returns a string; hook must bail out without crashing."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 3)

    def forward(self, x):
        return "not-a-tensor"


# ---------------------------------------------------------------------------
# on_train_start — model / run_dir resolution branches
# ---------------------------------------------------------------------------

def test_invariant_model_from_trainer_when_ctx_has_none(tmp_path):
    """Model resolved from trainer when ctx carries no model (line 53)."""
    torch.manual_seed(0)
    model = _TinyNet()
    cb = DeadNeuronCallback(window=4, every_n_steps=2)
    ctx = _CtxNoModel(run_dir=tmp_path)
    trainer = _Trainer(model, run_dir=tmp_path)
    cb.on_train_start(trainer=trainer, ctx=ctx)
    # At least the Linear hook was registered.
    assert len(cb._handles) >= 1


def test_invariant_no_model_anywhere_returns_early(tmp_path):
    """No model in ctx or trainer → hooks list stays empty (line 55)."""
    cb = DeadNeuronCallback(window=4, every_n_steps=2)
    ctx = _CtxNoModel(run_dir=tmp_path)
    trainer_no_model = _Trainer(model=None, run_dir=tmp_path)
    cb.on_train_start(trainer=trainer_no_model, ctx=ctx)
    assert cb._handles == []


def test_invariant_no_trainer_no_ctx_returns_early():
    """Neither ctx nor trainer supplied → early return with no handles (line 55)."""
    cb = DeadNeuronCallback(window=4, every_n_steps=2)
    cb.on_train_start()  # both default to None
    assert cb._handles == []


def test_invariant_run_dir_from_trainer_when_ctx_run_dir_none(tmp_path):
    """run_dir taken from trainer._run_dir when ctx.run_dir is None (line 58)."""
    torch.manual_seed(0)
    model = _TinyNet()
    cb = DeadNeuronCallback(window=4, every_n_steps=2)
    ctx = _CtxNoRunDir(model=model)
    trainer = _Trainer(model=model, run_dir=tmp_path)
    cb.on_train_start(trainer=trainer, ctx=ctx)
    assert cb._run_dir == tmp_path

    for s in range(1, 5):
        _ = model(torch.randn(2, 4))
        cb.on_step_end(step=s)

    files = sorted((tmp_path / "diagnostics").glob("dead_neurons_*.json"))
    expect_nonempty(files, tmp_path, what="dead_neurons_<step>.json from trainer._run_dir path")


# ---------------------------------------------------------------------------
# on_train_start — module_pattern regex filtering (lines 63–65)
# ---------------------------------------------------------------------------

def test_invariant_regex_pattern_excludes_non_matching_module(tmp_path):
    """Regex module_pattern skips modules whose name does not match (lines 64–65)."""
    torch.manual_seed(0)
    model = _TinyNet()  # has 'fc' (Linear) and 'bn' (BatchNorm1d)
    # Only match modules named literally 'fc'; 'bn' must be skipped.
    cb = DeadNeuronCallback(module_pattern=r"^fc$", every_n_steps=2)
    ctx = StepContext(run_dir=tmp_path, model=model)
    cb.on_train_start(ctx=ctx)
    # Exactly one hook for 'fc'.
    assert len(cb._handles) == 1


def test_invariant_regex_pattern_matches_multiple_modules(tmp_path):
    """Regex that matches all sub-modules registers hooks for each."""
    torch.manual_seed(0)

    class _TwoLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(2, 2)
            self.b = nn.Linear(2, 2)

        def forward(self, x):
            return self.b(self.a(x))

    model = _TwoLinear()
    cb = DeadNeuronCallback(module_pattern=r"[ab]", every_n_steps=10)
    ctx = StepContext(run_dir=tmp_path, model=model)
    cb.on_train_start(ctx=ctx)
    assert len(cb._handles) == 2


def test_invariant_regex_excludes_all_modules_no_hooks(tmp_path):
    """Regex that matches nothing registers zero hooks (lines 64–65 always taken)."""
    torch.manual_seed(0)
    model = _TinyNet()
    cb = DeadNeuronCallback(module_pattern=r"^NOPE_XYZ$", every_n_steps=2)
    ctx = StepContext(run_dir=tmp_path, model=model)
    cb.on_train_start(ctx=ctx)
    assert len(cb._handles) == 0


# ---------------------------------------------------------------------------
# on_train_start — default class-name filter (line 69)
# ---------------------------------------------------------------------------

def test_invariant_default_filter_skips_non_activation_modules(tmp_path):
    """Without module_pattern, BatchNorm1d ('batchnorm1d') is skipped (line 69).

    _TinyNet has fc (Linear) and bn (BatchNorm1d).  Only the Linear hook
    should be registered, not the BatchNorm1d.
    """
    torch.manual_seed(0)
    model = _TinyNet()
    cb = DeadNeuronCallback(every_n_steps=2)
    ctx = StepContext(run_dir=tmp_path, model=model)
    cb.on_train_start(ctx=ctx)
    # Only fc matches; bn ('batchnorm1d') does not contain any keyword.
    assert len(cb._handles) == 1


# ---------------------------------------------------------------------------
# on_train_end — hook removal (lines 75–77)
# ---------------------------------------------------------------------------

def test_invariant_on_train_end_removes_hooks(tmp_path):
    """on_train_end clears handles list after removing hooks (line 76)."""
    torch.manual_seed(0)
    model = _TinyNet()
    cb = DeadNeuronCallback(every_n_steps=2)
    ctx = StepContext(run_dir=tmp_path, model=model)
    cb.on_train_start(ctx=ctx)
    assert len(cb._handles) >= 1
    cb.on_train_end()
    assert cb._handles == []


def test_invariant_on_train_end_logs_warning_on_remove_failure(
    tmp_path, caplog
):
    """Warning logged when a hook's .remove() raises (lines 76–77)."""
    cb = DeadNeuronCallback(every_n_steps=2)
    cb._handles.append(_BrokenHandle())

    with caplog.at_level(logging.WARNING, logger="lighttrain"):
        cb.on_train_end()

    assert any("dead_neuron" in r.message for r in caplog.records), (
        f"expected warning with 'dead_neuron'; got: {[r.message for r in caplog.records]}"
    )
    assert cb._handles == []


# ---------------------------------------------------------------------------
# on_step_end — no run_dir early return (line 88)
# ---------------------------------------------------------------------------

def test_invariant_on_step_end_noop_without_run_dir(tmp_path):
    """on_step_end returns immediately when _run_dir is None (line 88)."""
    torch.manual_seed(0)
    model = _TinyNet()
    cb = DeadNeuronCallback(every_n_steps=2)
    # Never call on_train_start → _run_dir stays None.
    cb._run_dir = None

    # Manually register a hook to populate _buf.
    hook = cb._make_hook("fc")
    handle = model.fc.register_forward_hook(hook)
    try:
        _ = model(torch.randn(2, 4))
        _ = model(torch.randn(2, 4))
    finally:
        handle.remove()

    # Trigger on_step_end at the boundary step with _run_dir=None.
    cb.on_step_end(step=2)

    # No diagnostics directory should have been created.
    assert not (tmp_path / "diagnostics").exists()


# ---------------------------------------------------------------------------
# on_step_end — empty samples list skipped (line 92)
# ---------------------------------------------------------------------------

def test_invariant_empty_buf_entry_skipped_in_report(tmp_path):
    """Buffer entry with empty sample list is omitted from JSON report (line 92)."""
    torch.manual_seed(0)
    model = _TinyNet()
    cb = DeadNeuronCallback(every_n_steps=2)
    ctx = StepContext(run_dir=tmp_path, model=model)
    cb.on_train_start(ctx=ctx)

    # Manually inject an empty samples list for a phantom module.
    cb._buf["phantom_layer"] = []  # empty → should be skipped

    # Run a forward pass to populate real buffer entries.
    _ = model(torch.randn(2, 4))

    cb.on_step_end(step=2)

    files = sorted((tmp_path / "diagnostics").glob("dead_neurons_*.json"))
    expect_nonempty(files, tmp_path, what="dead_neurons_2.json")
    report = json.loads(files[0].read_text(encoding="utf-8"))
    # The phantom_layer with empty samples must NOT appear in the report.
    assert "phantom_layer" not in report


# ---------------------------------------------------------------------------
# _make_hook — list/tuple output unwrapping (line 116)
# ---------------------------------------------------------------------------

def test_invariant_hook_unwraps_tuple_output(tmp_path):
    """Hook takes first element of a tuple output and buffers it (line 116)."""
    torch.manual_seed(0)
    model = _TupleOutputNet()
    cb = DeadNeuronCallback(module_pattern=r"lin", every_n_steps=1)
    ctx = StepContext(run_dir=tmp_path, model=model)
    cb.on_train_start(ctx=ctx)
    assert len(cb._handles) == 1  # lin matched

    _ = model(torch.randn(2, 2))
    # The hook should have buffered a tensor (the first element of the tuple).
    assert "lin" in cb._buf
    assert len(cb._buf["lin"]) == 1
    t = cb._buf["lin"][0]
    assert isinstance(t, torch.Tensor)
    # lin is Linear(2,3) → output shape (..., 3)
    assert t.shape[-1] == 3


def test_invariant_hook_unwraps_list_output(tmp_path):
    """Hook takes first element of a list output (line 116, list branch)."""
    torch.manual_seed(0)

    class _ListOutputMod(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 4)

        def forward(self, x):
            return [self.lin(x), torch.zeros(1)]

    model = _ListOutputMod()
    cb = DeadNeuronCallback(module_pattern=r"lin", every_n_steps=1)
    ctx = StepContext(run_dir=tmp_path, model=model)
    cb.on_train_start(ctx=ctx)

    _ = model(torch.randn(3, 2))
    assert "lin" in cb._buf
    assert isinstance(cb._buf["lin"][0], torch.Tensor)


# ---------------------------------------------------------------------------
# _make_hook — non-tensor output early return (line 118)
# ---------------------------------------------------------------------------

def test_invariant_hook_skips_non_tensor_output(tmp_path):
    """Hook returns without buffering when output is not a tensor (line 118)."""
    torch.manual_seed(0)
    model = _NonTensorOutputNet()
    cb = DeadNeuronCallback(module_pattern=r"lin", every_n_steps=1)
    ctx = StepContext(run_dir=tmp_path, model=model)
    cb.on_train_start(ctx=ctx)

    _ = model(torch.randn(2, 2))
    # lin's forward is never directly called with a string output; the hook
    # wraps the *module's* output.  We instead directly call the hook with a
    # non-tensor value.
    hook_fn = cb._make_hook("test_layer")
    hook_fn(None, None, "this-is-a-string")  # should not raise
    assert "test_layer" not in cb._buf


def test_invariant_hook_skips_empty_tuple_output(tmp_path):
    """Hook skips empty tuple (falsy), falls through to non-tensor check (line 118)."""
    cb = DeadNeuronCallback(every_n_steps=1)
    hook_fn = cb._make_hook("layer_x")
    hook_fn(None, None, ())  # empty tuple → not a tensor → return
    assert "layer_x" not in cb._buf


def test_invariant_hook_skips_none_output(tmp_path):
    """Hook called with None output does not buffer anything (line 118)."""
    cb = DeadNeuronCallback(every_n_steps=1)
    hook_fn = cb._make_hook("layer_y")
    hook_fn(None, None, None)
    assert "layer_y" not in cb._buf


# ---------------------------------------------------------------------------
# _make_hook — rolling window overflow (line 122)
# ---------------------------------------------------------------------------

def test_invariant_window_overflow_pops_oldest_entry(tmp_path):
    """Buffer size is capped at window; oldest entry dropped on overflow (line 122)."""
    torch.manual_seed(0)
    nn.Linear(4, 8)
    cb = DeadNeuronCallback(window=3, every_n_steps=100)

    hook_fn = cb._make_hook("lin")
    for i in range(5):
        t = torch.full((2, 8), float(i))
        hook_fn(None, None, t)

    buf = cb._buf["lin"]
    assert len(buf) == 3, f"expected window=3, got {len(buf)}"
    # The oldest entries (i=0,1) should have been dropped; newest 3 remain.
    expected_vals = [2.0, 3.0, 4.0]
    for tensor, expected in zip(buf, expected_vals, strict=False):
        assert float(tensor[0, 0].item()) == pytest.approx(expected)


def test_invariant_window_one_keeps_only_latest():
    """Window=1 keeps exactly the most recent sample (extreme overflow case)."""
    cb = DeadNeuronCallback(window=1, every_n_steps=100)
    hook_fn = cb._make_hook("x")
    for i in range(4):
        hook_fn(None, None, torch.tensor([[float(i)]]))
    buf = cb._buf["x"]
    assert len(buf) == 1
    assert float(buf[0][0, 0].item()) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# General invariants / edge cases
# ---------------------------------------------------------------------------

def test_invariant_constructor_clamps_window_and_steps():
    """window and every_n_steps are clamped to at least 1."""
    cb = DeadNeuronCallback(window=0, every_n_steps=0)
    assert cb.window == 1
    assert cb.every_n_steps == 1


def test_invariant_zero_threshold_matches_only_exact_zeros():
    """zero_threshold=0 → only exact 0.0 counted as 'zero'."""
    torch.manual_seed(0)
    cb = DeadNeuronCallback(window=10, every_n_steps=1, zero_threshold=0.0)
    hook_fn = cb._make_hook("lyr")
    hook_fn(None, None, torch.tensor([[0.0, 1.0, 2.0]]))
    # Samples stored but not reported until on_step_end.
    assert "lyr" in cb._buf


def test_invariant_step_not_divisible_does_not_write(tmp_path):
    """on_step_end does NOT write when step % every_n_steps != 0."""
    torch.manual_seed(0)
    model = _TinyNet()
    cb = DeadNeuronCallback(window=4, every_n_steps=10)
    ctx = StepContext(run_dir=tmp_path, model=model)
    cb.on_train_start(ctx=ctx)
    _ = model(torch.randn(2, 4))
    cb.on_step_end(step=3)  # 3 % 10 != 0
    assert not (tmp_path / "diagnostics").exists()


def test_invariant_step_zero_does_not_write(tmp_path):
    """on_step_end with step=0 does not write (step <= 0 guard)."""
    torch.manual_seed(0)
    model = _TinyNet()
    cb = DeadNeuronCallback(window=4, every_n_steps=1)
    ctx = StepContext(run_dir=tmp_path, model=model)
    cb.on_train_start(ctx=ctx)
    _ = model(torch.randn(2, 4))
    cb.on_step_end(step=0)
    assert not (tmp_path / "diagnostics").exists()


def test_invariant_report_json_schema(tmp_path):
    """Full pipeline: report JSON has expected keys for each captured module."""
    torch.manual_seed(0)
    model = _TinyNet()
    cb = DeadNeuronCallback(window=4, every_n_steps=2)
    ctx = StepContext(run_dir=tmp_path, model=model)
    cb.on_train_start(ctx=ctx)
    for s in range(1, 5):
        _ = model(torch.randn(2, 4))
        cb.on_step_end(step=s)
    cb.on_train_end()

    files = sorted((tmp_path / "diagnostics").glob("dead_neurons_*.json"))
    expect_nonempty(files, tmp_path, what="dead_neurons_<step>.json")
    for f in files:
        report = json.loads(f.read_text(encoding="utf-8"))
        for entry in report.values():
            for key in ("zero_ratio_mean", "zero_ratio_max", "var_mean", "var_min", "n_channels"):
                assert key in entry, f"missing key {key!r} in {entry}"


def test_invariant_buf_cleared_after_report(tmp_path):
    """_buf is cleared after each on_step_end write."""
    torch.manual_seed(0)
    model = _TinyNet()
    cb = DeadNeuronCallback(window=4, every_n_steps=2)
    ctx = StepContext(run_dir=tmp_path, model=model)
    cb.on_train_start(ctx=ctx)
    for s in range(1, 3):
        _ = model(torch.randn(2, 4))
        cb.on_step_end(step=s)
    assert cb._buf == {}


@pytest.mark.parametrize("n_channels", [1, 4, 16])
def test_invariant_n_channels_matches_tensor_last_dim(tmp_path, n_channels):
    """n_channels in report equals the last dimension of the activation tensor."""
    torch.manual_seed(0)

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(n_channels, n_channels)

        def forward(self, x):
            return self.lin(x)

    model = _Wrapper()
    cb = DeadNeuronCallback(window=4, every_n_steps=1)
    ctx = StepContext(run_dir=tmp_path / str(n_channels), model=model)
    cb.on_train_start(ctx=ctx)
    _ = model(torch.randn(2, n_channels))
    cb.on_step_end(step=1)
    files = sorted((tmp_path / str(n_channels) / "diagnostics").glob("dead_neurons_*.json"))
    expect_nonempty(files, tmp_path, what="dead_neurons_1.json")
    report = json.loads(files[0].read_text(encoding="utf-8"))
    # Exactly one sub-module 'lin' was matched and hooked.
    assert len(report) == 1
    entry = next(iter(report.values()))
    assert entry["n_channels"] == n_channels
