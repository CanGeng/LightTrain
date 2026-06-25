"""Coverage tests for ``lighttrain.observability.diagnostics.crash_bundle``.

Pins every reachable uncovered branch in the original module:

* Line 81-82  – ``capture_env()`` raises → warning logged, env.json still written
* Line 100-111 – tokenizer decode loop: success path and ``decode`` failure path
* Line 120-121 – ``_save_model`` raises → warning logged, bundle continues
* Line 135-136 – ``_infer_spec`` raises → warning logged, bundle continues
* Line 154-155 – ``rng_state()`` raises → warning logged, rng.pt absent
* Line 165-166 – metrics JSONL write raises → warning logged, file absent
* Line 173     – ``recent_logs`` non-empty → logs_tail.txt written
* Line 179-187 – ``_scalar`` with a 0-d tensor (float path) and multi-element
                  tensor that fails ``.item()`` (None path)
* Line 188-190 – ``_scalar`` with primitive types and fallback ``str(v)``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from lighttrain.observability.diagnostics.crash_bundle import (
    _scalar,
    write_crash_bundle,
)
from tests._diagnostics import expect_exists

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal two-parameter model usable with safetensors."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.lin(x)


class _GoodTokenizer:
    """Stub tokenizer whose `decode` succeeds."""

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)


class _BadTokenizer:
    """Stub tokenizer whose `decode` always raises."""

    def decode(self, ids: list[int]) -> str:
        raise ValueError("decode intentionally broken")


def _exc(msg: str = "synthetic") -> RuntimeError:
    return RuntimeError(msg)


def _minimal_bundle(tmp_path: Path, **kw) -> Path:
    """Call write_crash_bundle with only the required args, allowing overrides."""
    defaults = dict(exception=_exc(), step=1)
    defaults.update(kw)
    return write_crash_bundle(tmp_path, **defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Line 81-82: env capture failure branch
# ---------------------------------------------------------------------------


def test_invariant_env_json_written_even_when_capture_env_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Even if ``capture_env`` raises, env.json is still written with base fields.

    The import is local (``from lighttrain.utils.env_capture import capture_env``
    inside the try-block), so we patch at the source module level.
    """
    with patch(
        "lighttrain.utils.env_capture.capture_env",
        side_effect=OSError("probe failed"),
    ):
        with caplog.at_level(logging.WARNING, logger="lighttrain.observability.diagnostics.crash_bundle"):
            bundle = _minimal_bundle(tmp_path)

    env_path = bundle / "env.json"
    expect_exists(env_path, bundle, what="env.json")
    env = json.loads(env_path.read_text(encoding="utf-8"))
    # Base keys must always be present
    assert "exception_type" in env
    assert "step" in env
    assert env["exception_type"] == "RuntimeError"
    # A warning must have been emitted
    assert any("env capture failed" in r.message for r in caplog.records), caplog.text


# ---------------------------------------------------------------------------
# Lines 100-111: tokenizer decode loop (success + failure per sample)
# ---------------------------------------------------------------------------


def test_invariant_decoded_txt_written_with_good_tokenizer(tmp_path: Path) -> None:
    """When a tokenizer with ``decode`` is supplied, decoded.txt contains each sample."""
    batch = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])}
    tokenizer = _GoodTokenizer()
    bundle = _minimal_bundle(tmp_path, batch=batch, tokenizer=tokenizer)

    decoded_path = bundle / "decoded.txt"
    expect_exists(decoded_path, bundle, what="decoded.txt")
    content = decoded_path.read_text(encoding="utf-8")
    assert "# sample[0]" in content
    assert "# sample[1]" in content


def test_invariant_decoded_txt_records_placeholder_on_decode_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """When ``tokenizer.decode`` raises, decoded.txt records a placeholder."""
    batch = {"input_ids": torch.tensor([[10, 20, 30]])}
    tokenizer = _BadTokenizer()
    with caplog.at_level(logging.WARNING, logger="lighttrain.observability.diagnostics.crash_bundle"):
        bundle = _minimal_bundle(tmp_path, batch=batch, tokenizer=tokenizer)

    decoded_path = bundle / "decoded.txt"
    expect_exists(decoded_path, bundle, what="decoded.txt")
    content = decoded_path.read_text(encoding="utf-8")
    assert "<decode error>" in content
    assert any("tokenizer.decode failed" in r.message for r in caplog.records), caplog.text


def test_pin_current_behavior_no_decoded_txt_without_tokenizer(tmp_path: Path) -> None:
    """Pin: when tokenizer=None (default), decoded.txt is still written but empty.

    The current code writes decoded.txt whenever batch is a dict (line 112-114),
    even if no tokenizer is supplied, resulting in an empty file.
    """
    batch = {"input_ids": torch.tensor([[1, 2, 3]])}
    bundle = _minimal_bundle(tmp_path, batch=batch)

    # decoded.txt should be present (written by the batch branch)
    decoded_path = bundle / "decoded.txt"
    expect_exists(decoded_path, bundle, what="decoded.txt")
    assert decoded_path.read_text(encoding="utf-8") == ""


def test_invariant_no_decode_loop_when_input_ids_missing(tmp_path: Path) -> None:
    """No decode attempted when batch has no 'input_ids' key."""
    batch = {"attention_mask": torch.ones(2, 4)}
    tokenizer = _GoodTokenizer()
    bundle = _minimal_bundle(tmp_path, batch=batch, tokenizer=tokenizer)
    decoded_path = bundle / "decoded.txt"
    expect_exists(decoded_path, bundle, what="decoded.txt")
    # No sample headers should appear
    assert "# sample[" not in decoded_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Lines 120-121: _save_model failure branch
# ---------------------------------------------------------------------------


def test_invariant_bundle_continues_when_save_model_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """If safetensors ``save_model`` raises, a warning is logged but the bundle
    is still returned and traceback.txt still exists."""
    model = _TinyModel()
    with patch(
        "lighttrain.observability.diagnostics.crash_bundle._save_model",
        side_effect=RuntimeError("disk full"),
    ):
        with caplog.at_level(logging.WARNING, logger="lighttrain.observability.diagnostics.crash_bundle"):
            bundle = _minimal_bundle(tmp_path, model=model)

    expect_exists(bundle / "traceback.txt", bundle, what="traceback.txt")
    assert not (bundle / "model_state.safetensors").exists()
    assert any("model state save failed" in r.message for r in caplog.records), caplog.text


# ---------------------------------------------------------------------------
# Lines 135-136: _infer_spec failure branch
# ---------------------------------------------------------------------------


def test_invariant_bundle_continues_when_infer_spec_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """If ``_infer_spec`` raises, a warning is logged and model_spec.json is absent."""
    model = _TinyModel()
    with patch(
        "lighttrain.observability.diagnostics.crash_bundle._save_model"
    ), patch(
        "lighttrain.observability.diagnostics.nan_repro._infer_spec",
        side_effect=AttributeError("no spec"),
    ):
        with caplog.at_level(logging.WARNING, logger="lighttrain.observability.diagnostics.crash_bundle"):
            bundle = _minimal_bundle(tmp_path, model=model)

    assert not (bundle / "model_spec.json").exists()
    assert any("model spec inference failed" in r.message for r in caplog.records), caplog.text


# ---------------------------------------------------------------------------
# Lines 154-155: rng_state failure branch
# ---------------------------------------------------------------------------


def test_invariant_bundle_continues_when_rng_state_raises(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """If ``rng_state`` raises, a warning is logged and rng.pt is absent."""
    with patch(
        "lighttrain.observability.diagnostics.crash_bundle.rng_state",
        side_effect=RuntimeError("rng exploded"),
    ):
        with caplog.at_level(logging.WARNING, logger="lighttrain.observability.diagnostics.crash_bundle"):
            bundle = _minimal_bundle(tmp_path)

    assert not (bundle / "rng.pt").exists()
    assert any("RNG state capture failed" in r.message for r in caplog.records), caplog.text


# ---------------------------------------------------------------------------
# Lines 165-166: metrics_recent.jsonl write failure branch
# ---------------------------------------------------------------------------


def test_pin_current_behavior_metrics_write_raises_warning_and_bundle_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Pin: if the metrics serialization raises inside the try/except block
    (lines 162-168), a warning IS logged and the bundle is still returned.

    Note (suspected bug): Python's ``with open(…) as f`` creates (truncates) the
    file before ``f.write(…)`` executes, so metrics_recent.jsonl may exist as
    an empty file even when the write fails.  The warning text "bundle omits
    metrics_recent.jsonl" is therefore misleading — pinning current behavior.
    """
    metrics = {"loss": 0.5}
    with patch(
        "lighttrain.observability.diagnostics.crash_bundle._scalar",
        side_effect=RuntimeError("scalar boom"),
    ):
        with caplog.at_level(logging.WARNING, logger="lighttrain.observability.diagnostics.crash_bundle"):
            bundle = _minimal_bundle(tmp_path, metrics=metrics)

    # The bundle directory was still returned
    assert bundle.is_dir()
    # Warning was emitted (the key correctness invariant)
    assert any("recent metrics write failed" in r.message for r in caplog.records), caplog.text
    # Pin current behavior: file may exist as an empty stub due to open() creating it
    jsonl_path = bundle / "metrics_recent.jsonl"
    if jsonl_path.exists():
        # If the file exists it must be empty or contain no valid JSON row
        content = jsonl_path.read_text(encoding="utf-8").strip()
        assert content == "", f"expected empty file on write failure, got: {content!r}"


# ---------------------------------------------------------------------------
# Line 173: recent_logs path
# ---------------------------------------------------------------------------


def test_invariant_logs_tail_written_when_recent_logs_provided(tmp_path: Path) -> None:
    """When ``recent_logs`` is non-empty, logs_tail.txt is written."""
    bundle = _minimal_bundle(tmp_path, recent_logs="step 1: loss=0.9\nstep 2: loss=0.8")
    logs_path = bundle / "logs_tail.txt"
    expect_exists(logs_path, bundle, what="logs_tail.txt")
    assert "step 1: loss=0.9" in logs_path.read_text(encoding="utf-8")


def test_invariant_no_logs_tail_when_recent_logs_empty(tmp_path: Path) -> None:
    """When ``recent_logs`` is the default empty string, logs_tail.txt is absent."""
    bundle = _minimal_bundle(tmp_path)
    assert not (bundle / "logs_tail.txt").exists()


# ---------------------------------------------------------------------------
# Lines 179-190: _scalar helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (torch.tensor(3.14), pytest.approx(3.14)),
        (torch.tensor(0.0), pytest.approx(0.0)),
        (torch.tensor(-1.5), pytest.approx(-1.5)),
    ],
)
def test_invariant_scalar_converts_scalar_tensor_to_float(
    value: torch.Tensor, expected: object
) -> None:
    """``_scalar`` converts a 0-d or single-element tensor to float."""
    result = _scalar(value)
    assert isinstance(result, float)
    assert result == expected


def test_invariant_scalar_returns_none_for_multi_element_tensor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``_scalar`` returns None and logs a warning when ``.item()`` fails (multi-element tensor).

    This pins the branch at line 183-187.
    """
    multi = torch.tensor([1.0, 2.0, 3.0])  # .item() raises on >1-element tensors
    with caplog.at_level(logging.WARNING, logger="lighttrain.observability.diagnostics.crash_bundle"):
        result = _scalar(multi)
    assert result is None
    assert any("tensor metric could not be coerced to float" in r.message for r in caplog.records), caplog.text


@pytest.mark.parametrize(
    "value",
    [42, 3.14, "hello", True, False, None],
)
def test_invariant_scalar_passes_through_primitives(value: object) -> None:
    """``_scalar`` returns int/float/str/bool/None unchanged."""
    assert _scalar(value) is value or _scalar(value) == value


def test_invariant_scalar_converts_unknown_type_to_str() -> None:
    """``_scalar`` converts unrecognised objects to ``str(v)`` (line 190)."""

    class _Weird:
        def __str__(self) -> str:
            return "weird_repr"

    result = _scalar(_Weird())
    assert result == "weird_repr"


def test_invariant_scalar_converts_list_to_str() -> None:
    """Lists are not primitive; ``_scalar`` must fall back to str()."""
    result = _scalar([1, 2, 3])
    assert result == "[1, 2, 3]"


# ---------------------------------------------------------------------------
# Integration: full bundle with all optional fields populated
# ---------------------------------------------------------------------------


def test_invariant_full_bundle_with_recent_logs_and_metrics(tmp_path: Path) -> None:
    """Smoke test: full bundle with model, batch, optimizer, metrics, and logs."""
    torch.manual_seed(0)
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    exc = ValueError("oops")
    bundle = write_crash_bundle(
        tmp_path,
        exception=exc,
        step=77,
        model=model,
        batch=batch,
        optimizer=optimizer,
        metrics={"loss": torch.tensor(1.5), "ppl": 4.4, "extra": [1, 2]},
        recent_logs="INFO step 77\n",
        tokenizer=_GoodTokenizer(),
    )
    assert bundle.is_dir()
    expect_exists(bundle / "traceback.txt", bundle, what="traceback.txt")
    expect_exists(bundle / "env.json", bundle, what="env.json")
    expect_exists(bundle / "batch.pt", bundle, what="batch.pt")
    expect_exists(bundle / "rng.pt", bundle, what="rng.pt")
    expect_exists(bundle / "logs_tail.txt", bundle, what="logs_tail.txt")

    # metrics_recent.jsonl must parse cleanly
    jsonl = (bundle / "metrics_recent.jsonl").read_text(encoding="utf-8").strip()
    row = json.loads(jsonl)
    assert row["step"] == 77
    assert isinstance(row["loss"], float)
    assert isinstance(row["ppl"], float)
    # The list "extra" falls back to str()
    assert row["extra"] == "[1, 2]"


def test_invariant_bundle_with_string_run_dir(tmp_path: Path) -> None:
    """``run_dir`` can be passed as a plain string (Path conversion on line 60)."""
    bundle = write_crash_bundle(
        str(tmp_path),
        exception=_exc("str path"),
        step=0,
    )
    assert bundle.is_dir()


def test_invariant_traceback_text_captures_exception(tmp_path: Path) -> None:
    """traceback.txt contains the exception class name and message."""
    try:
        raise KeyError("missing_key")
    except KeyError as e:
        exc = e

    bundle = write_crash_bundle(tmp_path, exception=exc, step=5)
    tb = (bundle / "traceback.txt").read_text(encoding="utf-8")
    assert "KeyError" in tb


def test_invariant_env_json_step_matches_arg(tmp_path: Path) -> None:
    """env.json 'step' field matches the ``step`` argument passed in."""
    bundle = write_crash_bundle(tmp_path, exception=_exc(), step=123)
    env = json.loads((bundle / "env.json").read_text(encoding="utf-8"))
    assert env["step"] == 123


def test_invariant_batch_pt_is_cpu_tensors(tmp_path: Path) -> None:
    """Saved batch.pt tensors are on CPU (detach+cpu in line 91-93)."""
    batch = {
        "input_ids": torch.randint(0, 10, (2, 8)),
        "label": "text",  # non-tensor value should be preserved as-is
    }
    bundle = _minimal_bundle(tmp_path, batch=batch)
    loaded = torch.load(str(bundle / "batch.pt"), weights_only=False)
    assert loaded["input_ids"].device == torch.device("cpu")
    assert loaded["label"] == "text"


def test_invariant_no_batch_pt_when_batch_is_none(tmp_path: Path) -> None:
    """When ``batch=None``, batch.pt is not written."""
    bundle = _minimal_bundle(tmp_path)
    assert not (bundle / "batch.pt").exists()


def test_invariant_no_optimizer_state_when_optimizer_none(tmp_path: Path) -> None:
    """When ``optimizer=None``, optimizer_state.pt is not written."""
    bundle = _minimal_bundle(tmp_path)
    assert not (bundle / "optimizer_state.pt").exists()


def test_invariant_no_optimizer_state_when_no_state_dict(tmp_path: Path) -> None:
    """When ``optimizer`` lacks ``state_dict``, optimizer_state.pt is not written."""

    class _OptimizerWithoutStateDict:
        pass

    bundle = _minimal_bundle(tmp_path, optimizer=_OptimizerWithoutStateDict())
    assert not (bundle / "optimizer_state.pt").exists()


def test_invariant_optimizer_state_written_when_available(tmp_path: Path) -> None:
    """When a real optimizer is supplied, optimizer_state.pt is written."""
    torch.manual_seed(0)
    model = _TinyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bundle = _minimal_bundle(tmp_path, optimizer=opt)
    expect_exists(bundle / "optimizer_state.pt", bundle, what="optimizer_state.pt")


def test_invariant_metrics_not_written_when_metrics_falsy(tmp_path: Path) -> None:
    """When ``metrics={}`` (falsy), metrics_recent.jsonl is not written."""
    bundle = _minimal_bundle(tmp_path, metrics={})
    assert not (bundle / "metrics_recent.jsonl").exists()


def test_invariant_decode_loop_multiple_samples(tmp_path: Path) -> None:
    """Each sample index is labelled in decoded.txt when N > 1."""
    ids = torch.arange(6).reshape(3, 2)  # 3 samples
    batch = {"input_ids": ids}
    bundle = _minimal_bundle(tmp_path, batch=batch, tokenizer=_GoodTokenizer())
    content = (bundle / "decoded.txt").read_text(encoding="utf-8")
    for i in range(3):
        assert f"# sample[{i}]" in content
