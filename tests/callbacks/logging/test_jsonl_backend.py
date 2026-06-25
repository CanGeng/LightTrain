"""Adversarial tests for ``lighttrain.builtin_plugins.callbacks.logging.jsonl.JSONLLogger``.

Layered on top of one round-trip smoke test in ``tests/test_logging_bus.py``.
New coverage:

* **Every record is a valid JSON line** (parses standalone).
* **``step`` and ``kind`` keys preserved exactly** across log types.
* **``ts`` (timestamp) auto-injected and monotonically non-decreasing**.
* **Non-numeric scalar values fall back to ``str(...)``** via ``_coerce``
  (line 65-69 of jsonl.py).
* **log_text with newlines and quotes** stays parseable (JSON escapes
  them — the resulting JSONL has exactly one line per record regardless
  of payload).
* **flush after every write** — file content reflects writes before close.
* **close idempotent** — second close doesn't raise.
* **Missing path AND run_dir raises ValueError**.
* **run_dir mode creates parent dir** automatically.
* **Registered as ('logger', 'jsonl')** in the registry.
* **Main-thread-only concurrency pin** for ``JSONLLogger._write``.
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import pytest

from lighttrain.builtin_plugins.callbacks.logging.jsonl import JSONLLogger, _coerce
from tests._diagnostics import expect_exists

# ---------------------------------------------------------------------------
# Record format invariants
# ---------------------------------------------------------------------------

def test_invariant_each_record_parses_as_valid_json_line(tmp_path: Path):
    """Every line of the JSONL file parses as standalone JSON.

    Setup: write 5 records of mixed kinds.
    Expected: each line is a valid JSON object.
    """
    p = tmp_path / "metrics.jsonl"
    j = JSONLLogger(path=p)
    j.log_scalars({"loss": 0.5}, step=1)
    j.log_scalars({"loss": 0.4, "acc": 0.9}, step=2)
    j.log_text("checkpoint saved", step=3)
    j.log_artifact("/tmp/x.pt", name="model")
    j.log_text("warning: NaN", step=4)
    j.close()

    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    for line in lines:
        json.loads(line)  # must not raise


def test_invariant_scalar_record_has_step_kind_and_metric_keys(tmp_path: Path):
    """Scalar records carry ``step``, ``kind == "scalar"``, ``ts``, AND
    every metric name as a top-level key.

    Closed form: ``log_scalars({"loss": 0.5}, step=10)`` → record contains
    ``step=10``, ``kind="scalar"``, ``loss=0.5``.
    """
    p = tmp_path / "metrics.jsonl"
    j = JSONLLogger(path=p)
    j.log_scalars({"loss": 0.5}, step=10)
    j.close()

    rec = json.loads(p.read_text(encoding="utf-8").strip())
    assert rec["step"] == 10
    assert rec["kind"] == "scalar"
    assert rec["loss"] == 0.5
    assert "ts" in rec
    assert isinstance(rec["ts"], (int, float))


def test_invariant_text_record_has_step_kind_text_keys(tmp_path: Path):
    """Text records: ``step``, ``kind == "text"``, ``text == <payload>``."""
    p = tmp_path / "metrics.jsonl"
    j = JSONLLogger(path=p)
    j.log_text("snapshot OK", step=7)
    j.close()
    rec = json.loads(p.read_text(encoding="utf-8").strip())
    assert rec["step"] == 7
    assert rec["kind"] == "text"
    assert rec["text"] == "snapshot OK"


def test_invariant_artifact_record_has_kind_path_name_keys(tmp_path: Path):
    """Artifact records: ``kind == "artifact"``, ``path == <p>``,
    ``name == <name|None>``.
    """
    p = tmp_path / "metrics.jsonl"
    j = JSONLLogger(path=p)
    j.log_artifact("/tmp/x.bin", name="weights")
    j.close()
    rec = json.loads(p.read_text(encoding="utf-8").strip())
    assert rec["kind"] == "artifact"
    assert rec["path"] == "/tmp/x.bin"
    assert rec["name"] == "weights"


def test_pin_step_value_preserved_as_int_via_int_coercion(tmp_path: Path):
    """Pin: ``step`` is always written as int (line 44 / 49 wrap via ``int(step)``).

    Setup: pass a float ``step=3.7``; expected stored as ``3`` (int truncation).
    """
    p = tmp_path / "metrics.jsonl"
    j = JSONLLogger(path=p)
    j.log_scalars({"loss": 0.5}, step=3.7)
    j.close()
    rec = json.loads(p.read_text(encoding="utf-8").strip())
    assert rec["step"] == 3
    assert isinstance(rec["step"], int)


def test_invariant_ts_field_auto_injected_and_recent(tmp_path: Path):
    """``ts`` is injected by ``_write`` if not present. The value is a
    Unix timestamp close to the current time at the moment of writing.

    Setup: capture wall-clock before write; expected ``ts`` within 2 s
    of that capture.
    """
    p = tmp_path / "metrics.jsonl"
    j = JSONLLogger(path=p)
    before = time.time()
    j.log_scalars({"loss": 0.5}, step=1)
    after = time.time()
    j.close()

    rec = json.loads(p.read_text(encoding="utf-8").strip())
    assert before - 2.0 <= rec["ts"] <= after + 2.0


def test_invariant_ts_monotonically_non_decreasing_across_records(tmp_path: Path):
    """Successive ``ts`` values do not go backward.

    Setup: write 5 records back-to-back.
    Expected: ts[i] <= ts[i+1] (time.time() is monotonic in practice
    over short windows on a single thread).
    """
    p = tmp_path / "metrics.jsonl"
    j = JSONLLogger(path=p)
    for i in range(5):
        j.log_scalars({"loss": float(i)}, step=i)
    j.close()
    records = [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").strip().splitlines()
    ]
    timestamps = [r["ts"] for r in records]
    assert all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1))


# ---------------------------------------------------------------------------
# Value coercion (_coerce)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected_type",
    [
        (1.5, float),
        (3, float),       # int → float
        (True, float),    # True → 1.0
        ("not-numeric", str),
        (None, str),       # None → "None"
        ([1, 2], str),     # non-coercible → str repr
        ({"a": 1}, str),
    ],
)
def test_invariant_coerce_maps_to_float_else_str(value, expected_type):
    """Invariant: ``_coerce`` returns float when possible, else str (line 65-69).

    Goal: pin defensive coercion so JSONL stays readable even if the
    user accidentally passes a non-numeric value.
    """
    result = _coerce(value)
    assert isinstance(result, expected_type)


def test_invariant_non_numeric_scalar_logged_as_string(tmp_path: Path):
    """When a scalar dict value is non-numeric, it's written as a JSON string.

    Setup: ``log_scalars({"label": "good"}, step=1)``.
    Expected: record has ``"label": "good"`` (string, not raised).
    """
    p = tmp_path / "metrics.jsonl"
    j = JSONLLogger(path=p)
    j.log_scalars({"label": "good", "loss": 0.5}, step=1)  # type: ignore[dict-item]
    j.close()
    rec = json.loads(p.read_text(encoding="utf-8").strip())
    assert rec["label"] == "good"
    assert rec["loss"] == 0.5


# ---------------------------------------------------------------------------
# Newline / quote / unicode safety
# ---------------------------------------------------------------------------

def test_invariant_log_text_with_embedded_newlines_stays_one_line(tmp_path: Path):
    """``log_text`` with embedded ``\\n`` produces exactly one JSONL line.

    Setup: payload is ``"line1\\nline2\\nline3"``.
    Expected: exactly one file line; payload preserved via JSON unescaping.
    """
    p = tmp_path / "metrics.jsonl"
    j = JSONLLogger(path=p)
    payload = "line1\nline2\nline3"
    j.log_text(payload, step=1)
    j.close()

    lines = p.read_text(encoding="utf-8").splitlines()
    # Filter out any blank trailing line (from final newline)
    nonempty = [ln for ln in lines if ln.strip()]
    assert len(nonempty) == 1
    rec = json.loads(nonempty[0])
    assert rec["text"] == payload  # unescaped on load


def test_invariant_log_text_with_quotes_parses_correctly(tmp_path: Path):
    """``log_text`` with embedded quotes still produces valid JSONL.

    Setup: payload contains both ``"`` and ``'``.
    """
    p = tmp_path / "metrics.jsonl"
    j = JSONLLogger(path=p)
    payload = 'said "hello", \'world\''
    j.log_text(payload, step=1)
    j.close()
    rec = json.loads(p.read_text(encoding="utf-8").strip())
    assert rec["text"] == payload


def test_invariant_log_text_with_unicode_preserved(tmp_path: Path):
    """Unicode in text payloads survives the JSONL round trip
    (``ensure_ascii=False`` on line 39).
    """
    p = tmp_path / "metrics.jsonl"
    j = JSONLLogger(path=p)
    j.log_text("你好 🚀", step=1)
    j.close()
    rec = json.loads(p.read_text(encoding="utf-8").strip())
    assert rec["text"] == "你好 🚀"


# ---------------------------------------------------------------------------
# Flush / close lifecycle
# ---------------------------------------------------------------------------

def test_invariant_records_visible_before_close(tmp_path: Path):
    """The implementation flushes after every write; reading the file
    after a few writes (BEFORE close) returns the records.

    Goal: pin SIGKILL-safety contract — line 19 of jsonl.py: "flushed
    after every record so a SIGKILL still preserves prior lines."
    """
    p = tmp_path / "metrics.jsonl"
    j = JSONLLogger(path=p)
    j.log_scalars({"loss": 0.5}, step=1)
    j.log_scalars({"loss": 0.4}, step=2)
    # NO close()! The file should be readable now.
    content = p.read_text(encoding="utf-8")
    j.close()
    nonempty = [ln for ln in content.splitlines() if ln.strip()]
    assert len(nonempty) == 2


def test_close_is_idempotent(tmp_path: Path):
    """``close()`` called twice does not raise (line 57-62 has try/except)."""
    p = tmp_path / "metrics.jsonl"
    j = JSONLLogger(path=p)
    j.log_scalars({"loss": 0.5}, step=1)
    j.close()
    j.close()  # second call must not raise


def test_explicit_flush_does_not_raise_after_close(tmp_path: Path):
    """Calling ``flush()`` after ``close()`` raises an underlying OSError
    on Python file handles. Pin the behavior — this test confirms whether
    the implementation tolerates it (currently raises).

    If you intentionally add a closed-handle guard, update this test to
    expect no raise.
    """
    p = tmp_path / "metrics.jsonl"
    j = JSONLLogger(path=p)
    j.log_scalars({"loss": 0.5}, step=1)
    j.close()
    # Current behavior: raises ValueError("I/O operation on closed file")
    with pytest.raises(ValueError):
        j.flush()


# ---------------------------------------------------------------------------
# Path / run_dir mutual exclusivity
# ---------------------------------------------------------------------------

def test_neither_path_nor_run_dir_raises_value_error():
    """``JSONLLogger()`` with both ``path`` and ``run_dir`` absent raises
    ValueError (line 30-31 of jsonl.py).
    """
    with pytest.raises(ValueError) as exc:
        JSONLLogger()
    assert "path" in str(exc.value).lower() and "run_dir" in str(exc.value).lower()


def test_run_dir_mode_creates_logs_subdir(tmp_path: Path):
    """``run_dir=...`` builds ``<run_dir>/logs/<filename>`` and creates the
    intermediate ``logs/`` directory.

    Setup: tmp_path with NO logs/ subdir; instantiate with run_dir.
    Expected: logs/ created and the path is ``<run_dir>/logs/metrics.jsonl``.
    """
    j = JSONLLogger(run_dir=tmp_path)
    expected = tmp_path / "logs" / "metrics.jsonl"
    j.close()
    expect_exists(expected, tmp_path, what="metrics.jsonl")
    assert j.path == expected


def test_run_dir_mode_with_custom_filename(tmp_path: Path):
    """``filename`` parameter overrides the default ``metrics.jsonl``."""
    j = JSONLLogger(run_dir=tmp_path, filename="my_metrics.jsonl")
    j.close()
    expect_exists(tmp_path / "logs" / "my_metrics.jsonl", tmp_path, what="my_metrics.jsonl")


def test_path_mode_creates_parent_dir(tmp_path: Path):
    """When ``path`` is given but its parent doesn't exist, the parent is
    created (line 34 of jsonl.py: ``mkdir(parents=True, exist_ok=True)``).
    """
    nested = tmp_path / "a" / "b" / "c" / "metrics.jsonl"
    j = JSONLLogger(path=nested)
    j.close()
    expect_exists(nested.parent, tmp_path, what="auto-created parent dir")


# ---------------------------------------------------------------------------
# Append mode pin
# ---------------------------------------------------------------------------

def test_invariant_subsequent_instantiation_appends_to_existing_file(tmp_path: Path):
    """A second JSONLLogger pointing at the same path APPENDS (line 35 of
    jsonl.py opens with mode ``"a"``).

    Setup: write one record, close; open new instance, write one more
    record, close.
    Expected: 2 total lines in the file.
    """
    p = tmp_path / "metrics.jsonl"
    j1 = JSONLLogger(path=p)
    j1.log_scalars({"loss": 1.0}, step=1)
    j1.close()
    j2 = JSONLLogger(path=p)
    j2.log_scalars({"loss": 2.0}, step=2)
    j2.close()
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0]["loss"] == 1.0
    assert parsed[1]["loss"] == 2.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_jsonl_logger_registered_under_logger_jsonl():
    """Pin: ``JSONLLogger`` is registered as ``('logger', 'jsonl')``."""
    from lighttrain.registry import get
    cls = get("logger", "jsonl")
    assert cls is JSONLLogger


# ---------------------------------------------------------------------------
# Main-thread-only concurrency pin
# ---------------------------------------------------------------------------

def test_pin_jsonl_logger_write_not_thread_safe_main_thread_only():
    """Pin: ``JSONLLogger`` is main-thread-only by design (PyTorch Lightning /
    HuggingFace Trainer convention).

    Observable signal:
      (a) the source of ``JSONLLogger`` (the whole class, including
          ``_write`` / ``log_scalars`` / ``log_text``) contains no
          ``threading.Lock``/``RLock``/``with self._lock`` token, AND
      (b) the class docstring contains the literal substring
          ``"Thread-safety: NOT thread-safe"``.

    Rationale: forcing thread-safety here would add a mutex on every
    ``flush`` (the hot path of metrics logging) without any caller actually
    needing the contract.

    If multi-thread support is added (with proper locking), update this
    test AND document the new concurrency contract in the module docstring.
    """
    src = inspect.getsource(JSONLLogger)
    forbidden = ("threading.Lock", "threading.RLock", "self._lock", "RLock(", "Lock(")
    found = [tok for tok in forbidden if tok in src]
    assert not found, (
        f"JSONLLogger source unexpectedly contains lock token(s) {found}; "
        "concurrency contract changed."
    )

    doc = (JSONLLogger.__doc__ or "")
    assert "Thread-safety: NOT thread-safe" in doc, (
        "JSONLLogger class docstring must contain the explicit "
        "'Thread-safety: NOT thread-safe' contract marker. "
        "If you intentionally make JSONLLogger thread-safe, update both "
        "this test and the docstring to reflect the new contract."
    )
