"""Adversarial tests for ``lighttrain.lab.compare``.

Focus on the pure helpers + the rendered report's structure:

* ``_flatten`` recursive flattening to dot-keys.
* ``_diff_configs`` reports only differing keys.
* ``_load_run_config`` priority order:
  resolved.yaml > snapshot.yaml > config.yaml.
* ``_read_last_metrics`` skips the ``step`` key and tolerates malformed
  lines.
* ``compare`` end-to-end with two run dirs.
* ``render_ascii`` contains expected section headers and metric values.
* **Pin: no statistical-significance keys in CompareReport** (no t-test,
  p-value, etc. — current design surface).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lighttrain.lab.compare import (
    CompareReport,
    _diff_configs,
    _flatten,
    _load_run_config,
    _read_last_metrics,
    compare,
    render_ascii,
)

# ---------------------------------------------------------------------------
# _flatten
# ---------------------------------------------------------------------------

def test_flatten_dot_separated_keys_recursive():
    """Closed form: ``{a: {b: {c: 1}}}`` → ``{"a.b.c": 1}``."""
    out = _flatten({"a": {"b": {"c": 1}}})
    assert out == {"a.b.c": 1}


def test_flatten_preserves_leaf_lists_and_scalars():
    """Lists at leaves stay as lists (not flattened further)."""
    out = _flatten({"a": [1, 2, 3], "b": "hello"})
    assert out == {"a": [1, 2, 3], "b": "hello"}


def test_flatten_with_prefix_prepends():
    """Calling with prefix prefixes every produced key."""
    out = _flatten({"a": 1, "b": 2}, prefix="root")
    assert out == {"root.a": 1, "root.b": 2}


def test_flatten_empty_dict_yields_empty():
    """Empty input → empty output."""
    assert _flatten({}) == {}


def test_flatten_non_dict_input_yields_empty():
    """Non-dict input (list, scalar) → empty dict (line 66 guard)."""
    assert _flatten([1, 2, 3]) == {}
    assert _flatten(42) == {}


# ---------------------------------------------------------------------------
# _diff_configs
# ---------------------------------------------------------------------------

def test_diff_configs_returns_only_changed_keys():
    """Keys whose values are identical across configs are dropped from the diff."""
    cfgs = [
        {"a": 1, "b": 2, "c": 3},
        {"a": 1, "b": 99, "c": 3},
    ]
    diff = _diff_configs(cfgs)
    assert "a" not in diff
    assert "c" not in diff
    assert "b" in diff
    assert diff["b"] == [2, 99]


def test_diff_configs_uses_repr_for_comparison():
    """Pin: ``repr`` is used to compare values (line 88), so ``1`` (int)
    and ``True`` (bool) are considered DIFFERENT despite ``1 == True``.

    Setup: cfg1 has ``flag: 1``, cfg2 has ``flag: True``.
    Expected: both appear in the diff.
    """
    cfgs = [{"flag": 1}, {"flag": True}]
    diff = _diff_configs(cfgs)
    assert "flag" in diff


def test_diff_configs_missing_key_in_one_run_appears_as_none():
    """Run A has key X, run B does not → X appears with [val, None]."""
    cfgs = [{"a": 1, "extra": "x"}, {"a": 1}]
    diff = _diff_configs(cfgs)
    assert diff["extra"] == ["x", None]


def test_diff_configs_empty_input_yields_empty():
    """``_diff_configs([])`` returns empty dict."""
    assert _diff_configs([]) == {}


def test_diff_configs_single_run_yields_empty():
    """A single run has no differences with itself."""
    assert _diff_configs([{"a": 1}]) == {}


# ---------------------------------------------------------------------------
# _load_run_config priority
# ---------------------------------------------------------------------------

def test_pin_load_config_prefers_resolved_yaml(tmp_path: Path):
    """Pin: ``config.resolved.yaml`` wins over snapshot.yaml and config.yaml
    (line 50 of source — first item in the priority list).
    """
    (tmp_path / "config.resolved.yaml").write_text("source: resolved", encoding="utf-8")
    (tmp_path / "config.snapshot.yaml").write_text("source: snapshot", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("source: plain", encoding="utf-8")
    cfg = _load_run_config(tmp_path)
    assert cfg.get("source") == "resolved"


def test_pin_load_config_falls_back_to_snapshot(tmp_path: Path):
    """When resolved is missing, snapshot is used."""
    (tmp_path / "config.snapshot.yaml").write_text("source: snapshot", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("source: plain", encoding="utf-8")
    cfg = _load_run_config(tmp_path)
    assert cfg.get("source") == "snapshot"


def test_load_config_returns_empty_when_none_present(tmp_path: Path):
    """No config files → empty dict (not raise)."""
    assert _load_run_config(tmp_path) == {}


def test_load_config_tolerates_malformed_yaml(tmp_path: Path):
    """Malformed YAML falls through to the empty dict (line 58-60)."""
    (tmp_path / "config.resolved.yaml").write_text("[ unclosed", encoding="utf-8")
    assert _load_run_config(tmp_path) == {}


# ---------------------------------------------------------------------------
# _read_last_metrics
# ---------------------------------------------------------------------------

def test_read_last_metrics_returns_last_value_per_metric(tmp_path: Path):
    """Each metric → its last numeric value (overwritten by later lines)."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "metrics.jsonl").write_text(
        '{"step": 1, "loss": 1.0, "acc": 0.5}\n'
        '{"step": 2, "loss": 0.5, "acc": 0.7}\n',
        encoding="utf-8",
    )
    out = _read_last_metrics(tmp_path)
    assert out == {"loss": 0.5, "acc": 0.7}


def test_invariant_read_last_metrics_skips_step_key():
    """Pin: the ``step`` key is explicitly excluded from the metrics dict
    (line 116: ``if isinstance(v, (int, float)) and k != "step"``).
    """
    # Inline test — build a fake JSONL.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        logs = tmp_path / "logs"
        logs.mkdir()
        (logs / "metrics.jsonl").write_text(
            '{"step": 99, "loss": 0.5}\n', encoding="utf-8"
        )
        out = _read_last_metrics(tmp_path)
        assert "step" not in out
        assert out["loss"] == pytest.approx(0.5)


def test_read_last_metrics_skips_non_numeric_values(tmp_path: Path):
    """String values are excluded (line 116 isinstance check)."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "metrics.jsonl").write_text(
        '{"step": 1, "loss": 0.5, "label": "checkpoint"}\n',
        encoding="utf-8",
    )
    out = _read_last_metrics(tmp_path)
    assert "label" not in out
    assert out["loss"] == pytest.approx(0.5)


def test_read_last_metrics_returns_empty_when_jsonl_missing(tmp_path: Path):
    """No metrics file → empty dict."""
    assert _read_last_metrics(tmp_path) == {}


# ---------------------------------------------------------------------------
# compare end-to-end
# ---------------------------------------------------------------------------

def _make_run(root: Path, name: str, cfg: dict, metrics: list[dict]) -> Path:
    rd = root / name
    rd.mkdir()
    (rd / "config.resolved.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    logs = rd / "logs"
    logs.mkdir()
    (logs / "metrics.jsonl").write_text(
        "\n".join(json.dumps(m) for m in metrics) + "\n",
        encoding="utf-8",
    )
    return rd


def test_compare_two_runs_yields_config_diff_and_metrics_table(tmp_path: Path):
    """End-to-end compare: 2 runs with different lr produce the expected
    diff + metrics table.
    """
    a = _make_run(
        tmp_path, "run_a",
        cfg={"optim": {"lr": 1e-3}, "seed": 0},
        metrics=[{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.8}],
    )
    b = _make_run(
        tmp_path, "run_b",
        cfg={"optim": {"lr": 1e-4}, "seed": 0},
        metrics=[{"step": 1, "loss": 1.0}, {"step": 2, "loss": 0.5}],
    )

    report = compare([a, b])
    assert isinstance(report, CompareReport)
    assert "optim.lr" in report.config_diff
    assert report.config_diff["optim.lr"] == [1e-3, 1e-4]
    # seed is identical so not in diff
    assert "seed" not in report.config_diff
    # Final loss differs
    assert report.metrics_table["loss"] == [pytest.approx(0.8), pytest.approx(0.5)]


def test_render_ascii_contains_required_sections(tmp_path: Path):
    """The rendered ASCII report contains the canonical section headers."""
    a = _make_run(
        tmp_path, "run_a", cfg={"lr": 1e-3},
        metrics=[{"step": 1, "loss": 0.5}],
    )
    b = _make_run(
        tmp_path, "run_b", cfg={"lr": 1e-4},
        metrics=[{"step": 1, "loss": 0.7}],
    )
    text = render_ascii(compare([a, b]))
    assert "=== Run summary ===" in text
    assert "=== Config diff" in text
    assert "=== Final metrics ===" in text
    assert "lr" in text
    assert "loss" in text


def test_render_ascii_handles_no_diff_case(tmp_path: Path):
    """When configs are identical, the report says 'no differences'."""
    a = _make_run(tmp_path, "a", cfg={"lr": 1e-3}, metrics=[{"step": 1, "loss": 0.5}])
    b = _make_run(tmp_path, "b", cfg={"lr": 1e-3}, metrics=[{"step": 1, "loss": 0.7}])
    text = render_ascii(compare([a, b]))
    assert "no differences" in text


# ---------------------------------------------------------------------------
# Three-run compare (merged from tests/test_lab_compare.py)
# ---------------------------------------------------------------------------

def test_diff_configs_three_runs_collects_all_values():
    """Diff over 3 configs collects one value per run, in order
    (merged from test_lab_compare.test_diff_three_runs).
    """
    diff = _diff_configs([{"lr": 1e-4}, {"lr": 3e-4}, {"lr": 3e-4}])
    assert "lr" in diff
    assert diff["lr"] == [1e-4, 3e-4, 3e-4]


def test_compare_three_runs_produces_three_column_tables(tmp_path: Path):
    """End-to-end compare over 3 runs yields 3-wide diff/metrics columns
    (merged from test_lab_compare.test_compare_three_runs).
    """
    r1 = _make_run(tmp_path, "r1", cfg={"lr": 1e-4}, metrics=[{"step": 1, "loss": 3.0}])
    r2 = _make_run(tmp_path, "r2", cfg={"lr": 3e-4}, metrics=[{"step": 1, "loss": 2.0}])
    r3 = _make_run(tmp_path, "r3", cfg={"lr": 1e-3}, metrics=[{"step": 1, "loss": 2.5}])
    report = compare([r1, r2, r3])
    assert len(report.metrics_table["loss"]) == 3
    assert len(report.config_diff["lr"]) == 3


# ---------------------------------------------------------------------------
# Fork-ancestry detection + rendering (merged from tests/test_lab_compare.py)
# ---------------------------------------------------------------------------

def _make_run_with_fork(
    base: Path, name: str, cfg: dict, metrics: list[dict], fork_of: str | None = None
) -> Path:
    rd = _make_run(base, name, cfg, metrics)
    if fork_of:
        (rd / "fork_meta.json").write_text(
            json.dumps({"fork_of_run_dir": fork_of}), encoding="utf-8"
        )
    return rd


def test_compare_detects_fork_ancestry_from_fork_meta(tmp_path: Path):
    """A run with ``fork_meta.json`` pointing at a parent is reported in
    ``fork_ancestry``; a non-forked run maps to None
    (merged from test_lab_compare.test_compare_detects_fork_ancestry).
    """
    r1 = _make_run_with_fork(tmp_path, "run1", {}, [{"step": 1}])
    r2 = _make_run_with_fork(tmp_path, "run2", {}, [{"step": 1}], fork_of=str(r1))
    report = compare([r1, r2])
    assert report.fork_ancestry[str(r2)] == str(r1)
    assert report.fork_ancestry[str(r1)] is None


def test_render_ascii_shows_fork_of_when_ancestry_present(tmp_path: Path):
    """The rendered report surfaces a 'fork of' annotation when a run has a
    parent (merged from test_lab_compare.test_render_ascii_shows_fork_ancestry).
    """
    r1 = _make_run_with_fork(tmp_path, "run1", {}, [{"step": 1}])
    r2 = _make_run_with_fork(tmp_path, "run2", {}, [{"step": 1}], fork_of=str(r1))
    text = render_ascii(compare([r1, r2]))
    assert "fork of" in text.lower()


# ---------------------------------------------------------------------------
# CompareReport surface pin
# ---------------------------------------------------------------------------

def test_pin_compare_report_has_no_statistical_significance_keys():
    """Pin: CompareReport has no ``p_value``, ``t_stat``, ``ci_lower``,
    ``ci_upper`` fields — the current design does not include statistical
    significance testing.

    If you intentionally add t-test / Welch / bootstrap CI etc., update
    this test AND document the new contract.
    """
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(CompareReport)}
    forbidden = {"p_value", "t_stat", "ci_lower", "ci_upper", "welch_t", "effect_size"}
    overlap = field_names & forbidden
    assert overlap == set(), (
        f"CompareReport gained statistical-significance fields {overlap}; "
        "pin needs updating."
    )
    # Sanity: the documented fields ARE present.
    assert field_names == {
        "runs", "config_diff", "metrics_table", "fork_ancestry",
    }
