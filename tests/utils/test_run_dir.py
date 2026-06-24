"""Adversarial tests for ``lighttrain.utils.run_dir``.

Coverage:

* ``slugify`` lowercases, replaces non-allowed chars with ``-``, trims
  trailing ``-_``, applies ``max_len``, falls back to ``"run"`` for empty.
* ``make_run_dir`` creates ``logs/`` and ``checkpoints/`` subdirs.
* ``make_run_dir`` writes the snapshot/resolved YAML files when provided.
* ``make_run_dir`` writes env.json containing the captured env.
* ``make_run_dir`` produces a unique hash component so concurrent calls
  with the same inputs but different snapshot YAML get distinct dirs.
"""

from __future__ import annotations

import json
import re

import pytest

from lighttrain.utils.run_dir import make_run_dir, slugify

# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("simple", "simple"),
        ("With Caps", "with-caps"),
        ("with  spaces", "with-spaces"),
        ("special!@#chars", "special-chars"),
        ("under_score", "under_score"),       # underscore is allowed
        ("dashed-text", "dashed-text"),
        ("---leading", "leading"),             # strip leading -
        ("trailing---", "trailing"),
        ("_underscore_", "underscore"),         # strip leading/trailing _
        ("", "run"),                            # empty → "run" fallback
        ("   ", "run"),                         # whitespace-only → "run"
    ],
)
def test_invariant_slugify_canonical_forms(text, expected):
    """Closed-form slugify outputs for canonical inputs."""
    assert slugify(text) == expected


def test_invariant_slugify_max_len_truncation():
    """``slugify("x" * 100, max_len=8)`` returns 8-char output."""
    out = slugify("x" * 100, max_len=8)
    assert len(out) == 8
    assert out == "xxxxxxxx"


def test_invariant_slugify_unicode_collapsed():
    """Unicode characters become ``-`` (not allowed by the regex).

    Setup: "你好world" — non-ASCII collapses to ``-``.
    Expected: "world" (leading dashes stripped).
    """
    out = slugify("你好world")
    assert out == "world"


# ---------------------------------------------------------------------------
# make_run_dir
# ---------------------------------------------------------------------------

def test_invariant_make_run_dir_creates_logs_and_checkpoints(tmp_path):
    """``make_run_dir`` creates ``logs/`` and ``checkpoints/`` subdirs."""
    rd = make_run_dir(tmp_path, exp="my_experiment")
    assert (rd / "logs").is_dir()
    assert (rd / "checkpoints").is_dir()


def test_make_run_dir_under_exp_slug(tmp_path):
    """Run dir lives under ``<root>/<exp_slug>/``."""
    rd = make_run_dir(tmp_path, exp="Hello World")
    assert rd.parent.name == "hello-world"


def test_make_run_dir_writes_snapshot_yaml_when_provided(tmp_path):
    """When ``snapshot_yaml`` is non-empty, the file is written."""
    rd = make_run_dir(tmp_path, exp="t", snapshot_yaml="mode: lab\n")
    assert (rd / "config.snapshot.yaml").exists()
    assert (rd / "config.snapshot.yaml").read_text() == "mode: lab\n"


def test_make_run_dir_skips_snapshot_when_empty(tmp_path):
    """Empty ``snapshot_yaml`` → file not written (line 55-56 guard)."""
    rd = make_run_dir(tmp_path, exp="t", snapshot_yaml="")
    assert not (rd / "config.snapshot.yaml").exists()


def test_make_run_dir_writes_resolved_yaml(tmp_path):
    """``resolved_yaml`` writes ``config.resolved.yaml``."""
    rd = make_run_dir(tmp_path, exp="t", resolved_yaml="mode: prod\n")
    assert (rd / "config.resolved.yaml").exists()
    assert "prod" in (rd / "config.resolved.yaml").read_text()


def test_invariant_make_run_dir_writes_env_json(tmp_path):
    """``env.json`` is written with the env capture."""
    rd = make_run_dir(tmp_path, exp="t")
    env_path = rd / "env.json"
    assert env_path.exists()
    parsed = json.loads(env_path.read_text())
    assert isinstance(parsed, dict)


def test_make_run_dir_env_json_includes_extra_env(tmp_path):
    """``extra_env`` overrides / extends the captured env keys."""
    rd = make_run_dir(tmp_path, exp="t", extra_env={"_test_marker": "X"})
    parsed = json.loads((rd / "env.json").read_text())
    assert parsed["_test_marker"] == "X"


def test_invariant_make_run_dir_distinct_snapshots_yield_distinct_hashes(tmp_path):
    """Two run dirs created with different snapshot_yaml values have
    distinct trailing hash components (line 48-49 of source).
    """
    rd1 = make_run_dir(tmp_path, exp="t", snapshot_yaml="mode: lab\n")
    rd2 = make_run_dir(tmp_path, exp="t", snapshot_yaml="mode: prod\n")
    # hash component is the last token after the final '-'
    h1 = rd1.name.split("-")[-1]
    h2 = rd2.name.split("-")[-1]
    assert h1 != h2


def test_make_run_dir_uses_slug_override_when_provided(tmp_path):
    """When ``slug=`` is passed, it's used in the directory name instead
    of ``exp``.
    """
    rd = make_run_dir(tmp_path, exp="exp_a", slug="my_slug")
    # The slug appears in the dir name between ts and hash.
    rd.name.split("-")
    # parts: [ts_date, ts_time, "my_slug", ..., hash]
    # We use a more permissive check.
    assert "my_slug" in rd.name


def test_invariant_make_run_dir_name_matches_ts_slug_hash_format(tmp_path):
    """Run-dir basename pins the ``<YYYYMMDD>-<HHMMSS>-<slug>-<8hex>`` shape.

    Setup: ``slug="smoke"``. Expected: full closed-form regex match (8-digit
    date, 6-digit time, slug, 8-char hex hash).
    """
    rd = make_run_dir(tmp_path, exp="tiny_pretrain", slug="smoke")
    assert re.match(r"^\d{8}-\d{6}-smoke-[0-9a-f]{8}$", rd.name)
