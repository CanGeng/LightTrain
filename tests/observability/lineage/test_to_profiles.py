"""v0.1.8 `migrate config --to-profiles` — text-level model-block rewrite.

The rewrite is surgical (preserves comments, blank lines, and ${...}
interpolations) and idempotent. It is orthogonal to the schema_version DAG, so
it has its own text transform rather than going through `migrate_payload`.
"""

from __future__ import annotations

import yaml
from typer.testing import CliRunner

from lighttrain.cli._app import app
from lighttrain.config import load_config
from lighttrain.config._resolver import select_model_spec
from lighttrain.observability.lineage.migration import (
    migrate_model_to_profiles_text,
    rewrite_model_to_profiles_file,
)
from tests._diagnostics import expect_exists

runner = CliRunner()

_BLOCK = """\
# header comment
mode: lab
seed: 7

model:
  name: tiny_lm
  d_model: 128  # inline comment
  vocab_size: ${seed}

data:
  name: simple
"""


def test_block_form_rewrite_structure():
    out, changed = migrate_model_to_profiles_text(_BLOCK)
    assert changed
    d = yaml.safe_load(out)
    assert d["model"] == "default"
    assert d["model_profiles"]["default"]["name"] == "tiny_lm"
    assert d["model_profiles"]["default"]["d_model"] == 128


def test_comments_and_interpolation_preserved():
    out, _ = migrate_model_to_profiles_text(_BLOCK)
    assert "# header comment" in out
    assert "# inline comment" in out
    # ${seed} stays a literal string, not coerced.
    assert "${seed}" in out
    assert select_model_spec(*_load_model(out))["vocab_size"] == "${seed}"


def test_idempotent():
    once, c1 = migrate_model_to_profiles_text(_BLOCK)
    twice, c2 = migrate_model_to_profiles_text(once)
    assert c1 is True and c2 is False
    assert once == twice  # second pass is a no-op


def test_flow_form_rewrite():
    raw = "mode: lab\nmodel: {name: tiny_lm, d_model: 64}\ndata: {name: simple}\n"
    out, changed = migrate_model_to_profiles_text(raw)
    assert changed
    d = yaml.safe_load(out)
    assert d["model"] == "default"
    assert d["model_profiles"]["default"] == {"name": "tiny_lm", "d_model": 64}


def test_no_model_block_is_noop():
    raw = "mode: lab\nparallel:\n  dp: 4\n"
    out, changed = migrate_model_to_profiles_text(raw)
    assert changed is False
    assert out == raw


def test_custom_profile_name():
    out, _ = migrate_model_to_profiles_text(_BLOCK, profile_name="transformer")
    d = yaml.safe_load(out)
    assert d["model"] == "transformer"
    assert "transformer" in d["model_profiles"]


def test_file_rewrite_creates_backup_and_is_loadable(tmp_yaml):
    p = tmp_yaml(_BLOCK)
    changed = rewrite_model_to_profiles_file(p, in_place=True)
    assert changed
    bak = p.with_suffix(p.suffix + ".pre-migration-bak")
    expect_exists(bak, bak.parent, what="pre-migration backup")
    assert bak.read_text() == _BLOCK
    # Migrated file loads and resolves through the v0.1.8 selector path.
    cfg = load_config(p)
    spec = select_model_spec(cfg.model, cfg.model_profiles)
    assert spec["name"] == "tiny_lm" and cfg.model == "default"


def test_cli_to_profiles_in_place(tmp_yaml):
    p = tmp_yaml(_BLOCK)
    res = runner.invoke(app, ["migrate", "config", str(p), "--to-profiles", "--in-place"])
    assert res.exit_code == 0, res.output
    assert "migrated" in res.output
    cfg = load_config(p)
    assert cfg.model == "default"
    assert cfg.model_profiles["default"]["name"] == "tiny_lm"


def test_cli_to_profiles_print_mode_does_not_write(tmp_yaml):
    p = tmp_yaml(_BLOCK)
    before = p.read_text()
    res = runner.invoke(app, ["migrate", "config", str(p), "--to-profiles"])
    assert res.exit_code == 0
    assert "model_profiles" in res.output
    assert p.read_text() == before  # print mode leaves the file untouched


def _load_model(text: str):
    d = yaml.safe_load(text)
    return d.get("model"), d.get("model_profiles")
