"""Hand-rolled .env parser + load_dotenv_if_present."""

from __future__ import annotations

import os
from pathlib import Path

from lighttrain.utils.env import load_dotenv_if_present, parse_dotenv


def test_parse_basic_kv_pairs():
    text = "FOO=bar\nBAZ=qux\n"
    assert parse_dotenv(text) == {"FOO": "bar", "BAZ": "qux"}


def test_parse_strips_quotes_and_export_prefix():
    text = (
        '# this is a comment\n'
        'export HF_TOKEN="abcd"\n'
        "  HF_ENDPOINT='https://hf-mirror.com'  \n"
        "EMPTY=\n"
    )
    out = parse_dotenv(text)
    assert out["HF_TOKEN"] == "abcd"
    assert out["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert out["EMPTY"] == ""


def test_parse_handles_inline_comments_only_when_unquoted():
    text = 'A=1 # inline\nB="2 # not inline"\n'
    out = parse_dotenv(text)
    assert out["A"] == "1"
    assert out["B"] == "2 # not inline"


def test_load_dotenv_writes_only_missing_keys(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text(
        "LIGHTTRAIN_TEST_NEW=fresh\nLIGHTTRAIN_TEST_KEEP=should_not_overwrite\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("LIGHTTRAIN_TEST_NEW", raising=False)
    monkeypatch.setenv("LIGHTTRAIN_TEST_KEEP", "original")

    written = load_dotenv_if_present(tmp_path)
    assert "LIGHTTRAIN_TEST_NEW" in written
    assert "LIGHTTRAIN_TEST_KEEP" not in written
    assert os.environ["LIGHTTRAIN_TEST_NEW"] == "fresh"
    assert os.environ["LIGHTTRAIN_TEST_KEEP"] == "original"


def test_load_dotenv_silent_when_absent(tmp_path: Path):
    assert load_dotenv_if_present(tmp_path) == []
