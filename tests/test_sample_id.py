"""Sample id derivation tests — stable cross-process; insensitive to dict order."""

from __future__ import annotations

from lighttrain.data.core._schema import derive_sample_id


def test_derive_sample_id_stable_for_same_content():
    s1 = {"input_ids": [1, 2, 3], "metadata": {"src": "a", "lang": "en"}}
    s2 = {"input_ids": [1, 2, 3], "metadata": {"lang": "en", "src": "a"}}
    assert derive_sample_id(s1) == derive_sample_id(s2)


def test_derive_sample_id_changes_when_content_changes():
    s1 = {"input_ids": [1, 2, 3], "metadata": {"src": "a"}}
    s2 = {"input_ids": [1, 2, 4], "metadata": {"src": "a"}}
    assert derive_sample_id(s1) != derive_sample_id(s2)


def test_derive_sample_id_uses_meta_alias():
    s1 = {"input_ids": [9, 8], "meta": {"src": "x"}}
    s2 = {"input_ids": [9, 8], "metadata": {"src": "x"}}
    assert derive_sample_id(s1) == derive_sample_id(s2)


def test_derive_sample_id_only_uses_first_64_tokens():
    head = list(range(64))
    a = {"input_ids": head + [99], "metadata": {}}
    b = {"input_ids": head + [100], "metadata": {}}
    assert derive_sample_id(a) == derive_sample_id(b)


def test_derive_sample_id_format():
    sid = derive_sample_id({"input_ids": [1, 2, 3]}, prefix="x")
    assert sid.startswith("x_")
    assert len(sid) == len("x") + 1 + 16
    sid2 = derive_sample_id({"input_ids": [1, 2, 3]})
    assert sid2.startswith("s_")
