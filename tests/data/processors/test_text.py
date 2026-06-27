"""Edge-case unit tests for ``lighttrain.builtin_plugins.data.processors.text``.

Mirror of ``lighttrain/builtin_plugins/data/processors/text.py`` (the
``builtin_plugins/`` layer is stripped from the mirror path). The flat
``tests/data/test_processors.py`` only smoke-tests the happy path of
``ChatTemplateProcessor``; here we drive the uncovered branches.

What we pin:

* ``_format_segment`` — the ``head not in template`` early return and the
  in-template substitution branch.
* ``ChatTemplateProcessor`` — ``max_len`` truncation of ids/labels, the
  Mapping tokenizer spec resolved lazily through the registry, missing
  ``content`` defaulting to ``""``, and ``label_ignore``/coercion of flags.
* ``HFTextProcessor`` — ``__init__`` field coercion, lazy
  ``_ensure_tokenizer`` (import, ``from_pretrained``, ``chat_template``
  override, instance caching), and ``__call__`` rendering / encoding /
  ``max_len`` / attention-mask fallback / response-only prefix masking for
  both the assistant-terminated and non-assistant-terminated turn lists.

``transformers`` is stubbed via a hand-rolled fake tokenizer
(``_FakeTokenizer``); nothing touches the network or model files.
"""

from __future__ import annotations

from unittest import mock

import pytest
import transformers

# Importing the tokenizers module registers the ``byte`` tokenizer so the
# registry-resolved Mapping spec below succeeds.
import lighttrain.builtin_plugins.data.core.tokenizers  # noqa: F401
from lighttrain.builtin_plugins.data.core.tokenizers import ByteTokenizer
from lighttrain.builtin_plugins.data.processors.text import (
    _DEFAULT_TEMPLATE,
    ChatTemplateProcessor,
    HFTextProcessor,
    _format_segment,
)

# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Deterministic stand-in for an HF tokenizer.

    ``apply_chat_template`` renders turns as ``[role]content`` and, when
    ``add_generation_prompt`` is set, appends a bare ``[assistant]`` tag so the
    masking prefix is a strict prefix of the full rendering. ``__call__`` maps
    each character to a byte code; ``with_attention`` toggles whether the
    encoding dict carries an ``attention_mask`` (to exercise the fallback).
    """

    def __init__(self, *, with_attention: bool = True) -> None:
        self.chat_template: str | None = None
        self.with_attention = with_attention
        self.apply_calls: list[tuple[tuple[str, ...], bool]] = []
        self.encode_calls = 0

    def apply_chat_template(self, turns, *, tokenize, add_generation_prompt):
        assert tokenize is False
        self.apply_calls.append(
            (tuple(t["role"] for t in turns), add_generation_prompt)
        )
        rendered = "".join(f"[{t['role']}]{t.get('content', '')}" for t in turns)
        if add_generation_prompt:
            rendered += "[assistant]"
        return rendered

    def __call__(self, text, *, return_offsets_mapping=False, add_special_tokens=False):
        assert return_offsets_mapping is False
        assert add_special_tokens is False
        self.encode_calls += 1
        ids = [ord(c) % 256 for c in text]
        out = {"input_ids": ids}
        if self.with_attention:
            out["attention_mask"] = [1] * len(ids)
        return out


def _patch_from_pretrained(fake):
    """Patch ``AutoTokenizer.from_pretrained`` to return ``fake``."""
    return mock.patch.object(
        transformers.AutoTokenizer, "from_pretrained", return_value=fake
    )


# ---------------------------------------------------------------------------
# _format_segment
# ---------------------------------------------------------------------------


def test_invariant_format_segment_missing_head_returns_default_block():
    """When ``<|role|>`` is absent the helper returns a fresh
    ``<|role|>\\n{content}\\n`` block (line 32 early return)."""
    out = _format_segment("only {user} text", "assistant", "X")
    assert out == "<|assistant|>\nX\n"


def test_invariant_format_segment_substitutes_content_within_template():
    """When the head is present the body's ``{role}`` placeholder is replaced
    and only the section up to the next ``<|`` token is kept."""
    out = _format_segment(_DEFAULT_TEMPLATE, "user", "HELLO")
    assert out == "<|user|>\nHELLO\n"


def test_invariant_format_segment_last_role_runs_to_eos():
    """The trailing role has no following ``<|`` token, so its body runs to the
    end of the template (``end < 0`` branch)."""
    out = _format_segment(_DEFAULT_TEMPLATE, "assistant", "BYE")
    assert out == "<|assistant|>\nBYE\n"


# ---------------------------------------------------------------------------
# ChatTemplateProcessor
# ---------------------------------------------------------------------------


def test_invariant_chat_template_max_len_truncates_all_fields():
    """``max_len`` clips ids, labels and attention_mask to the same length
    (lines 99-100)."""
    proc = ChatTemplateProcessor(tokenizer=ByteTokenizer(), max_len=5)
    out = proc(
        [
            {"role": "user", "content": "hello world this is quite long"},
            {"role": "assistant", "content": "a reply that is also long"},
        ]
    )
    assert len(out["input_ids"]) == 5
    assert len(out["labels"]) == 5
    assert len(out["attention_mask"]) == 5
    assert out["attention_mask"] == [1] * 5


def test_invariant_chat_template_mapping_spec_resolved_via_registry():
    """A Mapping tokenizer spec is resolved lazily through the registry into a
    real ``ByteTokenizer`` instance (lines 71-74)."""
    proc = ChatTemplateProcessor(tokenizer={"name": "byte"})
    assert isinstance(proc.tokenizer, ByteTokenizer)
    out = proc(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Yo"},
        ]
    )
    assert out["modality"] == "text"
    assert len(out["input_ids"]) == len(out["labels"])


def test_invariant_chat_template_missing_content_defaults_to_empty():
    """A turn without a ``content`` key is treated as empty content (no
    KeyError); the rendered segment is still tokenized."""
    proc = ChatTemplateProcessor(tokenizer=ByteTokenizer())
    out = proc([{"role": "user"}])
    assert out["input_ids"]
    assert all(x == -100 for x in out["labels"])  # user turn fully masked


def test_invariant_chat_template_response_only_mask_partitions_labels():
    """With ``response_only_mask`` the non-assistant turns get ``label_ignore``
    while the assistant turn keeps its token ids."""
    proc = ChatTemplateProcessor(tokenizer=ByteTokenizer(), response_only_mask=True)
    out = proc(
        [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert any(x == -100 for x in out["labels"])
    assert any(x != -100 for x in out["labels"])


def test_invariant_chat_template_no_mask_keeps_every_label():
    """With ``response_only_mask=False`` labels equal input_ids verbatim."""
    proc = ChatTemplateProcessor(tokenizer=ByteTokenizer(), response_only_mask=False)
    out = proc(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
    )
    assert out["labels"] == out["input_ids"]


def test_invariant_chat_template_custom_label_ignore_used():
    """A custom ``label_ignore`` replaces the default ``-100`` for masked
    tokens, and the int-coercion of the flag/sentinel holds."""
    proc = ChatTemplateProcessor(
        tokenizer=ByteTokenizer(),
        response_only_mask=True,
        label_ignore=-999,
    )
    out = proc([{"role": "user", "content": "hi"}])
    assert proc.label_ignore == -999
    assert all(x == -999 for x in out["labels"])


@pytest.mark.parametrize("falsy_max_len", [None, 0])
def test_invariant_chat_template_falsy_max_len_disables_truncation(falsy_max_len):
    """Both ``None`` and ``0`` leave ``max_len`` as ``None`` (no truncation)."""
    proc = ChatTemplateProcessor(tokenizer=ByteTokenizer(), max_len=falsy_max_len)
    assert proc.max_len is None
    out = proc([{"role": "user", "content": "abcdef"}])
    assert len(out["input_ids"]) > 5  # would be clipped to 5 if truncating


# ---------------------------------------------------------------------------
# HFTextProcessor.__init__ + _ensure_tokenizer
# ---------------------------------------------------------------------------


def test_invariant_hf_init_coerces_and_stores_fields():
    """``__init__`` stores config and coerces flags/sentinels (lines 132-139);
    the tokenizer stays unloaded until first use."""
    proc = HFTextProcessor(
        model_name_or_path="org/model",
        chat_template="TMPL",
        max_len=8,
        add_generation_prompt=1,  # type: ignore[arg-type]  # truthy -> coerced to bool True
        response_only_mask=0,  # type: ignore[arg-type]  # falsy -> coerced to bool False
        label_ignore=-7,
        from_pretrained_kwargs={"revision": "main"},
    )
    assert proc.model_name_or_path == "org/model"
    assert proc.chat_template == "TMPL"
    assert proc.max_len == 8
    assert proc.add_generation_prompt is True
    assert proc.response_only_mask is False
    assert proc.label_ignore == -7
    assert proc._fp_kwargs == {"revision": "main"}
    assert proc._fp_kwargs is not None  # copied, not the original mapping
    assert proc._tokenizer is None


@pytest.mark.parametrize("falsy_max_len", [None, 0])
def test_invariant_hf_init_falsy_max_len_is_none(falsy_max_len):
    """``max_len`` of ``None``/``0`` collapses to ``None`` (line 134)."""
    proc = HFTextProcessor(model_name_or_path="x", max_len=falsy_max_len)
    assert proc.max_len is None


def test_invariant_hf_init_default_fp_kwargs_empty_dict():
    """Omitting ``from_pretrained_kwargs`` yields an empty dict, not None."""
    proc = HFTextProcessor(model_name_or_path="x")
    assert proc._fp_kwargs == {}


def test_invariant_hf_ensure_tokenizer_loads_applies_template_and_caches():
    """``_ensure_tokenizer`` calls ``from_pretrained`` once, applies the
    ``chat_template`` override, and caches the instance (lines 141-151)."""
    fake = _FakeTokenizer()
    proc = HFTextProcessor(
        model_name_or_path="org/model",
        chat_template="MY_TEMPLATE",
        from_pretrained_kwargs={"revision": "abc"},
    )
    with _patch_from_pretrained(fake) as m:
        tk1 = proc._ensure_tokenizer()
        tk2 = proc._ensure_tokenizer()

    assert tk1 is fake
    assert tk2 is fake  # cached: second call returns the same object
    m.assert_called_once_with("org/model", revision="abc")
    assert fake.chat_template == "MY_TEMPLATE"  # override applied


def test_invariant_hf_ensure_tokenizer_no_template_override_when_falsy():
    """A None/empty ``chat_template`` leaves the tokenizer's template
    untouched (the ``if self.chat_template`` guard is False)."""
    fake = _FakeTokenizer()
    fake.chat_template = "ORIGINAL"
    proc = HFTextProcessor(model_name_or_path="x", chat_template=None)
    with _patch_from_pretrained(fake):
        proc._ensure_tokenizer()
    assert fake.chat_template == "ORIGINAL"


# ---------------------------------------------------------------------------
# HFTextProcessor.__call__
# ---------------------------------------------------------------------------


def test_invariant_hf_call_masks_prefix_for_assistant_terminated_turns():
    """When the turn list ends with an assistant turn, the prefix (all earlier
    turns + generation prompt) is masked and only the final assistant span
    keeps its labels (lines 158-184, the ``turns[-1]['role']=='assistant'``
    branch on line 175)."""
    fake = _FakeTokenizer()
    proc = HFTextProcessor(model_name_or_path="x", response_only_mask=True)
    turns = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Yo"},
    ]
    with _patch_from_pretrained(fake):
        out = proc(turns)

    assert out["modality"] == "text"
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"])
    masked = [x for x in out["labels"] if x == -100]
    kept = [x for x in out["labels"] if x != -100]
    assert masked and kept
    # prefix call drops the trailing assistant turn and adds a generation prompt
    assert fake.apply_calls == [(("user", "assistant"), False), (("user",), True)]


def test_invariant_hf_call_masks_all_turns_when_not_assistant_terminated():
    """When the last turn is NOT assistant, the prefix is the full turn list
    (line 175 ``else turns``); every label up to ``cut`` is masked."""
    fake = _FakeTokenizer()
    proc = HFTextProcessor(model_name_or_path="x", response_only_mask=True)
    turns = [{"role": "user", "content": "Hi"}]
    with _patch_from_pretrained(fake):
        out = proc(turns)

    assert all(x == -100 for x in out["labels"])
    # second apply_chat_template uses the whole turn list with generation prompt
    assert fake.apply_calls[1] == (("user",), True)


def test_invariant_hf_call_max_len_truncates_ids_and_attention():
    """``max_len`` clips ids and attention_mask before labels are built
    (lines 168-170); labels track the truncated ids length."""
    fake = _FakeTokenizer()
    proc = HFTextProcessor(model_name_or_path="x", max_len=3, response_only_mask=True)
    with _patch_from_pretrained(fake):
        out = proc([{"role": "user", "content": "abcdef"}])

    assert len(out["input_ids"]) == 3
    assert len(out["attention_mask"]) == 3
    assert len(out["labels"]) == 3


def test_invariant_hf_call_attention_mask_fallback_when_absent():
    """If the encoding omits ``attention_mask`` a default all-ones mask of the
    id length is synthesized (line 167)."""
    fake = _FakeTokenizer(with_attention=False)
    proc = HFTextProcessor(model_name_or_path="x", response_only_mask=False)
    with _patch_from_pretrained(fake):
        out = proc([{"role": "user", "content": "hey"}])

    assert out["attention_mask"] == [1] * len(out["input_ids"])


def test_invariant_hf_call_no_mask_keeps_labels_equal_to_ids():
    """With ``response_only_mask=False`` the prefix-masking block is skipped
    and labels equal input_ids (line 172 guard False)."""
    fake = _FakeTokenizer()
    proc = HFTextProcessor(model_name_or_path="x", response_only_mask=False)
    with _patch_from_pretrained(fake):
        out = proc(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ]
        )

    assert out["labels"] == out["input_ids"]
    # only one apply_chat_template call (no prefix render) when masking is off
    assert len(fake.apply_calls) == 1


def test_invariant_hf_call_accepts_generator_turns():
    """``turns`` is materialized with ``list(...)`` so a one-shot generator is
    consumed safely (line 159)."""
    fake = _FakeTokenizer()
    proc = HFTextProcessor(model_name_or_path="x", response_only_mask=True)
    gen = (
        t
        for t in (
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Yo"},
        )
    )
    with _patch_from_pretrained(fake):
        out = proc(gen)
    assert out["input_ids"]
    assert out["modality"] == "text"


def test_invariant_hf_call_passes_add_generation_prompt_flag():
    """``add_generation_prompt`` configured on the processor is forwarded to the
    main (full) ``apply_chat_template`` render (lines 160-164)."""
    fake = _FakeTokenizer()
    proc = HFTextProcessor(
        model_name_or_path="x",
        add_generation_prompt=True,
        response_only_mask=False,
    )
    with _patch_from_pretrained(fake):
        proc([{"role": "user", "content": "hi"}])
    # the (sole) render uses add_generation_prompt=True
    assert fake.apply_calls == [(("user",), True)]
