"""Unit tests for ``lighttrain.builtin_plugins.judges.judge``.

Coverage targets (previously uncovered lines):
* VerifierJudge.score — 3-tuple item unpack (line 77)
* VerifierJudge.score — answer_pattern + reference_key branch, including:
  - ref value extracted via pattern match (lines 82-87)
  - ref value used verbatim when pattern does not match ref (lines 85-86)
  - correct / incorrect comparison (line 87)
* VerifierJudge.score — fallback s=0.0 when no verify_fn and no pattern (line 92)
* PairwiseLLMJudge.score — full loop (lines 150-167):
  - winner == "A" → 1.0 (line 161)
  - winner == "B" → 0.0 (line 161)
  - no match → 0.5 tie (line 163)
  - judge_model_fn raises → 0.5 + warning logged (lines 164-166)
* VerifierJudge constructor: verify_pattern alias overrides answer_pattern
* PairwiseLLMJudge constructor: custom prompt_template; custom win_pattern
* Registration in registry under "judge" category
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from lighttrain.builtin_plugins.judges.judge import PairwiseLLMJudge, VerifierJudge

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _always_one(prompt: str, response: str) -> float:
    """Stub verify_fn that always returns 1.0."""
    return 1.0


def _always_zero(prompt: str, response: str) -> float:
    """Stub verify_fn that always returns 0.0."""
    return 0.0


def _continuous(prompt: str, response: str) -> float:
    """Stub verify_fn that returns a continuous score."""
    return 0.75


# ---------------------------------------------------------------------------
# VerifierJudge — constructor
# ---------------------------------------------------------------------------


def test_invariant_default_answer_pattern_compiles():
    """Default answer_pattern ``r'#### (\\S+)'`` is compiled to a regex object."""
    j = VerifierJudge()
    assert j.answer_pattern is not None
    assert j.answer_pattern.search("#### 42") is not None


def test_invariant_verify_pattern_alias_overrides_answer_pattern():
    """``verify_pattern`` kwarg (RL recipe alias) overrides ``answer_pattern``."""
    j = VerifierJudge(verify_pattern=r"ANSWER: (\S+)")
    assert j.answer_pattern is not None
    m = j.answer_pattern.search("ANSWER: hello")
    assert m is not None and m.group(1) == "hello"


def test_invariant_answer_pattern_none_disabled():
    """Passing ``answer_pattern=None`` disables regex mode."""
    j = VerifierJudge(answer_pattern=None)
    assert j.answer_pattern is None


def test_invariant_custom_reference_key():
    """``reference_key`` kwarg is stored on the instance."""
    j = VerifierJudge(reference_key="gold")
    assert j.reference_key == "gold"


def test_invariant_reward_kind_is_pointwise():
    """``reward_kind`` class attribute is ``'pointwise'``."""
    assert VerifierJudge.reward_kind == "pointwise"


# ---------------------------------------------------------------------------
# VerifierJudge.score — verify_fn branch (2-tuple and 3-tuple items)
# ---------------------------------------------------------------------------


def test_invariant_verify_fn_called_with_2_tuple():
    """verify_fn receives (prompt, response) from a 2-tuple item."""
    received: list[tuple[str, str]] = []

    def _capture(p: str, r: str) -> float:
        received.append((p, r))
        return 1.0

    j = VerifierJudge(verify_fn=_capture)
    scores = j.score([("hello", "world")])
    assert scores == [1.0]
    assert received == [("hello", "world")]


def test_invariant_verify_fn_called_with_3_tuple():
    """3-tuple items unpack correctly; extras are ignored when verify_fn is set (line 77)."""
    received: list[tuple[str, str]] = []

    def _capture(p: str, r: str) -> float:
        received.append((p, r))
        return 0.5

    j = VerifierJudge(verify_fn=_capture)
    scores = j.score([("q", "r", {"answer": "42"})])
    assert scores == [pytest.approx(0.5)]
    assert received == [("q", "r")]


def test_invariant_verify_fn_float_cast():
    """verify_fn return value is cast to float (int return is accepted)."""

    def _int_return(p: str, r: str) -> float:
        return 1

    j = VerifierJudge(verify_fn=_int_return)
    scores = j.score([("p", "r")])
    assert isinstance(scores[0], float)
    assert scores[0] == 1.0


def test_invariant_verify_fn_multiple_items():
    """verify_fn is called per item; returned list has same length."""
    j = VerifierJudge(verify_fn=_always_one)
    items = [("q1", "r1"), ("q2", "r2"), ("q3", "r3")]
    scores = j.score(items)
    assert scores == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("fn,expected", [
    (_always_one, 1.0),
    (_always_zero, 0.0),
    (_continuous, 0.75),
])
def test_invariant_verify_fn_values(fn, expected):
    """verify_fn scores are forwarded unchanged through score()."""
    j = VerifierJudge(verify_fn=fn)
    scores = j.score([("p", "r")])
    assert scores[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# VerifierJudge.score — answer_pattern + reference_key branch (lines 82-87)
# ---------------------------------------------------------------------------


def test_invariant_answer_pattern_correct_match_scores_1():
    """When extracted pred == ref value, score is 1.0 (line 87)."""
    j = VerifierJudge()  # pattern = r"#### (\S+)"
    # Both response and reference contain a matching pattern
    items = [("What is 1+1?", "Some text #### 2", {"answer": "#### 2"})]
    scores = j.score(items)
    assert scores == [1.0]


def test_invariant_answer_pattern_incorrect_match_scores_0():
    """When pred != ref value, score is 0.0 (line 87)."""
    j = VerifierJudge()
    items = [("What is 1+1?", "The answer is #### 3", {"answer": "#### 2"})]
    scores = j.score(items)
    assert scores == [0.0]


def test_invariant_answer_pattern_ref_without_pattern_uses_verbatim(  # lines 85-86
):
    """When the ref string does NOT match the pattern, ref is used verbatim (line 86)."""
    j = VerifierJudge()  # pattern = r"#### (\S+)"
    # Reference "42" has no "#### " prefix — ref_m is None → ref_val = ref = "42"
    items = [("q", "some text #### 42", {"answer": "42"})]
    scores = j.score(items)
    assert scores == [1.0]


def test_invariant_answer_pattern_response_no_match_gives_empty_pred():
    """When response has no pattern match, pred=''; if ref is also '' → 1.0."""
    j = VerifierJudge()
    # response has no "####" → pred = ""; ref_m on "   " also fails → ref_val = "   "
    # pred("").strip() != ref_val("   ").strip() ("" vs "")... actually strip("   ") == ""
    # So pred="" and ref_val="   ".strip()="" → equal → 1.0? Pin current behavior:
    items = [("q", "no match here", {"answer": "   "})]
    scores = j.score(items)
    # Both strip to "" — score should be 1.0 (current behavior)
    assert scores == [1.0]


def test_pin_current_behavior_answer_pattern_pred_empty_ref_nonempty_scores_0():
    """Pin: response has no match → pred=''; ref has no pattern match → ref_val=ref.
    If ref is non-empty after strip, pred != ref_val → 0.0 (current behavior, line 87).
    """
    j = VerifierJudge()
    items = [("q", "no match", {"answer": "42"})]
    scores = j.score(items)
    assert scores == [0.0]


def test_invariant_answer_pattern_strips_before_compare():
    """Comparison uses .strip() on both sides (line 87)."""
    VerifierJudge()
    # response: "#### 42 " with trailing space; ref value: " 42" with leading space
    # ref "#### 42 " → ref_m.group(1)=" 42 "... actually \S+ stops at space boundary
    # Let's use a simpler custom pattern that captures with spaces
    j2 = VerifierJudge(answer_pattern=r"ANS:(.*)")
    items = [("q", "ANS:  42 ", {"answer": "ANS: 42"})]
    scores = j2.score(items)
    # pred = "  42 ", ref_val = " 42"; strip both → "42" == "42" → 1.0
    assert scores == [1.0]


def test_invariant_answer_pattern_skipped_when_reference_key_missing():
    """When reference_key is absent from extras, pattern branch is skipped.

    Falls through to the 'pure response regex mode' (no reference), line 89-90.
    """
    j = VerifierJudge()  # reference_key="answer"
    # extras has no "answer" key → should fall to pure-regex mode
    items: list[tuple[str, str, dict[str, Any]]] = [("q", "text #### 42", {})]
    scores = j.score(items)
    # pure-regex mode: pattern matches → 1.0
    assert scores == [1.0]


def test_invariant_pure_regex_mode_no_match_scores_0():
    """Pure-regex mode (no reference): pattern doesn't match → 0.0 (line 90)."""
    j = VerifierJudge()
    items: list[tuple[str, str, dict[str, Any]]] = [("q", "no hash here", {})]
    scores = j.score(items)
    assert scores == [0.0]


def test_invariant_answer_pattern_3_tuple_item_with_extras():
    """3-tuple items correctly split (prompt, response, extras) for pattern mode (line 77)."""
    j = VerifierJudge()
    items = [("prompt", "response #### 7", {"answer": "7"})]
    scores = j.score(items)
    assert scores == [1.0]


# ---------------------------------------------------------------------------
# VerifierJudge.score — fallback branch: no verify_fn, no answer_pattern (line 92)
# ---------------------------------------------------------------------------


def test_invariant_no_verify_fn_no_pattern_scores_zero():
    """Without verify_fn and without answer_pattern, score is always 0.0 (line 92)."""
    j = VerifierJudge(verify_fn=None, answer_pattern=None)
    scores = j.score([("prompt", "response"), ("q", "r", {"answer": "x"})])
    assert scores == [0.0, 0.0]


# ---------------------------------------------------------------------------
# VerifierJudge.score — empty input
# ---------------------------------------------------------------------------


def test_invariant_empty_items_returns_empty_list():
    """score([]) returns an empty list (no crash)."""
    j = VerifierJudge()
    assert j.score([]) == []


# ---------------------------------------------------------------------------
# VerifierJudge — registry
# ---------------------------------------------------------------------------


def test_invariant_verifier_judge_registered(clean_registry):
    """VerifierJudge is registered under category='judge', name='verifier'."""
    from lighttrain.registry import get_registry
    reg = get_registry()
    cls = reg.get("judge", "verifier")
    assert cls is VerifierJudge


# ---------------------------------------------------------------------------
# PairwiseLLMJudge — constructor
# ---------------------------------------------------------------------------


def test_invariant_default_template_used_when_none():
    """When prompt_template=None, the _DEFAULT_TEMPLATE is used."""
    def fn(p):
        return "Response A"
    j = PairwiseLLMJudge(fn)
    assert j.prompt_template is PairwiseLLMJudge._DEFAULT_TEMPLATE


def test_invariant_custom_template_stored():
    """A provided prompt_template is stored verbatim."""
    tpl = "Q: {question} A: {response_a} B: {response_b}"
    j = PairwiseLLMJudge(lambda p: "Response A", prompt_template=tpl)
    assert j.prompt_template == tpl


def test_invariant_win_pattern_compiled_case_insensitive():
    """Default win_pattern is compiled with re.IGNORECASE."""
    import re
    j = PairwiseLLMJudge(lambda p: "")
    assert j.win_pattern.flags & re.IGNORECASE


def test_invariant_reward_kind_is_pairwise():
    """``reward_kind`` class attribute is ``'pairwise'``."""
    assert PairwiseLLMJudge.reward_kind == "pairwise"


# ---------------------------------------------------------------------------
# PairwiseLLMJudge.score — core branches (lines 150-167)
# ---------------------------------------------------------------------------


def test_invariant_pairwise_winner_a_scores_1(caplog):
    """When judge returns 'Response A', score is 1.0 (line 161)."""
    j = PairwiseLLMJudge(lambda p: "Response A")
    scores = j.score([("What is 1+1?", "2", "two")])
    assert scores == [1.0]


def test_invariant_pairwise_winner_b_scores_0():
    """When judge returns 'Response B', score is 0.0 (line 161)."""
    j = PairwiseLLMJudge(lambda p: "Response B is clearly better.")
    scores = j.score([("q", "a", "b")])
    assert scores == [0.0]


def test_invariant_pairwise_no_match_scores_half():
    """When judge output has no match, score is 0.5 (tie, line 163)."""
    j = PairwiseLLMJudge(lambda p: "I cannot decide which is better.")
    scores = j.score([("q", "a", "b")])
    assert scores == [0.5]


def test_invariant_pairwise_exception_scores_half_and_logs_warning(caplog):
    """When judge_model_fn raises, score is 0.5 and a warning is logged (lines 164-166)."""
    def _boom(prompt: str) -> str:
        raise RuntimeError("API is down")

    j = PairwiseLLMJudge(_boom)
    with caplog.at_level(logging.WARNING, logger="lighttrain.builtin_plugins.judges.judge"):
        scores = j.score([("q", "a", "b")])
    assert scores == [0.5]
    assert any("pairwise scoring failed" in r.message for r in caplog.records)


def test_invariant_pairwise_exception_does_not_propagate():
    """Any exception from judge_model_fn is swallowed (line 164 BLE001 guard)."""
    def _raises(p: str) -> str:
        raise ValueError("bad response format")

    j = PairwiseLLMJudge(_raises)
    # Must not raise
    scores = j.score([("q", "a", "b")])
    assert scores == [0.5]


def test_invariant_pairwise_empty_items_returns_empty():
    """score([]) returns [] (no crash)."""
    j = PairwiseLLMJudge(lambda p: "Response A")
    assert j.score([]) == []


def test_invariant_pairwise_multiple_items():
    """Multiple items are scored independently in order."""
    responses = ["Response A wins", "Response B is best", "unclear result"]
    idx = [0]

    def _cycle(p: str) -> str:
        r = responses[idx[0]]
        idx[0] += 1
        return r

    j = PairwiseLLMJudge(_cycle)
    scores = j.score([("q1", "a1", "b1"), ("q2", "a2", "b2"), ("q3", "a3", "b3")])
    assert scores == [1.0, 0.0, 0.5]


def test_invariant_pairwise_case_insensitive_winner():
    """Default win_pattern is case-insensitive; 'response a' (lowercase) → 1.0."""
    j = PairwiseLLMJudge(lambda p: "response a is better")
    scores = j.score([("q", "a", "b")])
    assert scores == [1.0]


def test_invariant_pairwise_prompt_template_formatted_correctly():
    """judge_model_fn receives the prompt with question/response_a/response_b inserted."""
    received: list[str] = []

    def _capture(prompt: str) -> str:
        received.append(prompt)
        return "Response A"

    j = PairwiseLLMJudge(_capture)
    j.score([("my question", "first answer", "second answer")])
    assert len(received) == 1
    assert "my question" in received[0]
    assert "first answer" in received[0]
    assert "second answer" in received[0]


def test_invariant_pairwise_custom_template_used():
    """Custom prompt_template is used for formatting."""
    received: list[str] = []
    tpl = "Q:{question}|A:{response_a}|B:{response_b}"

    def _capture(p: str) -> str:
        received.append(p)
        return "Response A"

    j = PairwiseLLMJudge(_capture, prompt_template=tpl)
    j.score([("QQ", "AA", "BB")])
    assert received[0] == "Q:QQ|A:AA|B:BB"


def test_invariant_pairwise_custom_win_pattern():
    """A custom win_pattern is honoured; e.g., 'WINNER: A' pattern."""
    j = PairwiseLLMJudge(
        lambda p: "WINNER: A",
        win_pattern=r"WINNER: ([AB])",
    )
    scores = j.score([("q", "a", "b")])
    assert scores == [1.0]


def test_invariant_pairwise_custom_win_pattern_b():
    """Custom win_pattern extracting B → 0.0."""
    j = PairwiseLLMJudge(
        lambda p: "WINNER: B",
        win_pattern=r"WINNER: ([AB])",
    )
    scores = j.score([("q", "a", "b")])
    assert scores == [0.0]


def test_invariant_pairwise_item_sliced_at_3():
    """score uses item[:3] so extra fields in a tuple are silently ignored."""
    j = PairwiseLLMJudge(lambda p: "Response A")
    scores = j.score([("q", "a", "b", "extra_field")])
    assert scores == [1.0]


# ---------------------------------------------------------------------------
# PairwiseLLMJudge — registry
# ---------------------------------------------------------------------------


def test_invariant_pairwise_llm_judge_registered(clean_registry):
    """PairwiseLLMJudge is registered under category='judge', name='pairwise_llm'."""
    from lighttrain.registry import get_registry
    reg = get_registry()
    cls = reg.get("judge", "pairwise_llm")
    assert cls is PairwiseLLMJudge
