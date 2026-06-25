"""Coverage-gap tests for lighttrain.builtin_plugins.losses.core.

Pins the uncovered branches (lines 24, 26-29, 34, 145, 149) that the existing
test_core.py does not exercise:

* ``_logits`` with a plain ``Mapping`` (dict) input — success path (lines 26-28).
* ``_logits`` with a ``Mapping`` that has no ``"logits"`` key — TypeError (line 29).
* ``_logits`` with a ``ModelOutput`` that lacks ``"logits"`` — KeyError (line 24).
* ``_logits`` with a non-Mapping, non-ModelOutput object — TypeError (line 29).
* ``_labels`` with a batch that has no ``"labels"`` key — KeyError (line 34).
* ``CompositeLoss`` with an empty children list — ValueError (line 145).
* ``CompositeLoss`` child entry missing ``name`` AND ``_target_`` — ValueError (line 149).
* General edge cases for each loss class via dict-input ``model_output``.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from lighttrain.builtin_plugins.losses.core import (
    CompositeLoss,
    CrossEntropyLoss,
    MaskedLMLoss,
    ZLoss,
)
from lighttrain.protocols import LossContext, ModelOutput

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _ctx() -> LossContext:
    """Return a fresh LossContext (no shared state)."""
    return LossContext()


class _NotAMapping:
    """Object that is neither ModelOutput nor Mapping — triggers TypeError."""

    pass


# ---------------------------------------------------------------------------
# _logits() internal function — tested indirectly via loss __call__
# ---------------------------------------------------------------------------


class TestLogitsHelper:
    """Tests covering the _logits() helper via CrossEntropyLoss.__call__."""

    def test_invariant_dict_with_logits_accepted(self):
        """A plain dict (Mapping) with a 'logits' key is a valid model_output.

        Covers lines 26-28: isinstance(model_output, Mapping) → 'logits' in
        model_output → return model_output['logits'].
        """
        torch.manual_seed(0)
        B, T, V = 2, 4, 6
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        model_output = {"logits": logits}
        loss_val = CrossEntropyLoss()(model_output, {"labels": labels}, _ctx())["loss"]
        assert loss_val.isfinite(), "Loss from dict model_output must be finite."

    def test_invariant_dict_output_matches_model_output_wrapper(self):
        """Dict and ModelOutput wrappers over same logits yield identical losses.

        Confirms lines 26-28 (Mapping path) and lines 22-25 (ModelOutput path)
        are numerically consistent.
        """
        torch.manual_seed(1)
        B, T, V = 2, 4, 5
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        batch = {"labels": labels}
        ctx = _ctx()
        loss_dict = CrossEntropyLoss()({"logits": logits}, batch, ctx)["loss"]
        loss_mo = CrossEntropyLoss()(ModelOutput(outputs={"logits": logits}), batch, ctx)["loss"]
        torch.testing.assert_close(loss_dict, loss_mo, atol=1e-6, rtol=1e-6)

    def test_invariant_model_output_missing_logits_raises_key_error(self):
        """ModelOutput.outputs without 'logits' key must raise KeyError (line 24)."""
        mo = ModelOutput(outputs={})  # no 'logits' key
        with pytest.raises(KeyError, match="logits"):
            CrossEntropyLoss()(mo, {"labels": torch.tensor([0])}, _ctx())

    def test_invariant_mapping_without_logits_raises_type_error(self):
        """A Mapping lacking 'logits' must raise TypeError (line 29).

        The Mapping branch (line 26-27) exits without returning when the key
        is absent; execution falls through to the raise on line 29.
        """
        model_output = {"other_key": torch.randn(2, 4)}  # no 'logits'
        with pytest.raises(TypeError, match="logits"):
            CrossEntropyLoss()(model_output, {"labels": torch.tensor([0])}, _ctx())

    def test_invariant_non_mapping_non_model_output_raises_type_error(self):
        """Passing a non-Mapping, non-ModelOutput object raises TypeError (line 29)."""
        with pytest.raises(TypeError, match="logits"):
            CrossEntropyLoss()(_NotAMapping(), {"labels": torch.tensor([0])}, _ctx())


# ---------------------------------------------------------------------------
# _labels() internal function
# ---------------------------------------------------------------------------


class TestLabelsHelper:
    """Tests covering _labels() via CrossEntropyLoss.__call__."""

    def test_invariant_batch_missing_labels_raises_key_error(self):
        """Batch without 'labels' key must raise KeyError (line 34)."""
        torch.manual_seed(2)
        logits = torch.randn(2, 4, 5)
        mo = ModelOutput(outputs={"logits": logits})
        with pytest.raises(KeyError, match="labels"):
            CrossEntropyLoss()(mo, {}, _ctx())

    def test_invariant_batch_missing_labels_via_dict_model_output(self):
        """Same KeyError when model_output is a plain dict (both helpers exercised)."""
        logits = torch.randn(2, 4, 5)
        with pytest.raises(KeyError, match="labels"):
            CrossEntropyLoss()({"logits": logits}, {}, _ctx())


# ---------------------------------------------------------------------------
# CrossEntropyLoss — dict model_output paths
# ---------------------------------------------------------------------------


class TestCrossEntropyLossDictInput:
    """Verify CE loss is correct when model_output is a plain dict (Mapping)."""

    def test_pin_current_behavior_ce_dict_matches_f_cross_entropy(self):
        """CE via dict model_output matches F.cross_entropy (with shift).

        Pin: the Mapping branch (lines 26-28) produces the same result as the
        ModelOutput branch; both paths converge at _logits().
        """
        torch.manual_seed(3)
        B, T, V = 2, 4, 5
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        batch = {"labels": labels}
        actual = CrossEntropyLoss()({"logits": logits}, batch, _ctx())["loss"]
        expected = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, V),
            labels[:, 1:].reshape(-1).long(),
            ignore_index=-100,
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    def test_invariant_ce_dict_no_labels_raises(self):
        """CE with dict model_output and no labels raises KeyError."""
        with pytest.raises(KeyError, match="labels"):
            CrossEntropyLoss()({"logits": torch.randn(2, 4, 5)}, {}, _ctx())


# ---------------------------------------------------------------------------
# MaskedLMLoss — dict model_output path
# ---------------------------------------------------------------------------


class TestMaskedLMLossDictInput:
    """Verify MLM loss accepts a plain dict model_output."""

    def test_invariant_mlm_dict_matches_f_cross_entropy(self):
        """MLM via dict model_output matches F.cross_entropy (no shift)."""
        torch.manual_seed(4)
        B, T, V = 2, 3, 7
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        batch = {"labels": labels}
        actual = MaskedLMLoss()({"logits": logits}, batch, _ctx())["loss"]
        expected = F.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1).long(),
            ignore_index=-100,
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    def test_invariant_mlm_missing_logits_mapping_raises_type_error(self):
        """MaskedLMLoss with Mapping lacking 'logits' raises TypeError."""
        with pytest.raises(TypeError, match="logits"):
            MaskedLMLoss()({"no_logits": torch.randn(2)}, {"labels": torch.tensor([0])}, _ctx())

    def test_invariant_mlm_missing_labels_raises_key_error(self):
        """MaskedLMLoss with batch missing 'labels' raises KeyError."""
        with pytest.raises(KeyError, match="labels"):
            MaskedLMLoss()({"logits": torch.randn(2, 3, 4)}, {}, _ctx())


# ---------------------------------------------------------------------------
# ZLoss — dict model_output path
# ---------------------------------------------------------------------------


class TestZLossDictInput:
    """Verify ZLoss accepts a plain dict model_output."""

    def test_invariant_zloss_dict_matches_closed_form(self):
        """ZLoss via dict model_output matches logsumexp-squared formula."""
        V = 4
        logits = torch.zeros(1, 1, V)
        w = 1.0
        actual = ZLoss(weight=w)({"logits": logits}, {}, _ctx())["loss"]
        expected = torch.tensor(w * (math.log(V) ** 2))
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)

    def test_invariant_zloss_missing_logits_in_mapping_raises_type_error(self):
        """ZLoss with Mapping lacking 'logits' raises TypeError."""
        with pytest.raises(TypeError, match="logits"):
            ZLoss()({"nope": torch.randn(2, 3, 4)}, {}, _ctx())


# ---------------------------------------------------------------------------
# CompositeLoss — constructor error paths
# ---------------------------------------------------------------------------


class TestCompositeLossErrors:
    """Tests for CompositeLoss constructor error paths (lines 145, 149)."""

    def test_invariant_empty_children_raises_value_error(self):
        """CompositeLoss([]) must raise ValueError (line 145)."""
        with pytest.raises(ValueError, match="at least one child"):
            CompositeLoss(children=[])

    def test_invariant_child_missing_name_and_target_raises_value_error(self):
        """Child without 'name' or '_target_' must raise ValueError (line 149).

        The entry has other keys (e.g. 'weight') but neither 'name' nor '_target_'.
        """
        with pytest.raises(ValueError, match=r"name.*_target_|_target_.*name"):
            CompositeLoss(children=[{"weight": 1.0, "params": {}}])

    def test_invariant_child_with_weight_but_no_name_raises(self):
        """Another variant: entry has only 'weight' — raises ValueError."""
        with pytest.raises(ValueError):
            CompositeLoss(children=[{"weight": 0.5}])

    def test_invariant_child_with_target_is_accepted(self, clean_registry):
        """A child entry using '_target_' (instead of 'name') is valid (line 148 pass).

        This confirms the 'or _target_' branch does NOT raise on line 149.
        """
        B, T, V = 1, 3, 4
        logits = torch.zeros(B, T, V)
        mo = ModelOutput(outputs={"logits": logits})
        labels = torch.randint(0, V, (B, T))
        batch = {"labels": labels}
        composite = CompositeLoss(
            children=[
                {
                    "_target_": "lighttrain.builtin_plugins.losses.core.ZLoss",
                    "weight": 1.0,
                    "weight_param": 1.0,
                }
            ]
        )
        out = composite(mo, batch, _ctx())
        assert "loss" in out
        assert out["loss"].isfinite()

    def test_invariant_multiple_children_first_missing_name_raises(self):
        """First child lacking 'name'/'_target_' raises even if later children are valid."""
        with pytest.raises(ValueError):
            CompositeLoss(
                children=[
                    {"weight": 1.0},  # invalid — missing name/_target_
                    {"name": "z_loss", "weight": 1.0},
                ]
            )


# ---------------------------------------------------------------------------
# CompositeLoss — dict model_output path
# ---------------------------------------------------------------------------


class TestCompositeLossDictInput:
    """Verify CompositeLoss accepts dict (Mapping) as model_output."""

    def test_invariant_composite_dict_model_output(self, clean_registry):
        """CompositeLoss works when model_output is a plain dict."""
        torch.manual_seed(5)
        V = 4
        logits = torch.zeros(1, 1, V)
        composite = CompositeLoss(
            children=[{"name": "z_loss", "weight": 1.0, "params": {"weight": 1.0}}]
        )
        out = composite({"logits": logits}, {}, _ctx())
        assert "loss" in out
        expected = math.log(V) ** 2
        torch.testing.assert_close(out["loss"], torch.tensor(expected), atol=1e-5, rtol=1e-4)
