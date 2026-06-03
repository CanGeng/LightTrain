"""Contract: the abstract core ``Trainer`` supplies **no** concrete default
objective; ``PretrainTrainer`` overrides it with next-token cross-entropy.

DESIGN §3.3 — core knows no specific loss; the CE default lives in the
builtin_plugins impl (``lighttrain.builtin_plugins.losses.core.CrossEntropyLoss``).

The companion contract — a ``requires_objective=True`` trainer (e.g. preference)
errors loudly instead of falling back to ``None`` — is covered by
``tests/test_gap_report_fixes.py::test_wire_requires_objective_raises``.
"""

from __future__ import annotations

from lighttrain.architectures.profile import LossOnlyObjective
from lighttrain.builtin_plugins.losses.core import CrossEntropyLoss
from lighttrain.trainers.base import Trainer
from lighttrain.builtin_plugins.trainers.pretrain import PretrainTrainer


def test_base_default_objective_is_none():
    # default_objective ignores self; object.__new__ skips the heavy __init__.
    inst = object.__new__(Trainer)
    assert inst.default_objective() is None


def test_pretrain_default_objective_is_cross_entropy():
    inst = object.__new__(PretrainTrainer)
    obj = inst.default_objective()
    assert isinstance(obj, LossOnlyObjective)
    assert isinstance(obj.loss_fn, CrossEntropyLoss)
    assert obj.loss_family == "next_token"
