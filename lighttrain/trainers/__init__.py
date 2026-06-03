"""Trainers — the abstract ``Trainer`` base + shared plumbing.

Concrete per-paradigm trainers (pretrain / preference / ppo / grpo /
reward_model) are registered impls and live in
``lighttrain.builtin_plugins.trainers`` (DESIGN §3.3). The base class + the
shared ``_primitives`` / ``_utils`` helpers stay here as core framework.
"""

from __future__ import annotations

from .base import Trainer

__all__ = ["Trainer"]
