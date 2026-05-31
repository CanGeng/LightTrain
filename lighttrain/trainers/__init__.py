"""Trainers — base + per-paradigm subclasses."""

from __future__ import annotations

from ._preference_base import PreferenceTrainer
from .base import Trainer
from .grpo import GRPOTrainer
from .ppo import PPOTrainer
from .pretrain import PretrainTrainer
from .rm import RewardModelTrainer

__all__ = [
    "GRPOTrainer",
    "PPOTrainer",
    "PreferenceTrainer",
    "PretrainTrainer",
    "RewardModelTrainer",
    "Trainer",
]
