"""Trainers — base + per-paradigm subclasses."""

from __future__ import annotations

from .base import Trainer
from .dpo import DPOTrainer
from .grpo import GRPOTrainer
from .ipo import IPOTrainer
from .kto import KTOTrainer
from .orpo import ORPOTrainer
from .ppo import PPOTrainer
from .pretrain import PretrainTrainer
from .rm import RewardModelTrainer
from .simpo import SimPOTrainer

__all__ = [
    "DPOTrainer",
    "GRPOTrainer",
    "IPOTrainer",
    "KTOTrainer",
    "ORPOTrainer",
    "PPOTrainer",
    "PretrainTrainer",
    "RewardModelTrainer",
    "SimPOTrainer",
    "Trainer",
]
