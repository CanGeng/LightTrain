"""RL — RolloutEngine / RolloutBuffer / GAE / ReferencePolicy."""

from .buffers import Episode, RolloutBuffer
from .gae import compute_gae, normalize_advantages
from .ref_policy import ReferencePolicy, freeze_as_ref, ref_log_probs
from .rollout import HFGenerateBackend, RolloutEngine

__all__ = [
    "Episode",
    "HFGenerateBackend",
    "ReferencePolicy",
    "RolloutBuffer",
    "RolloutEngine",
    "compute_gae",
    "freeze_as_ref",
    "normalize_advantages",
    "ref_log_probs",
]
