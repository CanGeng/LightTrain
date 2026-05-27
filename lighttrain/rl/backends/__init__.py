"""RL generation backends — HF generate (default) / vLLM (opt-in).

vLLM integration is opt-in via ``frontier_plugins/generation_backends/``.
"""

from ..rollout import HFGenerateBackend, RolloutEngine

__all__ = ["HFGenerateBackend", "RolloutEngine"]
