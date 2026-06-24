"""RL generation backends — HF generate (default) / vLLM (opt-in).

vLLM integration is opt-in via the ``vllm/`` subpackage here
(``builtin_plugins.rl.backends.vllm``); import it to register.
"""

from ..rollout import HFGenerateBackend, RolloutEngine

__all__ = ["HFGenerateBackend", "RolloutEngine"]
