"""VLLMBackend — vLLM rollout backend stub.

Registers ``@register("rl_backend", "vllm")`` as an opt-in generation backend.
Importing this module without vLLM installed will raise ``ImportError`` at
backend *construction* time (not at import time), so recipes can reference it
without breaking import-only tests.

Usage::

    # In recipe YAML:
    rollout_backend:
      name: vllm
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.9
      max_model_len: 2048

    # In Python (requires pip install vllm):
    from plugins.generation_backends.vllm import VLLMBackend
    backend = VLLMBackend(model_name_or_path="gpt2")

Interface matches ``HFGenerateBackend``::

    backend.generate(model, input_ids, attention_mask) -> Tensor (B*G, T)

Note: vLLM uses its own engine; the ``model`` argument is ignored at runtime —
the backend manages its own LLM instance initialised from ``model_name_or_path``.
"""

from __future__ import annotations

from typing import Any

import torch

from lighttrain.registry import register


@register("rl_backend", "vllm")
class VLLMBackend:
    """vLLM generation backend for high-throughput PPO/GRPO rollouts.

    This is a stub that raises ``ImportError`` unless ``vllm`` is installed.
    """

    def __init__(
        self,
        *,
        model_name_or_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 2048,
        max_new_tokens: int = 128,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "VLLMBackend requires 'vllm'. "
                "Install it with: pip install vllm"
            ) from exc

        from vllm import LLM, SamplingParams

        self.max_new_tokens = int(max_new_tokens)
        self.num_return_sequences = int(num_return_sequences)
        self._sampling_params = SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_new_tokens),
            n=int(num_return_sequences),
        )
        self._llm = LLM(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

    def generate(
        self,
        model: Any,  # ignored; vLLM uses its own engine
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run vLLM generation; returns (B * num_return_sequences, T_full)."""
        # Convert input_ids to token-id lists for vLLM
        prompts = input_ids.tolist()
        outputs = self._llm.generate(
            prompt_token_ids=prompts,
            sampling_params=self._sampling_params,
        )
        # Flatten: each prompt has num_return_sequences completions
        all_ids: list[list[int]] = []
        max_len = 0
        for out in outputs:
            for completion in out.outputs:
                ids = list(out.prompt_token_ids) + list(completion.token_ids)
                all_ids.append(ids)
                max_len = max(max_len, len(ids))

        # Pad to same length
        pad_id = 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in all_ids]
        return torch.tensor(padded, dtype=torch.long)


__all__ = ["VLLMBackend"]
