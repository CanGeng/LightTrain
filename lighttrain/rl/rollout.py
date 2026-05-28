"""RolloutEngine — collect on-policy episodes using HF generate.

The engine drives the model's generation, scores each response with a
reward function (judge), and packages results as :class:`~lighttrain.rl.buffers.Episode`
objects ready for the PPO/GRPO inner loop.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F

from ..registry import register
from .buffers import Episode
from .ref_policy import _sequence_log_probs


@register("rl_backend", "hf_generate")
class HFGenerateBackend:
    """HuggingFace generate() backend for rollout collection.

    Parameters
    ----------
    max_new_tokens : int
        Maximum tokens to generate per response.
    do_sample : bool
        Sample from the distribution (True) or greedy decode (False).
    temperature : float
        Sampling temperature (only used when ``do_sample=True``).
    top_p : float
        Nucleus sampling threshold (only used when ``do_sample=True``).
    num_return_sequences : int
        Number of response candidates per prompt (GRPO group size G).
    """

    def __init__(
        self,
        *,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        pad_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.num_return_sequences = int(num_return_sequences)
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def generate(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run model.generate() and return full sequences (prompt + response).

        Returns (B * num_return_sequences, T_full) token ids.
        """
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "num_return_sequences": self.num_return_sequences,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        if self.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = self.pad_token_id
        if self.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.eos_token_id
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            sequences = model.generate(input_ids=input_ids, **gen_kwargs)
        return sequences  # (B*G, T_prompt+T_response)


class RolloutEngine:
    """Collect on-policy rollouts using an RL backend.

    Parameters
    ----------
    backend : :class:`HFGenerateBackend` or compatible
        Generation backend.
    ignore_index : int
        Padding token index for ``labels`` construction.
    """

    def __init__(
        self,
        backend: HFGenerateBackend | Any,
        *,
        ignore_index: int = -100,
    ) -> None:
        self.backend = backend
        self.ignore_index = int(ignore_index)

    @torch.no_grad()
    def rollout(
        self,
        model: Any,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor | None,
        reward_fn: Callable[[torch.Tensor, torch.Tensor], list[float]],
        *,
        group_offset: int = 0,
    ) -> list[Episode]:
        """Run one rollout step.

        Parameters
        ----------
        model :
            The live policy model (in eval mode during collection).
        prompt_ids : (B, T_p)
            Tokenized prompt sequences.
        prompt_mask : (B, T_p) or None
            Attention mask for the prompts.
        reward_fn :
            Callable ``(prompt_ids, response_ids) -> list[float]`` of length
            ``B * G`` where G = ``backend.num_return_sequences``.
        group_offset : int
            Starting group id for this batch (used to assign GRPO group IDs).

        Returns
        -------
        List of :class:`Episode` objects, one per (prompt, response) pair.
        """
        G = self.backend.num_return_sequences
        B = prompt_ids.size(0)

        was_training = model.training
        model.eval()
        try:
            full_seqs = self.backend.generate(model, prompt_ids, prompt_mask)
        finally:
            if was_training:
                model.train()

        # full_seqs: (B*G, T_full)
        T_prompt = prompt_ids.size(1)
        episodes: list[Episode] = []

        for i in range(B * G):
            seq = full_seqs[i]                          # (T_full,)
            response = seq[T_prompt:]                   # (T_resp,)
            prompt_i = prompt_ids[i // G]              # (T_p,)

            # Construct labels: -100 for prompt positions, token id for response
            labels = torch.full_like(seq, self.ignore_index)
            labels[T_prompt:] = response

            # Build attention mask for full sequence
            attn = torch.ones_like(seq)

            # Compute log-probs under current policy (after generate returns)
            with torch.no_grad():
                out = model(input_ids=seq.unsqueeze(0))
                logits = (
                    out.outputs["logits"]
                    if hasattr(out, "outputs")
                    else out["logits"]
                )
                logits = logits.squeeze(0)              # (T_full, V)
                shift_logits = logits[:-1]              # (T_full-1, V)
                shift_labels = seq[1:]                  # (T_full-1,)
                per_token_lp = F.log_softmax(shift_logits, dim=-1)
                gathered = per_token_lp.gather(
                    1, shift_labels.unsqueeze(1)
                ).squeeze(1)                            # (T_full-1,)
                # Pad back to full length (first position has no log-prob)
                log_probs = torch.cat(
                    [torch.zeros(1, device=gathered.device), gathered]
                )                                       # (T_full,)

            group_id = group_offset + (i // G)
            episodes.append(
                Episode(
                    input_ids=seq.cpu(),
                    attention_mask=attn.cpu(),
                    labels=labels.cpu(),
                    reward=0.0,  # filled below by reward_fn
                    log_probs=log_probs.cpu(),
                    group_id=group_id,
                )
            )

        # Score responses
        prompt_ids_expanded = prompt_ids.repeat_interleave(G, dim=0)  # (B*G, T_p)
        responses = full_seqs[:, T_prompt:]                            # (B*G, T_r)
        rewards = reward_fn(prompt_ids_expanded, responses)

        for ep, r in zip(episodes, rewards):
            ep.reward = float(r)

        return episodes


__all__ = ["HFGenerateBackend", "RolloutEngine"]
