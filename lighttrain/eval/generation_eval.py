"""Generation-based evaluation using a judge.

:class:`GenerationEvalTask` wraps model.generate() + a judge to produce
scored responses, and writes evaluation results as ``evaluated_by`` edges
in the lineage graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class GenerationEvalResult:
    """Per-sample generation eval result."""

    prompt: str
    response: str
    score: float
    extras: dict[str, Any] = field(default_factory=dict)


class GenerationEvalTask:
    """Evaluate a model by generating responses and scoring them with a judge.

    Parameters
    ----------
    judge :
        Any object with ``.score(items, ctx) -> list[float]``.
    tokenizer :
        HF-compatible tokenizer for encoding prompts and decoding responses.
    prompts :
        List of prompt strings to evaluate on.
    max_new_tokens : int
        Maximum tokens to generate per prompt.
    do_sample : bool
        Whether to sample (True) or greedy decode (False).
    extras_per_prompt :
        Optional list of extra dicts (one per prompt), forwarded to the judge.
    lineage_store :
        Optional :class:`~lighttrain.lineage.LineageStore`; if provided, writes
        an ``evaluated_by`` edge for the artifact/checkpoint being evaluated.
    artifact_id :
        ID of the artifact/checkpoint being evaluated (for lineage).
    """

    def __init__(
        self,
        judge: Any,
        tokenizer: Any,
        prompts: list[str],
        *,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        extras_per_prompt: list[dict[str, Any]] | None = None,
        lineage_store: Any | None = None,
        artifact_id: str | None = None,
        name: str = "generation_eval",
    ) -> None:
        self.judge = judge
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.extras_per_prompt = extras_per_prompt or [{} for _ in prompts]
        self.lineage_store = lineage_store
        self.artifact_id = artifact_id
        self.name = str(name)

    def run(
        self,
        model: Any,
        *,
        device: torch.device | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        """Run generation + judge scoring.

        Returns
        -------
        dict with keys: ``mean_score``, ``results`` (list of GenerationEvalResult),
        ``task_name``.
        """
        results: list[GenerationEvalResult] = []
        model.eval()

        with torch.no_grad():
            for prompt, extras in zip(self.prompts, self.extras_per_prompt, strict=False):
                enc = self.tokenizer(prompt, return_tensors="pt")
                input_ids = enc["input_ids"]
                if device is not None:
                    input_ids = input_ids.to(device)

                gen_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                )
                # Decode only the new tokens.
                response_ids = gen_ids[:, input_ids.size(1):]
                response = self.tokenizer.decode(
                    response_ids[0].tolist(), skip_special_tokens=True
                )

                item = (prompt, response, extras) if extras else (prompt, response)
                score_list = self.judge.score([item])
                score = float(score_list[0])

                results.append(
                    GenerationEvalResult(prompt=prompt, response=response, score=score, extras=extras)
                )

        mean_score = sum(r.score for r in results) / max(1, len(results))

        # Write lineage evaluated_by edge if lineage_store is available.
        if self.lineage_store is not None and self.artifact_id is not None:
            try:
                self._write_lineage_edge(mean_score, step)
            except Exception:  # noqa: BLE001
                pass

        return {
            "task_name": self.name,
            "mean_score": mean_score,
            "results": results,
        }


    def _write_lineage_edge(self, mean_score: float, step: int | None) -> None:
        """Write an ``evaluated_by`` edge to the lineage store.

        ``artifact_id`` may be:
        - An ``int`` node ID (direct use).
        - A ``"kind:name:version"`` ref string (resolved via
          :meth:`~lighttrain.lineage.store.LineageStore.resolve_ref`).
        - A plain artifact name (resolved as ``artifact:<name>:latest``).

        The evaluation result node is upserted as a ``"run"`` node keyed by
        ``(eval:<task_name>, step=<step>)``.
        """
        store = self.lineage_store

        # Resolve source node ID
        if isinstance(self.artifact_id, int):
            src_id = self.artifact_id
        else:
            ref = str(self.artifact_id)
            if ":" not in ref:
                ref = f"artifact:{ref}:latest"
            src_id = store.resolve_ref(ref)
            if src_id is None:
                return  # source node not found — skip silently

        # Upsert a minimal eval-result node as the destination.
        import time as _time
        dst_id = store.upsert_node(
            kind="run",
            name=f"eval:{self.name}",
            version=f"step_{step}" if step is not None else "latest",
            payload={
                "mean_score": mean_score,
                "task": self.name,
                "step": step,
                "ts": _time.time(),
            },
        )
        store.add_edge(
            src=int(src_id),
            dst=int(dst_id),
            kind="evaluated_by",
            payload={"mean_score": mean_score, "step": step, "task": self.name},
        )


__all__ = ["GenerationEvalResult", "GenerationEvalTask"]
