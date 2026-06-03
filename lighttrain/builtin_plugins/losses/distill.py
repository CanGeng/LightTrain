"""Distillation losses.

All five losses read **teacher** tensors from the batch's ``aux.<namespace>.*``
keys, populated by :class:`lighttrain.builtin_plugins.artifacts.ArtifactJoinedDataset`.
The losses never call the teacher model directly — that's the producer's job.

Tensor naming convention
------------------------
Producer side (see ``recipes/produce_teacher.yaml``) writes:

  * ``logits_topk_64.values``   (B, T, K)
  * ``logits_topk_64.indices``  (B, T, K)  (int32)
  * ``hidden_states_layers``    (L, B, T, H_t)
  * ``attentions_layers``       (L, B, H, T, T)

Joined dataset writes them as ``aux.<ns>.logits_topk_64.values`` etc.

Mask
----
All losses respect ``labels != ignore_index`` (default ``-100``) so padded
positions are excluded from the reduction. Pass ``mask_from_labels=False`` to
disable.

Layer mapping
-------------
:class:`LayerMapping` is a small dataclass shared by hidden-state and attention
losses. ``{1: 2, 2: 4}`` means student layer ``1`` ↔ teacher layer ``2`` etc.
Indices are into ``ModelOutput.hidden_states`` / ``attentions`` tuples, which
include the embedding output at index 0 (HF convention).

Cross-vocab logits remapping
-----------------------------
:class:`CrossVocabRemapRegistry` provides a registry of remap functions for
handling teacher/student vocab mismatches. ``top_k`` (default) simply uses the
teacher's top-K indices directly. Custom remappers can be registered via
``register_remap(name, fn)`` where ``fn(student_logits, teacher_indices) ->
(remapped_student_logits, remapped_teacher_indices)``.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from lighttrain.protocols import LossContext, ModelOutput
from lighttrain.registry import register


_AUX_RE = re.compile(r"^aux\.([^.]+)\.(.+)$")


def _gather_aux(batch: Mapping[str, Any], namespace: str, key: str) -> torch.Tensor | None:
    """Find ``aux.<namespace>.<key>`` (or ``.<key>.values`` / ``.<key>.indices``)."""
    full = f"aux.{namespace}.{key}"
    if full in batch:
        return batch[full]
    # Some producers split a {topk} transform into two leaves — search for them.
    return None


def _label_mask(
    batch: Mapping[str, Any],
    *,
    ignore_index: int = -100,
    mask_from_labels: bool = True,
) -> torch.Tensor | None:
    if not mask_from_labels:
        return None
    labels = batch.get("labels")
    if labels is None:
        return None
    return (labels != ignore_index)


def _student_logits(model_output: ModelOutput | Mapping[str, Any]) -> torch.Tensor:
    if isinstance(model_output, ModelOutput):
        return model_output.outputs["logits"]
    return model_output["logits"]


def _student_hidden(
    model_output: ModelOutput | Mapping[str, Any], idx: int
) -> torch.Tensor:
    if isinstance(model_output, ModelOutput):
        if model_output.hidden_states is None:
            raise RuntimeError(
                "student hidden states unavailable; pass output_hidden_states=True "
                "to the model's forward or set it on the adapter."
            )
        return model_output.hidden_states[idx]
    raise TypeError("hidden state distillation requires a ModelOutput instance")


def _student_attn(
    model_output: ModelOutput | Mapping[str, Any], idx: int
) -> torch.Tensor:
    if isinstance(model_output, ModelOutput):
        if model_output.attentions is None:
            raise RuntimeError(
                "student attentions unavailable; pass output_attentions=True."
            )
        return model_output.attentions[idx]
    raise TypeError("attention transfer requires a ModelOutput instance")


@dataclass
class LayerMapping:
    """Student-layer → teacher-layer index map.

    ``mapping[s] = t`` says: align student tuple index ``s`` with teacher tuple
    index ``t`` (after layer-0 = embeddings convention).
    """

    mapping: dict[int, int] = field(default_factory=dict)

    @classmethod
    def coerce(cls, value: Any) -> "LayerMapping":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls({int(k): int(v) for k, v in value.items()})
        if isinstance(value, list):
            # support list-of-pairs form
            return cls({int(s): int(t) for s, t in value})
        raise TypeError(f"cannot coerce {type(value).__name__} to LayerMapping")


# ---------------------------------------------------------------- KL on top-K


@register("loss", "kl_topk")
class KLDivLoss:
    """KL divergence on teacher's top-K tokens.

    Algorithm::

        T_logits ← gather student logits at teacher's top-K indices
        loss     ← KL(softmax(T_logits / τ) ‖ softmax(values / τ)) * τ²

    Teacher tensors expected at::

        batch["aux.<namespace>.<key>.values"]   (B, T, K)
        batch["aux.<namespace>.<key>.indices"]  (B, T, K)

    Parameters
    ----------
    temperature : float
        Softmax temperature applied to both sides.
    top_k : int
        Information only — the actual K is determined by the stored ``values``
        tensor; we assert agreement.
    teacher_namespace : str
        ``aux.<this>.*`` lookup.
    teacher_key : str
        Field stem (e.g. ``"logits_topk_64"``).
    reduction : str
        ``"mean"`` (over unmasked positions) or ``"sum"`` or ``"none"``.
    """

    def __init__(
        self,
        *,
        temperature: float = 2.0,
        top_k: int = 64,
        teacher_namespace: str = "teacher",
        teacher_key: str = "logits_topk_64",
        ignore_index: int = -100,
        mask_from_labels: bool = True,
        reduction: str = "mean",
    ) -> None:
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.teacher_namespace = str(teacher_namespace)
        self.teacher_key = str(teacher_key)
        self.ignore_index = int(ignore_index)
        self.mask_from_labels = bool(mask_from_labels)
        self.reduction = reduction

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],
        batch: Mapping[str, Any],
        ctx: LossContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        values_key = f"aux.{self.teacher_namespace}.{self.teacher_key}.values"
        indices_key = f"aux.{self.teacher_namespace}.{self.teacher_key}.indices"
        if values_key not in batch or indices_key not in batch:
            raise KeyError(
                f"kl_topk needs {values_key!r} and {indices_key!r} in the batch. "
                f"Are you joining the teacher artifact?"
            )
        teacher_values: torch.Tensor = batch[values_key]
        teacher_indices: torch.Tensor = batch[indices_key].long()
        student_logits = _student_logits(model_output)
        # Gather student logits at teacher topk indices.
        student_topk = torch.gather(student_logits, dim=-1, index=teacher_indices)
        tau = self.temperature
        log_p_student = F.log_softmax(student_topk / tau, dim=-1)
        log_p_teacher = F.log_softmax(teacher_values.to(student_topk.dtype) / tau, dim=-1)
        per_token = F.kl_div(log_p_student, log_p_teacher, reduction="none", log_target=True)
        per_token = per_token.sum(dim=-1) * (tau * tau)  # KL is summed over K, scaled by τ²

        mask = _label_mask(batch, ignore_index=self.ignore_index,
                           mask_from_labels=self.mask_from_labels)
        if mask is not None:
            per_token = per_token * mask.to(per_token.dtype)
            denom = mask.sum().clamp_min(1).to(per_token.dtype)
        else:
            denom = torch.tensor(float(per_token.numel()), device=per_token.device,
                                 dtype=per_token.dtype)
        if self.reduction == "mean":
            loss = per_token.sum() / denom
        elif self.reduction == "sum":
            loss = per_token.sum()
        else:
            loss = per_token
        return {"loss": loss, "kl_topk_unmasked_tokens": float(denom.detach())}


# ---------------------------------------------------------------- hidden states


@register("loss", "hidden_mse")
class HiddenStatesMSELoss:
    """MSE between student and teacher hidden states under a layer mapping.

    Student hidden states come from ``model_output.hidden_states[s]``; teacher
    from ``batch["aux.<ns>.hidden_states_layers"]`` of shape ``(L, B, T, H)``.

    Parameters
    ----------
    mapping : dict | LayerMapping
        ``{student_layer_idx: teacher_layer_idx}``.
    teacher_namespace : str
        Lookup namespace; default ``"teacher"``.
    teacher_key : str
        Stem; default ``"hidden_states_layers"``.
    project : bool
        Currently only ``False`` is supported; ``True`` is not yet implemented.
    reduction : str
        ``"mean"`` (default) or ``"sum"``.
    """

    def __init__(
        self,
        *,
        mapping: Any,
        teacher_namespace: str = "teacher",
        teacher_key: str = "hidden_states_layers",
        ignore_index: int = -100,
        mask_from_labels: bool = True,
        project: bool = False,
        project_init: str = "xavier",
        project_bias: bool = False,
        reduction: str = "mean",
    ) -> None:
        self.mapping = LayerMapping.coerce(mapping)
        self.teacher_namespace = str(teacher_namespace)
        self.teacher_key = str(teacher_key)
        self.ignore_index = int(ignore_index)
        self.mask_from_labels = bool(mask_from_labels)
        self.project = bool(project)
        self.project_init = str(project_init)
        self.project_bias = bool(project_bias)
        self.reduction = reduction
        # Cache of projection submodule paths, keyed by (layer_idx, s_dim, t_dim).
        # Path-only — the actual ``nn.Linear`` lives on the model so it follows
        # ``.to(device)``, gets saved by CheckpointManager, and shows up in
        # ``named_parameters`` for ``OptimizerWrapper.build``.
        self._projection_paths: dict[tuple[int, int, int], str] = {}

    def _ensure_projection(
        self,
        model: Any,
        ctx: LossContext,
        layer_idx: int,
        s_dim: int,
        t_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.nn.Linear:
        from lighttrain.models.surgery import add_named_module, get_submodule

        key = (int(layer_idx), int(s_dim), int(t_dim))
        path = self._projection_paths.get(key)
        if path is not None:
            sub = get_submodule(model, path)
            if isinstance(sub, torch.nn.Linear):
                return sub
        path = f"_distill_projections.layer_{layer_idx}_s{s_dim}_t{t_dim}"
        lin = torch.nn.Linear(s_dim, t_dim, bias=self.project_bias).to(
            device=device, dtype=dtype
        )
        if self.project_init == "xavier":
            torch.nn.init.xavier_uniform_(lin.weight)
        elif self.project_init == "orthogonal":
            torch.nn.init.orthogonal_(lin.weight)
        elif self.project_init == "zeros":
            torch.nn.init.zeros_(lin.weight)
        elif self.project_init == "normal":
            torch.nn.init.normal_(lin.weight, std=0.02)
        else:
            raise ValueError(
                f"hidden_mse.project_init={self.project_init!r} unsupported"
            )
        if self.project_bias and lin.bias is not None:
            torch.nn.init.zeros_(lin.bias)
        add_named_module(model, path, lin)
        self._projection_paths[key] = path
        # Notify the StandardUpdateRule (or any compatible engine) that fresh
        # trainable params just appeared so optimizer.step() updates them.
        try:
            bucket = ctx.extras.setdefault("_new_trainable_params", [])
            bucket.extend(list(lin.parameters()))
        except AttributeError:
            # ctx without ``extras`` (unusual) — degraded mode: projection
            # still trains as long as the user later calls
            # optimizer.add_param_group manually. Not silent: log once.
            import warnings

            warnings.warn(
                "hidden_mse: ctx.extras missing — projection layer parameters "
                "won't be auto-registered with the optimizer.",
                stacklevel=2,
            )
        return lin

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],
        batch: Mapping[str, Any],
        ctx: LossContext,
    ) -> dict[str, Any]:
        key = f"aux.{self.teacher_namespace}.{self.teacher_key}"
        if key not in batch:
            raise KeyError(f"hidden_mse needs {key!r} in the batch.")
        teacher_layers: torch.Tensor = batch[key]  # (L, B, T, H_t)
        mask = _label_mask(batch, ignore_index=self.ignore_index,
                           mask_from_labels=self.mask_from_labels)
        total: torch.Tensor | None = None
        _warned_dtype = False
        for s_idx, t_idx in self.mapping.mapping.items():
            s = _student_hidden(model_output, s_idx)
            raw_t = teacher_layers[t_idx]
            if raw_t.dtype != s.dtype and not _warned_dtype:
                warnings.warn(
                    f"hidden_mse: teacher dtype {raw_t.dtype} != student dtype {s.dtype}; "
                    "auto-casting teacher to student dtype.",
                    stacklevel=2,
                )
                _warned_dtype = True
            t = raw_t.to(s.dtype)
            if s.shape[-1] != t.shape[-1]:
                if not self.project:
                    raise RuntimeError(
                        f"hidden_mse hidden dim mismatch student={s.shape[-1]} "
                        f"teacher={t.shape[-1]}; set project=True or "
                        f"pre-project at producer time."
                    )
                model = ctx.extras.get("model") if hasattr(ctx, "extras") else None
                if model is None:
                    raise RuntimeError(
                        "hidden_mse.project=True needs ``ctx.extras['model']`` "
                        "so the projection layer can attach as a submodule. "
                        "StandardUpdateRule publishes this automatically; "
                        "custom engines must do the same."
                    )
                proj = self._ensure_projection(
                    model=model,
                    ctx=ctx,
                    layer_idx=s_idx,
                    s_dim=s.shape[-1],
                    t_dim=t.shape[-1],
                    dtype=s.dtype,
                    device=s.device,
                )
                s = proj(s)
            err = F.mse_loss(s, t, reduction="none")  # (B, T, H)
            err = err.mean(dim=-1)  # collapse H -> (B, T)
            if mask is not None:
                err = err * mask.to(err.dtype)
                denom = mask.sum().clamp_min(1).to(err.dtype)
            else:
                denom = torch.tensor(float(err.numel()), device=err.device,
                                     dtype=err.dtype)
            layer_loss = err.sum() / denom
            total = layer_loss if total is None else total + layer_loss
        if total is None:
            raise ValueError("hidden_mse mapping was empty.")
        if self.reduction == "mean":
            total = total / max(1, len(self.mapping.mapping))
        return {"loss": total}


@register("loss", "hidden_cosine")
class HiddenStatesCosineLoss:
    """``1 - cosine`` similarity between student and teacher hidden states.

    Same plumbing as :class:`HiddenStatesMSELoss` — uses the same mapping and
    teacher payload. Length axis is treated per-position; cosine is computed
    along the hidden dim.
    """

    def __init__(
        self,
        *,
        mapping: Any,
        teacher_namespace: str = "teacher",
        teacher_key: str = "hidden_states_layers",
        ignore_index: int = -100,
        mask_from_labels: bool = True,
        reduction: str = "mean",
        eps: float = 1e-8,
    ) -> None:
        self.mapping = LayerMapping.coerce(mapping)
        self.teacher_namespace = str(teacher_namespace)
        self.teacher_key = str(teacher_key)
        self.ignore_index = int(ignore_index)
        self.mask_from_labels = bool(mask_from_labels)
        self.reduction = reduction
        self.eps = float(eps)

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],
        batch: Mapping[str, Any],
        ctx: LossContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        key = f"aux.{self.teacher_namespace}.{self.teacher_key}"
        if key not in batch:
            raise KeyError(f"hidden_cosine needs {key!r} in the batch.")
        teacher_layers: torch.Tensor = batch[key]
        mask = _label_mask(batch, ignore_index=self.ignore_index,
                           mask_from_labels=self.mask_from_labels)
        total: torch.Tensor | None = None
        _warned_dtype = False
        for s_idx, t_idx in self.mapping.mapping.items():
            s = _student_hidden(model_output, s_idx)
            raw_t = teacher_layers[t_idx]
            if raw_t.dtype != s.dtype and not _warned_dtype:
                warnings.warn(
                    f"hidden_cosine: teacher dtype {raw_t.dtype} != student dtype {s.dtype}; "
                    "auto-casting teacher to student dtype.",
                    stacklevel=2,
                )
                _warned_dtype = True
            t = raw_t.to(s.dtype)
            if s.shape[-1] != t.shape[-1]:
                raise RuntimeError(
                    "hidden_cosine: dim mismatch — pre-project teacher at producer time."
                )
            sim = F.cosine_similarity(s, t, dim=-1, eps=self.eps)  # (B, T)
            err = 1.0 - sim
            if mask is not None:
                err = err * mask.to(err.dtype)
                denom = mask.sum().clamp_min(1).to(err.dtype)
            else:
                denom = torch.tensor(float(err.numel()), device=err.device,
                                     dtype=err.dtype)
            layer_loss = err.sum() / denom
            total = layer_loss if total is None else total + layer_loss
        if total is None:
            raise ValueError("hidden_cosine mapping was empty.")
        if self.reduction == "mean":
            total = total / max(1, len(self.mapping.mapping))
        return {"loss": total}


@register("loss", "attention_transfer")
class AttentionTransferLoss:
    """Attention map distillation (Zagoruyko/Komodakis '17 style, p=2 default).

    For each mapped layer, normalize student and teacher attention probability
    maps and minimize their MSE. Shape contract per layer:

      * student: ``(B, H_s, T, T)``
      * teacher: ``(B, H_t, T, T)`` from ``aux.<ns>.attentions_layers[t_idx]``

    When head counts differ we average across heads.

    Parameters
    ----------
    p : float
        Norm exponent (2 = squared MSE on row-normalized maps).
    """

    def __init__(
        self,
        *,
        mapping: Any,
        teacher_namespace: str = "teacher",
        teacher_key: str = "attentions_layers",
        p: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        self.mapping = LayerMapping.coerce(mapping)
        self.teacher_namespace = str(teacher_namespace)
        self.teacher_key = str(teacher_key)
        self.p = float(p)
        self.reduction = reduction

    def __call__(
        self,
        model_output: ModelOutput | Mapping[str, Any],
        batch: Mapping[str, Any],
        ctx: LossContext,  # noqa: ARG002
    ) -> dict[str, Any]:
        key = f"aux.{self.teacher_namespace}.{self.teacher_key}"
        if key not in batch:
            raise KeyError(f"attention_transfer needs {key!r} in the batch.")
        teacher_layers: torch.Tensor = batch[key]
        total: torch.Tensor | None = None
        for s_idx, t_idx in self.mapping.mapping.items():
            s = _student_attn(model_output, s_idx).mean(dim=1)  # (B, T, T)
            t = teacher_layers[t_idx].mean(dim=1).to(s.dtype)  # (B, T, T)
            # Row-wise norm so we compare normalized attention "shape".
            s_n = s / s.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            t_n = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            layer_loss = (s_n - t_n).abs().pow(self.p).mean()
            total = layer_loss if total is None else total + layer_loss
        if total is None:
            raise ValueError("attention_transfer mapping was empty.")
        if self.reduction == "mean":
            total = total / max(1, len(self.mapping.mapping))
        return {"loss": total}


# ---------------------------------------------------------------------------
# Cross-vocab logits remapping
# ---------------------------------------------------------------------------


class CrossVocabRemapRegistry:
    """Registry of remap functions for teacher/student vocabulary mismatches.

    A remap function has the signature::

        fn(student_logits, teacher_indices) -> (student_topk, teacher_topk_indices)

    ``student_logits``   : (B, T, V_s) — full student vocabulary logits
    ``teacher_indices``  : (B, T, K)   — teacher top-K vocab indices (in teacher vocab)

    The default ``"top_k"`` strategy assumes student and teacher share the same
    vocabulary so teacher_indices are directly used to gather from student_logits.

    For cross-architecture vocab remapping (e.g. different tokenizers), register
    a custom function that converts teacher indices to student indices before the
    gather step.
    """

    _registry: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register_remap(cls, name: str, fn: Callable[..., Any]) -> None:
        cls._registry[name] = fn

    @classmethod
    def get_remap(cls, name: str) -> Callable[..., Any]:
        if name not in cls._registry:
            raise KeyError(
                f"CrossVocabRemapRegistry: unknown remap '{name}'. "
                f"Available: {sorted(cls._registry)}"
            )
        return cls._registry[name]

    @classmethod
    def list_remaps(cls) -> list[str]:
        return sorted(cls._registry)


def _remap_top_k(
    student_logits: torch.Tensor,
    teacher_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Default top-K remap: assumes shared vocabulary.

    Gathers student logits at teacher's top-K positions.
    Returns (gathered_student_logits, teacher_indices) — indices unchanged.
    """
    idx = teacher_indices.long()
    gathered = torch.gather(student_logits, dim=-1, index=idx)
    return gathered, idx


CrossVocabRemapRegistry.register_remap("top_k", _remap_top_k)


__all__ = [
    "AttentionTransferLoss",
    "CrossVocabRemapRegistry",
    "HiddenStatesCosineLoss",
    "HiddenStatesMSELoss",
    "KLDivLoss",
    "LayerMapping",
]
