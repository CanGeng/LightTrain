"""Component protocols.

Every registry category has a corresponding ``Protocol`` here. Implementations
need only structural conformity (``runtime_checkable``) — no inheritance
required. Contracts are defined here so config validation, registry typing,
and Trainer dispatch are testable in isolation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

# Distributed strategy protocols — imported here so that core components can
# type-check against them without depending on builtin_plugins.
from .distributed._protocols import (
    GradSyncStrategy as GradSyncStrategyProtocol,
)
from .distributed._protocols import (
    ModelParallelStrategy as ModelParallelStrategyProtocol,
)
from .distributed._protocols import (
    PipelineSchedule as PipelineScheduleProtocol,
)

# ---------------------------------------------------------------------------
# Generic data carriers
# ---------------------------------------------------------------------------


@dataclass
class ModelOutput:
    """Standardized model output.

    ``outputs`` carries arbitrarily named tensors (logits / eps / recon / ...);
    Trainer and LossFn consume by **key**, not by fixed attribute name. Any
    additional named extracts (per ExtraOutputSpec) are merged into ``extras``.
    """

    outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    loss: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
    state: Any | None = None  # for stateful architectures (RWKV / Mamba)


@dataclass
class LossContext:
    """Context passed to LossFn."""

    step: int = 0
    epoch: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    loss_family: str | None = None  # next_token / mlm / denoising / flow / jepa / rl / ...
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepOutput:
    """Typed return value from Trainer._step() and train_step().

    ``loss`` holds the primary optimization objective — may be a Python float,
    a detached scalar tensor, or None (e.g. RL trainers that aggregate across
    inner epochs before exposing a primary loss).

    ``metrics`` is the full metrics dict as returned by the underlying step
    method, including the ``"loss"`` key when present. Kept intact so that
    existing logging and checkpoint machinery can consume it unchanged.

    ``logs`` and ``extras`` are reserved for future hook/callback extensions.
    """

    loss: Any | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core 8 + checkpoint/logger
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelProtocol(Protocol):
    def forward(self, **batch: Any) -> ModelOutput: ...


@runtime_checkable
class GenerativeModelProtocol(ModelProtocol, Protocol):
    """Protocol for models used in RL rollouts (GRPO / PPO).

    Extends ModelProtocol with a generate() method compatible with the
    HFGenerateBackend signature. Required by GRPOTrainer / PPOTrainer.
    """

    def generate(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor: ...


@runtime_checkable
class LossFnProtocol(Protocol):
    def __call__(
        self,
        model_output: ModelOutput,
        batch: Mapping[str, Any],
        ctx: LossContext,
    ) -> dict[str, Any]: ...


@runtime_checkable
class OptimizerWrapperProtocol(Protocol):
    """Wrapper around a ``torch.optim.Optimizer``.

    **Full contract** (the update rule + checkpoint manager call all of these
    on the *wrapper*, not the inner optimizer — implement them or subclass
    ``lighttrain.optim.base.OptimizerWrapperBase`` which supplies them):

    * ``optimizer`` — the inner ``torch.optim.Optimizer``, set by ``build()``.
      Must expose ``.param_groups``: the LR logger reads
      ``optimizer.optimizer.param_groups[0]["lr"]``
      (``update_rules/standard.py``).
    * ``build(model)`` — construct and return the inner optimizer (once).
    * ``step()`` / ``zero_grad()`` — called each step. Canonical loop ordering
      is ``clip_grad_norm_`` → ``optimizer.step()`` → ``zero_grad()``; grads
      are **full-rank and intact** at ``step()`` time (no closure / pre-step
      grad mutation by the framework).
    * ``state_dict()`` / ``load_state_dict()`` — called by the checkpoint
      manager via ``torch.save(..., weights_only=False)``. Custom (non-tensor)
      optimizer state round-trips as long as it is picklable **and** importable
      at load time. For a *portable* checkpoint, serialize custom state as
      plain tensors so it loads without the optimizer's own package on path.

    **Optional** methods (called when present, else a sane default is used):

    * ``optim_state_bytes(model)`` — return the optimizer's real per-step state
      footprint in bytes. ``lab.estimate`` calls this when present, else falls
      back to ``2 × trainable_param_bytes`` (the full-rank Adam assumption).
      Memory-efficient optimizers (GaLore, 8-bit Adam, Adam-mini) override it
      so ``estimate`` can *see* their saving.
    """

    optimizer: torch.optim.Optimizer

    def build(self, model: Any) -> torch.optim.Optimizer: ...
    def step(self, *args: Any, **kwargs: Any) -> Any: ...
    def zero_grad(self, set_to_none: bool = True) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, sd: Mapping[str, Any]) -> None: ...

    # NOTE: ``optim_state_bytes(model) -> int`` is an *optional* hook (see the
    # class docstring). It is deliberately NOT declared here so it stays out
    # of the ``runtime_checkable`` required surface; ``estimate`` discovers it
    # via ``getattr`` and falls back to ``2 × params`` when absent.


@runtime_checkable
class SchedulerProtocol(Protocol):
    step_per_batch: bool

    def step(self, *args: Any, **kwargs: Any) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, sd: Mapping[str, Any]) -> None: ...


@runtime_checkable
class DataModuleProtocol(Protocol):
    def train_loader(self) -> Iterable[Any]: ...
    def val_loader(self) -> Iterable[Any] | None: ...
    def predict_loader(self) -> Iterable[Any] | None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, sd: Mapping[str, Any]) -> None: ...


@runtime_checkable
class TokenizerProtocol(Protocol):
    def encode(self, text: str, **kwargs: Any) -> list[int]: ...
    def decode(self, ids: list[int], **kwargs: Any) -> str: ...


@runtime_checkable
class CollatorProtocol(Protocol):
    def __call__(self, samples: list[Mapping[str, Any]]) -> dict[str, Any]: ...


@runtime_checkable
class SamplerProtocol(Protocol):
    def __iter__(self) -> Iterable[int]: ...
    def __len__(self) -> int: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, sd: Mapping[str, Any]) -> None: ...


@runtime_checkable
class ProcessorProtocol(Protocol):
    """Multimodal processor (text/image/audio/video)."""

    modality: str

    def __call__(self, inputs: Any, **kwargs: Any) -> Mapping[str, Any]: ...


@runtime_checkable
class MetricProtocol(Protocol):
    def update(self, *args: Any, **kwargs: Any) -> None: ...
    def compute(self) -> Any: ...
    def reset(self) -> None: ...


@runtime_checkable
class LoggerProtocol(Protocol):
    def log_scalars(self, scalars: Mapping[str, float], step: int) -> None: ...
    def log_histograms(self, hists: Mapping[str, Any], step: int) -> None: ...
    def log_text(self, text: str, step: int) -> None: ...
    def log_artifact(self, path: str, name: str | None = None) -> None: ...
    def flush(self) -> None: ...


@runtime_checkable
class CheckpointManagerProtocol(Protocol):
    def save(
        self,
        step: int,
        state: Mapping[str, Any],
        *,
        kind: str = "step",
        extras: Mapping[str, Any] | None = None,
    ) -> Path: ...
    def load(self, path: str | Path) -> dict[str, Any]: ...
    def latest(self) -> Path | None: ...
    def best(self) -> Path | None: ...


# ---------------------------------------------------------------------------
# Engine / UpdateRule / Callback / Trainer
# ---------------------------------------------------------------------------


@runtime_checkable
class EngineProtocol(Protocol):
    def step(self, batch: Mapping[str, Any], ctx: Any) -> dict[str, Any]: ...


@runtime_checkable
class UpdateRuleProtocol(Protocol):
    def setup(self, model: Any, sample: Any) -> None: ...
    def step(self, model: Any, batch: Mapping[str, Any], ctx: Any) -> dict[str, Any]: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, sd: Mapping[str, Any]) -> None: ...


CALLBACK_EVENTS: tuple[str, ...] = (
    # lifecycle
    "on_init_end",
    "on_train_start",
    "on_train_end",
    "on_epoch_begin",
    "on_epoch_end",
    "on_train_batch_start",
    "on_train_batch_end",
    # step internals
    "on_step_begin",
    "on_forward_pre",
    "on_forward_post",
    "on_loss_computed",
    "on_backward_pre",
    "on_backward_post",
    "on_clip_grad",
    "on_optimizer_step_pre",
    "on_optimizer_step_post",
    "on_scheduler_step",
    "on_zero_grad",
    "on_step_end",
    # eval
    "on_eval_begin",
    "on_eval_batch_start",
    "on_eval_batch_end",
    "on_eval_end",
    # persistence
    "on_save_checkpoint_pre",
    "on_save_checkpoint_post",
    "on_load_checkpoint_pre",
    "on_load_checkpoint_post",
    # exceptions
    "on_exception",
    "on_nan_detected",
    "on_oom",
    # RL
    "on_rollout_begin",
    "on_rollout_end",
    "on_reward_computed",
    "on_kl_computed",
    # lineage / artifact
    "on_artifact_finalized",
    "on_artifact_new_version",
    # distributed
    "on_distributed_init",
    "on_pipeline_schedule_begin",
    "on_pipeline_schedule_end",
    "on_microbatch_forward_pre",
    "on_microbatch_forward_post",
    "on_microbatch_backward_pre",
    "on_microbatch_backward_post",
    "on_pipeline_bubble",
    "on_rank_sync_pre",
    "on_rank_sync_post",
)


@runtime_checkable
class CallbackProtocol(Protocol):
    """Single callback Protocol — methods are optional; the EventBus checks via getattr.

    Callback signals (returnable): ``SKIP_STEP / STOP_TRAINING / RETRY_STEP``.
    """


@runtime_checkable
class TrainerProtocol(Protocol):
    def fit(self, *, steps: int | None = None) -> Any: ...
    def eval(self, *args: Any, **kwargs: Any) -> Any: ...
    def predict(self, *args: Any, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Frontier extension categories
# ---------------------------------------------------------------------------


@runtime_checkable
class GenerationStrategyProtocol(Protocol):
    def generate(
        self,
        model: Any,
        prompts: Any,
        sampling: Mapping[str, Any],
        scorer: Any | None = None,
        ctx: Any | None = None,
    ) -> Any: ...


@runtime_checkable
class JudgeProtocol(Protocol):
    def score(self, items: Iterable[Any], ctx: Any | None = None) -> list[Any]: ...


@runtime_checkable
class EnvironmentProtocol(Protocol):
    def reset(self, ctx: Any | None = None) -> Any: ...
    def step(self, action: Any) -> Any: ...


@runtime_checkable
class RetrieverProtocol(Protocol):
    def index(self, corpus: Any, ctx: Any | None = None) -> Any: ...
    def query(self, queries: Any, k: int, ctx: Any | None = None) -> Any: ...


@runtime_checkable
class ChunkerProtocol(Protocol):
    def chunk(self, rows: Iterable[Any], ctx: Any | None = None) -> Iterable[Any]: ...


@runtime_checkable
class ProbeProtocol(Protocol):
    def attach(self, model: Any, layers: Iterable[str], ctx: Any | None = None) -> Any: ...
    def compute(self, activations: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Architecture / Objective
# ---------------------------------------------------------------------------


@runtime_checkable
class ArchitectureProfileProtocol(Protocol):
    name: str
    forward_signature: str  # causal_lm / seq2seq / diffusion_eps / ...
    loss_family: str  # next_token / mlm / denoising / flow_matching / jepa / rl / ...

    def supports(self, capability: str) -> bool: ...


@runtime_checkable
class ObjectiveProtocol(Protocol):
    def corrupt(self, batch: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def forward_args(self, model: Any, corrupted_batch: Mapping[str, Any]) -> dict[str, Any]: ...
    def target(
        self, batch: Mapping[str, Any], corrupted_batch: Mapping[str, Any]
    ) -> torch.Tensor: ...
    def loss(self, model_output: ModelOutput, target: torch.Tensor) -> torch.Tensor: ...
    def sample(self, model: Any, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Artifact & PrepGraph
# ---------------------------------------------------------------------------


@runtime_checkable
class ArtifactProducerProtocol(Protocol):
    def prepare(self, cfg: Mapping[str, Any] | None = None) -> None: ...
    def produce(self, sample: Mapping[str, Any]) -> dict[str, torch.Tensor]: ...
    def finalize(self) -> Path: ...


@runtime_checkable
class ArtifactStoreProtocol(Protocol):
    def put(self, sample_id: str, tensors_dict: Mapping[str, torch.Tensor]) -> None: ...
    def get(self, sample_id: str) -> dict[str, torch.Tensor]: ...
    def contains(self, sample_id: str) -> bool: ...
    def iter_keys(self) -> Iterable[str]: ...


@runtime_checkable
class PrepNodeProtocol(Protocol):
    name: str
    kind: str  # load / tokenize / chunk / pack / mix / join / index / validate / materialize
    inputs: list[str]
    config: Mapping[str, Any]
    schema_kind: str  # name into SCHEMA_VERSION (e.g. "rows", "tokenized_rows", ...)

    def code_version(self) -> str: ...
    def fingerprint(self, input_fps: Iterable[str] = ()) -> str: ...
    def estimate(self, ctx: Any) -> Any: ...
    def run(self, ctx: Any) -> Any: ...


__all__ = [
    "ArchitectureProfileProtocol",
    "ArtifactProducerProtocol",
    "ArtifactStoreProtocol",
    "CALLBACK_EVENTS",
    "CallbackProtocol",
    "CheckpointManagerProtocol",
    "ChunkerProtocol",
    "CollatorProtocol",
    "DataModuleProtocol",
    "EngineProtocol",
    "EnvironmentProtocol",
    "GenerationStrategyProtocol",
    "GenerativeModelProtocol",
    "GradSyncStrategyProtocol",
    "JudgeProtocol",
    "LossContext",
    "LossFnProtocol",
    "StepOutput",
    "LoggerProtocol",
    "MetricProtocol",
    "ModelOutput",
    "ModelParallelStrategyProtocol",
    "ModelProtocol",
    "ObjectiveProtocol",
    "OptimizerWrapperProtocol",
    "PipelineScheduleProtocol",
    "PrepNodeProtocol",
    "ProbeProtocol",
    "ProcessorProtocol",
    "RetrieverProtocol",
    "SamplerProtocol",
    "SchedulerProtocol",
    "TokenizerProtocol",
    "TrainerProtocol",
    "UpdateRuleProtocol",
]
