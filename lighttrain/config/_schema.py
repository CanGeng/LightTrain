"""Pydantic v2 schema for lighttrain configs.

The ``trainer`` and ``engine`` sections are typed; the rest stays permissive
so users can add experimental fields without schema churn.
``extra='allow'`` is preserved on RootConfig.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class GradSyncConfig(BaseModel):
    """Config for data-parallel / optimizer-sharding strategy."""

    model_config = ConfigDict(extra="allow")

    name: str = "noop"


class ParallelSection(BaseModel):
    """Distributed topology knobs. Absent = single-GPU degenerate mode.

    Only data parallelism is supported (DDP / FSDP / DeepSpeed ZeRO via
    ``grad_sync``). Tensor / pipeline / expert / sequence parallelism were
    removed.
    """

    model_config = ConfigDict(extra="allow")

    backend: Literal["nccl", "gloo"] = "nccl"
    dp: int = 1
    force_cpu: bool = False
    grad_sync: GradSyncConfig = Field(default_factory=GradSyncConfig)


class ComponentSpec(BaseModel):
    """A registry-resolvable component reference.

    Either ``name`` (short name into a registry category) **or** ``_target_``
    (dotted import path) is required, never both. Remaining keys become
    construction parameters via ``params`` (or are merged from the top level
    when constructed from a flat mapping).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str | None = None
    target: str | None = Field(default=None, alias="_target_")
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_xor(self) -> ComponentSpec:
        if (self.name is None) == (self.target is None):
            raise ValueError(
                "ComponentSpec requires exactly one of 'name' or '_target_' "
                "(got both or neither)."
            )
        return self


class TrainerSection(BaseModel):
    """Loop-level knobs."""

    model_config = ConfigDict(extra="allow")

    name: str = "pretrain"
    max_steps: int = 1000
    val_every: int = 0
    ckpt_every: int = 500
    log_every: int = 50
    grad_clip: float = 1.0
    accumulate: int = 1


class EngineSection(BaseModel):
    """Per-step engine wiring."""

    model_config = ConfigDict(extra="allow")

    name: str = "standard"
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    update_rule: dict[str, Any] = Field(default_factory=lambda: {"name": "standard"})


class RootConfig(BaseModel):
    """Root config schema. Component sections stay loosely typed for ergonomics."""

    model_config = ConfigDict(extra="allow")

    mode: Literal["lab", "prod"] = "lab"
    seed: int = 42
    run_dir: str | None = None
    run_root: str = "runs"
    exp: str = "default"

    # Component sections — typed as Any so users can drop in arbitrary mappings.
    # v0.1.8: ``model`` is a *string selector* into ``model_profiles``; a bare
    # dict here is rejected at build time (run ``migrate config --to-profiles``).
    model: Any | None = None
    model_profiles: dict[str, Any] | None = None
    # Multi-model / multi-optimizer set (keystone step 4). A lone ``model:`` +
    # ``model_profiles:`` and a lone ``optim:`` remain valid and desugar into
    # ``models: {main: {...}}`` / ``optimizers: {main: ...}`` at build time.
    # Each ``models`` entry: {spec, trainable, optimizer?, checkpoint?}; ``spec``
    # is an inline component spec or ``{profile: <name>}`` into ``model_profiles``.
    models: dict[str, Any] | None = None
    optimizers: dict[str, Any] | None = None
    data: Any | None = None
    optim: Any | None = None
    loss: Any | None = None
    # Non-standard training objective (diffusion/jepa/...). The canonical
    # training seam: when present it replaces ``loss:`` (the runtime wraps a
    # bare ``loss:`` into a LossOnlyObjective). ``loss:`` and ``objective:`` are
    # mutually exclusive at the top level (checked at load_config); an objective
    # may carry a *nested* ``loss:``/``aux_losses:`` for composite paradigms.
    objective: Any | None = None
    scheduler: Any | None = None
    callbacks: Any | None = None
    logger: Any | None = None
    prep_graph: Any | None = None

    trainer: TrainerSection | None = None
    engine: EngineSection | None = None
    parallel: ParallelSection | None = None
    user_modules: list[str] = Field(default_factory=list)


__all__ = [
    "ComponentSpec",
    "EngineSection",
    "GradSyncConfig",
    "ParallelSection",
    "RootConfig",
    "TrainerSection",
]
