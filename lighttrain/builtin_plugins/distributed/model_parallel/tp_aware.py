"""TPAwareStrategy — hand-written TP-aware model adapter.

For models that cannot be automatically surgically modified (non-standard
attention, custom operators, etc.), authors subclass ``TPAwareModelAdapter``
and override ``apply_tp_plan(parallel_ctx)``.
"""

from __future__ import annotations

import torch.nn as nn

from lighttrain.distributed._context import ParallelContext
from lighttrain.registry import register


class TPAwareModelAdapter(nn.Module):
    """Base class for TP-aware models.

    Subclass this and override ``apply_tp_plan(parallel_ctx)`` to perform
    custom TP surgery (replace Linear layers with ColumnParallelLinear etc.).
    The method must modify ``self`` in-place and return None.
    """

    def apply_tp_plan(self, parallel_ctx: ParallelContext) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must implement apply_tp_plan(parallel_ctx)"
        )


@register("model_parallel_strategy", "tp_aware")
class TPAwareStrategy:
    """Strategy that delegates TP surgery to the model itself.

    The model must implement ``apply_tp_plan(parallel_ctx)`` — either by
    inheriting from ``TPAwareModelAdapter`` or by duck-typing.
    """

    def apply(self, model: nn.Module, parallel_ctx: ParallelContext) -> nn.Module:
        from lighttrain.config._exceptions import ConfigError

        if not hasattr(model, "apply_tp_plan"):
            raise ConfigError(
                f"tp_aware strategy requires the model to implement "
                f"apply_tp_plan(parallel_ctx), but {type(model).__name__!r} "
                "does not have this method. "
                "Subclass TPAwareModelAdapter or add apply_tp_plan manually."
            )
        model.apply_tp_plan(parallel_ctx)  # type: ignore[attr-defined]
        return model

    def is_stateless(self) -> bool:
        return True


__all__ = ["TPAwareModelAdapter", "TPAwareStrategy"]
