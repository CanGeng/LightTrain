"""builtin_plugins.distributed — concrete distributed strategy implementations.

Registers all strategies into the lighttrain registry so recipes can select
them by name.  Heavy dependencies (torch.distributed, deepspeed, etc.) are
lazily imported inside each strategy's methods.

Strategies registered here:
  grad_sync_strategy:       ddp, fsdp, deepspeed
"""

from .strategies.ddp import DDPStrategy
from .strategies.fsdp import FSDPStrategy

try:
    from .strategies.zero import ZeROStrategy  # noqa: F401 — requires deepspeed
except ImportError:
    ZeROStrategy = None  # type: ignore[assignment,misc]

__all__ = [
    "DDPStrategy",
    "FSDPStrategy",
]
