"""builtin_plugins.distributed — concrete distributed strategy implementations.

Registers all strategies into the lighttrain registry so recipes can select
them by name.  Heavy dependencies (torch.distributed, deepspeed, etc.) are
lazily imported inside each strategy's methods.

Strategies registered here:
  grad_sync_strategy:       ddp, fsdp, deepspeed
  model_parallel_strategy:  tensor_parallel, tp_aware, sequence_parallel, expert_parallel
  pipeline_schedule:        1f1b, gpipe, interleaved_1f1b
"""

from .strategies.ddp import DDPStrategy
from .strategies.fsdp import FSDPStrategy
from .model_parallel.tp_auto import TensorParallelStrategy
from .model_parallel.tp_aware import TPAwareModelAdapter, TPAwareStrategy
from .model_parallel.sp import SequenceParallelStrategy
from .model_parallel.ep import ExpertParallelStrategy
from .pipeline.schedules import OneFOneBSchedule, GPipeSchedule, Interleaved1F1BSchedule

try:
    from .strategies.zero import ZeROStrategy  # noqa: F401 — requires deepspeed
except ImportError:
    ZeROStrategy = None  # type: ignore[assignment,misc]

__all__ = [
    "DDPStrategy",
    "ExpertParallelStrategy",
    "FSDPStrategy",
    "GPipeSchedule",
    "Interleaved1F1BSchedule",
    "OneFOneBSchedule",
    "SequenceParallelStrategy",
    "TPAwareModelAdapter",
    "TPAwareStrategy",
    "TensorParallelStrategy",
]
