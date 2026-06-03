"""lighttrain.distributed — distributed topology and strategy interfaces.

Always importable: ``ParallelContext.single_gpu()`` and the three Protocol
types work without ``torch.distributed`` or NCCL being installed/initialized.

Concrete strategy implementations (DDP, FSDP, ZeRO, TP, PP, SP, EP) live
in ``builtin_plugins/distributed/`` and are only imported when the user
selects them via config.
"""

from ._context import ParallelContext
from ._noop import NoopGradSyncStrategy
from ._protocols import GradSyncStrategy, ModelParallelStrategy, PipelineSchedule

__all__ = [
    "GradSyncStrategy",
    "ModelParallelStrategy",
    "NoopGradSyncStrategy",
    "ParallelContext",
    "PipelineSchedule",
]
