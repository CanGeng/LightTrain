"""lighttrain.distributed — distributed topology and strategy interfaces.

Always importable: ``ParallelContext.single_gpu()`` and the ``GradSyncStrategy``
Protocol work without ``torch.distributed`` or NCCL being installed/initialized.

Concrete strategy implementations (DDP, FSDP, ZeRO) live in
``builtin_plugins/distributed/`` and are only imported when the user
selects them via config.
"""

from ._context import ParallelContext
from ._protocols import GradSyncStrategy

__all__ = [
    "GradSyncStrategy",
    "ParallelContext",
]
