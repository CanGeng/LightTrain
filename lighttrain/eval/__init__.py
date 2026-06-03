"""EvalSuite — Evaluator / EvalTask / EvalReport / RegressionGate / metrics.

Judge implementations moved to ``lighttrain.builtin_plugins.judges`` (DESIGN §3.3:
specific judge impls are frontier; the Protocol stays in ``lighttrain.protocols``
and the runtime resolves judges via the ``judge`` registry category).
"""

from .generation_eval import GenerationEvalResult, GenerationEvalTask
from .suite import EvalReport, EvalTask, Evaluator, RegressionGate

__all__ = [
    "EvalReport",
    "EvalTask",
    "Evaluator",
    "GenerationEvalResult",
    "GenerationEvalTask",
    "RegressionGate",
]
