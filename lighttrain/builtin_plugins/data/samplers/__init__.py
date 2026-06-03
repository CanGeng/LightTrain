"""Advanced samplers.

Registers three sampler kinds:

* ``length_grouped`` — bucket by length to cut padding waste
* ``curriculum``    — sample within an evolving length percentile band
* ``stateful_resumable`` — chunked iteration with mid-epoch resume
"""

from lighttrain.registry import register
from .curriculum import CurriculumSampler
from .length_grouped import LengthGroupedSampler
from .stateful_resumable import StatefulResumableSampler

# Each class keeps its constructor; we register thin wrappers that forward
# ``dataset`` (the resolver injects it) plus the user-supplied kwargs.

register("sampler", "length_grouped")(LengthGroupedSampler)
register("sampler", "curriculum")(CurriculumSampler)
register("sampler", "stateful_resumable")(StatefulResumableSampler)


__all__ = [
    "CurriculumSampler",
    "LengthGroupedSampler",
    "StatefulResumableSampler",
]
