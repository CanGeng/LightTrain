"""Basic training objectives (next-token / masked-denoising).

Generative objectives (diffusion / flow-matching / JEPA) moved to
``lighttrain.plugins.objectives`` (DESIGN §3.3: specific objective impls are
frontier; the Protocol stays in ``lighttrain.protocols``).
"""

from .masked_denoising import MaskedDenoisingObjective
from .next_token import NextTokenObjective

__all__ = [
    "MaskedDenoisingObjective",
    "NextTokenObjective",
]
