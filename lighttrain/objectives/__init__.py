"""Training objectives (next-token / diffusion / flow / JEPA / masked-denoising)."""

from .diffusion import DiffusionObjective
from .flow_matching import FlowMatchingObjective
from .jepa import JEPAObjective
from .masked_denoising import MaskedDenoisingObjective
from .next_token import NextTokenObjective

__all__ = [
    "DiffusionObjective",
    "FlowMatchingObjective",
    "JEPAObjective",
    "MaskedDenoisingObjective",
    "NextTokenObjective",
]
