"""Architecture profile base infra (kept in core).

``ArchitectureProfile`` / ``ObjectiveProfile`` / ``LossOnlyObjective`` are the
core plumbing (the ``ArchitectureProfileProtocol`` is in
``lighttrain.protocols``). Concrete profile factories — the default
``transformer`` profile and the frontier rwkv / mamba / diffusion_unet / jepa
adapters — are registered impls living in
``lighttrain.builtin_plugins.optim.architectures`` (DESIGN §3.3).
"""

from .profile import ArchitectureProfile, ObjectiveProfile

__all__ = [
    "ArchitectureProfile",
    "ObjectiveProfile",
]
