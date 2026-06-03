"""Architecture profiles + base infra.

The JEPA architecture moved to ``lighttrain.builtin_plugins.architectures.jepa`` (DESIGN
§3.3: specific architecture adapters are frontier). ``ArchitectureProfile`` /
``ObjectiveProfile`` base infra + the transformer profile stay here.
"""

from .profile import ArchitectureProfile, ObjectiveProfile
from .transformer import transformer_profile

__all__ = [
    "ArchitectureProfile",
    "ObjectiveProfile",
    "transformer_profile",
]
