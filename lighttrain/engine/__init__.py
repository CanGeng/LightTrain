"""Engine — innermost training step.

``StepContext`` + the ``EngineProtocol`` (in ``lighttrain.protocols``) are the
core seam; the concrete ``StandardEngine`` is a registered impl living in
``lighttrain.builtin_plugins.engine.standard`` (DESIGN §3.3).
"""

from __future__ import annotations

from ._context import StepContext

__all__ = ["StepContext"]
