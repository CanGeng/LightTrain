"""Data system core: the ``Sample`` schema (``core``) plus cache / mixing /
packing plumbing consumed by the PrepGraph framework.

Concrete datasets / collators / samplers / tokenizers / processors / data
modules are registered impls and live in ``lighttrain.builtin_plugins.data``
(DESIGN §3.3). Import the schema from :mod:`lighttrain.data.core`.
"""

from __future__ import annotations
