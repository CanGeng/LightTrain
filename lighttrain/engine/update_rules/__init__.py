"""UpdateRule plumbing — shared ``_primitives`` kept in core.

The ``UpdateRuleProtocol`` is in ``lighttrain.protocols``; concrete update rules
(standard / sam / mezo / rl) are registered impls living in
``lighttrain.builtin_plugins.engine.update_rules`` (DESIGN §3.3). The shared
``_primitives`` helpers (apply_update / MicroState / ...) stay here as core
framework, consumed by both core trainers and the relocated rules.
"""

from __future__ import annotations
