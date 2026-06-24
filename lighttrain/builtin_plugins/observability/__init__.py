"""Bundled observability plugins.

Mirrors core ``lighttrain.observability``. Concrete diagnostic callbacks live
under ``diagnostics/``; registration happens via the recursive component walk
(``lighttrain.config._components.import_all_components``), so nothing needs to
be re-exported here.
"""

from __future__ import annotations
