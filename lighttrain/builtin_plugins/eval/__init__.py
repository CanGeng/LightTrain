"""Bundled eval plugins.

Mirrors core ``lighttrain.eval``. Registered eval-metric implementations land
under ``metrics/`` (the ``metric`` registry category); none are bundled yet.
Registration happens via the recursive component walk
(``lighttrain.config._components.import_all_components``), so nothing needs to
be re-exported here.
"""

from __future__ import annotations
