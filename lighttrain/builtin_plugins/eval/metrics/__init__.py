"""Bundled eval-metric plugins.

Mirrors core ``lighttrain.eval.metrics``. Concrete ``@register("metric", ...)``
implementations (accuracy / F1 / BLEU / …) land here as the discriminative and
generative paradigms grow; none are bundled yet. Registration is via the
recursive component walk, so nothing needs to be re-exported here.
"""

from __future__ import annotations
