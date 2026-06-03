"""Bundled first-party extension modules shipping inside lighttrain.

Walked eagerly by ``lighttrain.config._components.import_all_components`` (it sits
in ``_FIRST_PARTY_PACKAGES``), so its ``@register`` calls land in the registry
before recipes resolve. A submodule whose third-party extra
(``pip install -e .[<extras>]``: bnb/optuna/vllm/peft) is absent is skipped by the
per-module import contract; the package itself is always present.

Extension modules currently shipped:

* :mod:`builtin_plugins.layer_offload` — ``LayerOffloadEngine``
* :mod:`builtin_plugins.quant`         — bnb 4/8-bit + QLoRA
"""

__all__: list[str] = []
