"""Optional plug-ins shipping alongside lighttrain.

Each subpackage is opt-in via ``pip install -e .[<extras>]`` and gets
``import``-ed eagerly by ``lighttrain.cli._runtime._eager_import_components``
so its ``@register`` calls land in the registry before recipes resolve.

Plugins currently shipped:

* :mod:`frontier_plugins.layer_offload` — ``LayerOffloadEngine``
* :mod:`frontier_plugins.quant`         — bnb 4/8-bit + QLoRA
"""

__all__: list[str] = []
