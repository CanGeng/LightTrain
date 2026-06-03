"""Data-plane extensions: concrete datasets / collators / samplers / tokenizers
/ processors / data modules.

Registered impls (DESIGN §3.3: protocols in ``lighttrain.protocols``, the
``Sample`` schema + cache/mixing/packing plumbing stay in ``lighttrain.data``).
Discoverable via ``import_all_components`` (auto-discovery); import a submodule
directly when you need a specific class:

    from lighttrain.builtin_plugins.data.core.tokenizers import ByteTokenizer
    from lighttrain.builtin_plugins.data.processors.text import HFTextProcessor

Importing this package eagerly registers the **core** data components
(dataset / collator / sampler / tokenizer / data_module) — restoring the old
``lighttrain.data`` package-import side effect — so direct PrepGraph API callers
(``PrepGraph.from_config`` / ``PrepRunner.run`` without a prior
``import_all_components()``) can resolve nested short-name specs like
``tokenizer: {name: byte}``. ``core`` is dependency-light; the heavier
``processors`` (transformers / librosa / decord) stay lazy.
"""

from __future__ import annotations

from . import core  # noqa: F401 — eager registration side effect (see docstring)
