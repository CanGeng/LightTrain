"""Sample schema + sample-id helpers — the data contract kept in core.

Dataset / Collator / Sampler / Tokenizer / DataModule *implementations* are
registered impls and now live in ``lighttrain.builtin_plugins.data`` (DESIGN
§3.3: protocols in ``lighttrain.protocols``, impls in builtin_plugins). Only the
``Sample`` schema (``_schema``) stays here as framework plumbing.
"""

from __future__ import annotations

from ._schema import Sample, derive_sample_id, is_valid_sample

__all__ = ["Sample", "derive_sample_id", "is_valid_sample"]
