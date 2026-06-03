"""Quantization wiring — DESIGN §8.4 (M5).

Windows / no-bitsandbytes path:

* The ``qlora`` model short-name is registered (recipe parsing works).
* Constructing ``QLoRAAdapter`` without bnb raises a clear ImportError.

True numerical verification (R13) lands in ``tests/test_recipes_r13.py``
and is gated behind the ``heavy`` marker + Linux + GPU skip — see M5 doc.
"""

from __future__ import annotations

import pytest

import lighttrain.builtin_plugins.quant  # noqa: F401 — register qlora

from lighttrain.registry import contains as _has


def test_qlora_short_name_is_registered():
    assert _has("model", "qlora")


def test_qlora_construction_without_bnb_raises_clear_hint():
    pytest.importorskip("peft")
    try:
        import bitsandbytes  # noqa: F401

        pytest.skip("bitsandbytes is installed; this test exercises the missing-bnb path")
    except ImportError:
        pass
    from lighttrain.builtin_plugins.quant import QLoRAAdapter

    with pytest.raises(ImportError, match="bitsandbytes"):
        QLoRAAdapter(
            base={
                "name": "tiny_lm",
                "vocab_size": 32, "d_model": 8, "n_layers": 1, "n_heads": 2,
                "max_seq_len": 8,
            },
            bits=4,
            lora={"r": 4, "lora_alpha": 8},
        )
