"""Model adapter extensions.

Concrete model implementations live here (DESIGN §3.3: the ``ModelProtocol`` /
``GenerativeModelProtocol`` stay in ``lighttrain.protocols``; impls in
builtin_plugins). Registered via auto-discovery; import the submodule you need:

    from lighttrain.builtin_plugins.models.text.tiny_lm import TinyCausalLM
    from lighttrain.builtin_plugins.models.text.hf_causal import HFCausalLM
    from lighttrain.builtin_plugins.models.peft import LoRAAdapter, IA3Adapter
"""
