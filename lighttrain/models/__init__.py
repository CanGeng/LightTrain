"""Model plumbing kept in core: extras (extra-output hooks) + surgery.

Concrete model adapters (``tiny_lm`` / ``hf_causal``) and PEFT adapters
(``lora`` / ``adalora`` / ``ia3``) are registered implementations and now live in
``lighttrain.builtin_plugins.models`` (DESIGN §3.3: protocol in core, impls in
builtin_plugins). ``extras`` (public ``ExtraOutputSpec`` / ``ExtrasHookManager``)
and ``surgery`` stay here as framework plumbing.
"""
