# builtin_plugins

Bundled first-party extension modules — shipped inside lighttrain and always
imported by `lighttrain.config._components.import_all_components` (the package sits
in `_FIRST_PARTY_PACKAGES`), so their `@register` calls land before recipes resolve.
Each lives in its own subdirectory (`layer_offload/`, `agent/`, `rag/`,
`generation_strategies/`, `judge/`, `probes/`,
`architectures/{rwkv,mamba,diffusion_unet}/`,
`update_rules/{forward_forward,pcn,dfa}/`, `sweep_backends/optuna/`,
`generation_backends/vllm/`, `quant/`). A submodule whose third-party extra
(bnb/optuna/vllm/peft) is absent is skipped by the per-module import contract.
