# frontier_plugins

Opt-in plugins per DESIGN.md §3 / §4.1. Each plugin lives in its own subdirectory
(`layer_offload/`, `agent/`, `rag/`, `generation_strategies/`, `judge/`,
`probes/`, `architectures/{rwkv,mamba,diffusion_unet}/`,
`update_rules/{forward_forward,pcn,dfa}/`, `sweep_backends/optuna/`,
`generation_backends/vllm/`, `quant/`) and registers via Python entry points
(`lighttrain.plugins`).

M0 ships only this placeholder. Plugins are added per the milestone schedule
(§26): `layer_offload` + `quant` in M5, `architectures` + `update_rules/extras`
in M7, `generation_backends/vllm` and `sweep_backends/optuna` in M6/M8.
