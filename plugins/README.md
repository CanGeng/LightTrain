# plugins

Opt-in plugins. Each plugin lives in its own subdirectory
(`layer_offload/`, `agent/`, `rag/`, `generation_strategies/`, `judge/`,
`probes/`, `architectures/{rwkv,mamba,diffusion_unet}/`,
`update_rules/{forward_forward,pcn,dfa}/`, `sweep_backends/optuna/`,
`generation_backends/vllm/`, `quant/`) and registers via Python entry points
(`lighttrain.plugins`).
