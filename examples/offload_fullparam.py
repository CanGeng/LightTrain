"""Full-parameter training with LayerOffload — programmatic API example.

Equivalent CLI:
    lighttrain train -c recipes/offload_fullparam.yaml
"""

import lighttrain.builtin_plugins.layer_offload  # registers LayerOffloadEngine  # noqa: F401
from lighttrain.cli._runtime import setup_run_from_config
from lighttrain.config import load_config

cfg = load_config("recipes/offload_fullparam.yaml")
bundle = setup_run_from_config(cfg)
bundle["trainer"].fit()
