"""Full-parameter training with LayerOffload — programmatic API example.

Equivalent CLI:
    lighttrain train -c recipes/offload_fullparam.yaml
"""

from lighttrain.config import load_config
from lighttrain.cli._runtime import setup_run_from_config

import lighttrain.builtin_plugins.layer_offload  # registers LayerOffloadEngine  # noqa: F401

cfg = load_config("recipes/offload_fullparam.yaml")
bundle = setup_run_from_config(cfg)
bundle["trainer"].fit()
