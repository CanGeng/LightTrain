"""Forward-Forward algorithm — programmatic API example.

Equivalent CLI:
    lighttrain train -c recipes/ff_demo.yaml
"""

from lighttrain.config import load_config
from lighttrain.cli._runtime import setup_run_from_config

import lighttrain.plugins.update_rules  # registers ForwardForward update rule  # noqa: F401

cfg = load_config("recipes/ff_demo.yaml")
bundle = setup_run_from_config(cfg)
bundle["trainer"].fit()
