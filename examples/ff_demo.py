"""Forward-Forward algorithm — programmatic API example.

Equivalent CLI:
    lighttrain train -c recipes/ff_demo.yaml
"""

import lighttrain.builtin_plugins.engine.update_rules.forward_forward  # noqa: F401 — registers the ForwardForward update rule
from lighttrain.cli._runtime import setup_run_from_config
from lighttrain.config import load_config

cfg = load_config("recipes/ff_demo.yaml")
bundle = setup_run_from_config(cfg)
bundle["trainer"].fit()
