"""Predictive Coding Network — programmatic API example.

Equivalent CLI:
    lighttrain train -c recipes/pcn_demo.yaml
"""

from lighttrain.config import load_config
from lighttrain.cli._runtime import setup_run_from_config

import lighttrain.builtin_plugins.update_rules.pcn  # noqa: F401 — registers the PCN update rule

cfg = load_config("recipes/pcn_demo.yaml")
bundle = setup_run_from_config(cfg)
bundle["trainer"].fit()
