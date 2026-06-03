"""GRPO — programmatic API example.

Equivalent CLI:
    lighttrain train -c recipes/grpo.yaml
"""

from lighttrain.cli._runtime import setup_run_from_config
from lighttrain.config import load_config

cfg = load_config("recipes/grpo.yaml")
bundle = setup_run_from_config(cfg)
bundle["trainer"].fit()
