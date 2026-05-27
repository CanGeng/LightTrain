"""Online PPO — programmatic API example.

Equivalent CLI:
    lighttrain train -c recipes/ppo_online.yaml
"""

from lighttrain.config import load_config
from lighttrain.cli._runtime import setup_run_from_config

cfg = load_config("recipes/ppo_online.yaml")
bundle = setup_run_from_config(cfg)
bundle["trainer"].fit()
