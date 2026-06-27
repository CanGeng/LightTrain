"""Online PPO — programmatic API example.

Equivalent CLI:
    lighttrain train -c examples/references/recipes/ppo_online.yaml
"""

from lighttrain.cli._runtime import setup_run_from_config
from lighttrain.config import load_config

cfg = load_config("examples/references/recipes/ppo_online.yaml")
bundle = setup_run_from_config(cfg)
bundle["trainer"].fit()
