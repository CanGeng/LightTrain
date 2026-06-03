"""Diffusion ε-prediction — programmatic API example.

Equivalent CLI:
    lighttrain train -c recipes/diffusion_eps.yaml
"""

import lighttrain.builtin_plugins.architectures.diffusion_unet  # noqa: F401
from lighttrain.cli._runtime import setup_run_from_config
from lighttrain.config import load_config

cfg = load_config("recipes/diffusion_eps.yaml")
bundle = setup_run_from_config(cfg)
bundle["trainer"].fit()
