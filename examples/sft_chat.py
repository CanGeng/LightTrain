"""Supervised fine-tuning — programmatic API example.

Equivalent CLI:
    lighttrain train -c recipes/sft_chat.yaml
"""

from lighttrain.cli._runtime import setup_run_from_config
from lighttrain.config import load_config

cfg = load_config("recipes/sft_chat.yaml")
bundle = setup_run_from_config(cfg)
bundle["trainer"].fit()
