"""Stateful RWKV pretraining — programmatic API example.

Equivalent CLI:
    lighttrain train -c recipes/pretrain_rwkv.yaml
"""

from lighttrain.config import load_config
from lighttrain.cli._runtime import setup_run_from_config

import frontier_plugins.architectures.rwkv  # registers TinyRWKVModel  # noqa: F401

cfg = load_config("recipes/pretrain_rwkv.yaml")
bundle = setup_run_from_config(cfg)
bundle["trainer"].fit()
