"""QLoRA 4-bit fine-tuning — programmatic API example.

Requires: pip install -e '.[quant,peft]'

Equivalent CLI:
    lighttrain train -c recipes/qlora.yaml
"""

from lighttrain.config import load_config
from lighttrain.cli._runtime import setup_run_from_config

import frontier_plugins.quant  # registers BNB / QLoRA adapters  # noqa: F401

cfg = load_config("recipes/qlora.yaml")
bundle = setup_run_from_config(cfg)
bundle["trainer"].fit()
