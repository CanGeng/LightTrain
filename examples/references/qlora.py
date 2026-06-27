"""QLoRA 4-bit fine-tuning — programmatic API example.

Requires: pip install -e '.[quant,peft]'

Equivalent CLI:
    lighttrain train -c examples/references/recipes/qlora.yaml
"""

import lighttrain.builtin_plugins.quant  # registers BNB / QLoRA adapters  # noqa: F401
from lighttrain.cli._runtime import setup_run_from_config
from lighttrain.config import load_config

cfg = load_config("examples/references/recipes/qlora.yaml")
bundle = setup_run_from_config(cfg)
bundle["trainer"].fit()
