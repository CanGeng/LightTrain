"""Image classification — programmatic API example.

The reference supervised-vision workflow (TinyCNNClassifier + classification
loss on the generic ``pretrain`` trainer). The synthetic dataset is registered
via the recipe's ``user_modules:`` — ``load_config`` is the chokepoint that
imports them, so no explicit plugin import is needed here.

Equivalent CLI:
    lighttrain train -c recipes/image_cls.yaml
"""

from lighttrain.cli._runtime import setup_run_from_config
from lighttrain.config import load_config

cfg = load_config("recipes/image_cls.yaml")
bundle = setup_run_from_config(cfg)
bundle["trainer"].fit()
