"""Produce teacher logit artifacts — programmatic API example.

Equivalent CLI:
    lighttrain produce-artifact -c recipes/produce_teacher.yaml
"""

from pathlib import Path
from lighttrain.config import load_config
from lighttrain.cli._produce import run_produce

manifest = run_produce(Path("recipes/produce_teacher.yaml"))
print(f"Artifact finalized → {manifest}")
