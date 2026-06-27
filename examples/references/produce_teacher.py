"""Produce teacher logit artifacts — programmatic API example.

Equivalent CLI:
    lighttrain produce-artifact -c examples/references/recipes/produce_teacher.yaml
"""

from pathlib import Path

from lighttrain.cli._produce import run_produce

manifest = run_produce(Path("examples/references/recipes/produce_teacher.yaml"))
print(f"Artifact finalized → {manifest}")
