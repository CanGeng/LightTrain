"""Fork + resume with 3-generation lineage — programmatic API example (R16).

Equivalent CLI sequence:
    lighttrain train -c recipes/fork_resume.yaml
    lighttrain fork --from runs/fork_gen1/<run>/checkpoints/step_50 \
                    -c recipes/fork_resume.yaml ++optim.lr=1.5e-4 ++exp=fork_gen2
    lighttrain resume --run runs/fork_gen2/<run>
"""

from pathlib import Path

from lighttrain.cli._runtime import setup_run_from_config
from lighttrain.config import load_config
from lighttrain.lab.fork import fork

# Assume gen1 has already been trained.
# Locate the checkpoint (adjust path to your actual run dir):
gen1_ckpt = Path("runs/fork_gen1")
checkpoints = list(gen1_ckpt.glob("*/checkpoints/step_*"))
if not checkpoints:
    raise FileNotFoundError(
        "No gen1 checkpoint found. Run `lighttrain train -c recipes/fork_resume.yaml` first."
    )
ckpt = sorted(checkpoints)[-1]

# Fork gen1 → gen2 with halved LR
cfg2 = load_config("recipes/fork_resume.yaml", ["++optim.lr=1.5e-4", "++exp=fork_gen2"])
r2 = fork(ckpt, cfg2)
print(f"Gen2 run dir: {r2.new_run_dir}")
print(f"Lineage recorded: {r2.lineage_edge_recorded}")

# Resume gen2
bundle2 = setup_run_from_config(cfg2, existing_run_dir=r2.new_run_dir)
bundle2["trainer"].fit()
