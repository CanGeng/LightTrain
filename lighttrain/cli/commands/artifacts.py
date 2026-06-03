"""Artifact / export commands: produce-artifact / convert-checkpoint / export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from lighttrain.cli._context import console
from lighttrain.config import ConfigError, load_config


def produce_artifact_cmd(
    config: Path = typer.Option(..., "-c", "--config"),
    estimate: bool = typer.Option(False, "--estimate"),
    overrides: list[str] = typer.Argument(None, help="OmegaConf-style overrides."),
) -> None:
    """Run an ArtifactProducer offline.

    Reads ``cfg.artifacts.producer`` + ``cfg.artifacts.store`` from the recipe,
    iterates the configured train dataset (no DataLoader / collator), runs
    ``model.forward`` per sample under ``no_grad``, and writes the resulting
    tensors to the store.
    """
    from lighttrain.cli._produce import (
        run_produce,  # local import to avoid pulling torch eagerly
    )

    try:
        manifest = run_produce(
            config,
            overrides=list(overrides or []),
            estimate=estimate,
            console=console,
        )
    except (ConfigError, FileNotFoundError, RuntimeError) as e:
        console.print(f"[red]produce-artifact error:[/] {e}")
        raise typer.Exit(code=1) from e
    console.print(f"[green]artifact finalized[/] -> {manifest}")


def convert_checkpoint_cmd(
    from_: str = typer.Option(..., "--from", help="Source format: safetensors | pt | hf"),
    to: str = typer.Option(..., "--to", help="Target format: safetensors | pt | hf"),
    path: Path = typer.Option(..., "--path", help="Path to checkpoint file or directory."),
    out: Path | None = typer.Option(None, "--out", help="Output path (default: next to source)."),
) -> None:
    """Convert a checkpoint between storage formats.

    \\b
    Supported conversions:
      pt → safetensors   Load torch .pt state dict, save as safetensors
      safetensors → pt   Load safetensors, save as torch .pt
      hf → safetensors   Load HuggingFace model dir, save merged safetensors

    \\b
    Examples:
      lighttrain convert-checkpoint --from pt --to safetensors --path model.pt
      lighttrain convert-checkpoint --from safetensors --to pt --path model.safetensors
    """
    import torch

    from_ = from_.lower().strip()
    to = to.lower().strip()

    if not path.exists():
        console.print(f"[red]path not found:[/] {path}")
        raise typer.Exit(code=1)

    try:
        if from_ in ("pt", "torch") and to == "safetensors":
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            if hasattr(state, "items"):
                state_dict = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
            else:
                raise ValueError("checkpoint is not a state dict")
            from safetensors.torch import save_file

            out_path = out or path.with_suffix(".safetensors")
            save_file({k: v.contiguous() for k, v in state_dict.items()}, str(out_path))
            console.print(f"[green]written →[/] {out_path}")

        elif from_ == "safetensors" and to in ("pt", "torch"):
            from safetensors.torch import load_file

            state_dict = load_file(str(path))
            out_path = out or path.with_suffix(".pt")
            torch.save(state_dict, str(out_path))
            console.print(f"[green]written →[/] {out_path}")

        elif from_ == "hf" and to == "safetensors":
            try:
                from transformers import AutoModelForCausalLM
            except ImportError as exc:
                raise RuntimeError(
                    "hf→safetensors requires transformers: pip install transformers"
                ) from exc
            model = AutoModelForCausalLM.from_pretrained(str(path))
            from safetensors.torch import save_file

            out_path = out or (path / "model_merged.safetensors")
            save_file(
                {k: v.contiguous() for k, v in model.state_dict().items()},
                str(out_path),
            )
            console.print(f"[green]written →[/] {out_path}")

        else:
            console.print(
                f"[red]unsupported conversion:[/] {from_!r} → {to!r}. "
                "Supported: pt→safetensors, safetensors→pt, hf→safetensors"
            )
            raise typer.Exit(code=1)

    except Exception as exc:
        console.print(f"[red]convert-checkpoint error:[/] {exc}")
        raise typer.Exit(code=1) from exc


def export_cmd(
    to: str = typer.Option(..., "--to", help="Export format: safetensors | hf | gguf"),
    ckpt: Path = typer.Option(..., "--ckpt", help="Checkpoint directory (step_<n>/)."),
    out: Path = typer.Option(..., "--out", help="Output path or directory."),
    config: Path | None = typer.Option(
        None, "-c", "--config", help="Recipe YAML (needed for hf / gguf export)."
    ),
    overrides: list[str] = typer.Argument(None, help="OmegaConf-style overrides."),
) -> None:
    """Export a checkpoint to safetensors, HuggingFace, or GGUF format.

    \\b
    Examples:
      # Export model weights as a single safetensors file
      lighttrain export --to safetensors --ckpt runs/exp/run_001/checkpoints/step_500 \\
                        --out model.safetensors

      # Export as HuggingFace model directory (requires --config)
      lighttrain export --to hf --ckpt runs/exp/run_001/checkpoints/step_500 \\
                        --config recipes/pretrain_causal.yaml --out hf_model/

      # Export as GGUF (requires llama.cpp convert script on PATH)
      lighttrain export --to gguf --ckpt runs/exp/run_001/checkpoints/step_500 \\
                        --config recipes/pretrain_causal.yaml --out model.gguf
    """
    import torch

    to = to.lower().strip()

    if not ckpt.exists():
        console.print(f"[red]checkpoint not found:[/] {ckpt}")
        raise typer.Exit(code=1)

    # Locate model.safetensors or model.pt
    weight_file = ckpt / "model.safetensors"
    if not weight_file.exists():
        weight_file = ckpt / "model.pt"
    if not weight_file.exists():
        console.print(f"[red]no model weights found under:[/] {ckpt}")
        raise typer.Exit(code=1)

    try:
        if to == "safetensors":
            if weight_file.suffix == ".safetensors":
                import shutil as _sh

                out.parent.mkdir(parents=True, exist_ok=True)
                _sh.copy2(str(weight_file), str(out))
            else:
                state = torch.load(str(weight_file), map_location="cpu", weights_only=True)
                from safetensors.torch import save_file

                out.parent.mkdir(parents=True, exist_ok=True)
                save_file({k: v.contiguous() for k, v in state.items()}, str(out))
            console.print(f"[green]exported →[/] {out}")

        elif to == "hf":
            if config is None:
                console.print("[red]--config required for hf export[/]")
                raise typer.Exit(code=1)
            try:
                from transformers import AutoConfig, AutoModelForCausalLM
            except ImportError as exc:
                raise RuntimeError(
                    "hf export requires transformers: pip install transformers"
                ) from exc
            model = _export_primary_model(config, overrides)
            if weight_file.suffix == ".safetensors":
                from safetensors.torch import load_file

                state_dict = load_file(str(weight_file))
            else:
                state_dict = torch.load(str(weight_file), map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(out))
            console.print(f"[green]exported →[/] {out}")

        elif to == "gguf":
            import shutil as _sh
            import subprocess as _sp

            if config is None:
                console.print("[red]--config required for gguf export[/]")
                raise typer.Exit(code=1)
            convert_script = _sh.which("convert_hf_to_gguf.py") or _sh.which("convert.py")
            if convert_script is None:
                console.print(
                    "[red]gguf export requires llama.cpp convert script on PATH.[/] "
                    "Clone https://github.com/ggerganov/llama.cpp and add to PATH."
                )
                raise typer.Exit(code=1)
            # Step 1: export to a temporary HF directory
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                from transformers import AutoConfig, AutoModelForCausalLM  # noqa: F401

                model = _export_primary_model(config, overrides)
                if weight_file.suffix == ".safetensors":
                    from safetensors.torch import load_file as _load_sf

                    state_dict = _load_sf(str(weight_file))
                else:
                    state_dict = torch.load(
                        str(weight_file), map_location="cpu", weights_only=True
                    )
                model.load_state_dict(state_dict, strict=False)
                model.save_pretrained(tmpdir)
                # Step 2: invoke llama.cpp conversion script on the HF directory
                out.parent.mkdir(parents=True, exist_ok=True)
                result = _sp.run(
                    [convert_script, tmpdir, "--outfile", str(out)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    console.print(f"[red]gguf conversion failed:[/]\n{result.stderr}")
                    raise typer.Exit(code=1)
            console.print(f"[green]exported →[/] {out}")

        else:
            console.print(
                f"[red]unknown export format:[/] {to!r}. "
                "Expected: safetensors | hf | gguf"
            )
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]export error:[/] {exc}")
        raise typer.Exit(code=1) from exc


def _export_primary_model(config: Path, overrides: list[str] | None = None) -> Any:
    """Build the primary model to export, via the single source of truth
    (`build_primary_model`), so export supports every declaration form
    (`model:`+`model_profiles:` or a `models:` set) like every other command.

    Warns when the recipe declares multiple trainable models: export ships the
    primary one — the model the trainer checkpoints (`ctx.model`).
    """
    from lighttrain.config._models import build_primary_model

    cfg = load_config(config, overrides=list(overrides or []))
    model, n_trainable = build_primary_model(cfg)
    if n_trainable > 1:
        console.print(
            f"[yellow]note:[/] recipe declares {n_trainable} trainable models; "
            "exporting the primary one (the model the trainer checkpoints)."
        )
    return model
