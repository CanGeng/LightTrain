"""Per-domain CLI command modules.

Each module here only *defines* command functions (no Typer app, no decorators).
The single assembler in ``lighttrain.cli._app`` registers them onto ``app`` /
the sub-typers in the original source order so ``--help`` listing is unchanged.
"""
