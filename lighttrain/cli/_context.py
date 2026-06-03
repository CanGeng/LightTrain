"""Shared CLI context.

The singleton Rich ``console`` lives here (not in ``_app``) so command modules
under ``cli/commands/`` and the ``_app`` assembler can both import it without a
circular import back through the assembler.
"""

from __future__ import annotations

from rich.console import Console

console = Console()
