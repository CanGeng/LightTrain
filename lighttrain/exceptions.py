"""Framework-level exception hierarchy for lighttrain."""

from __future__ import annotations


class LightTrainError(RuntimeError):
    """Base class for all lighttrain framework errors."""


class BatchValidationError(LightTrainError):
    """Raised when a trainer receives a batch missing required keys."""

    def __init__(
        self,
        trainer_name: str,
        missing_keys: list[str],
        present_keys: list[str],
    ) -> None:
        present_summary = sorted(str(k) for k in present_keys)[:10]
        if len(present_keys) > 10:
            present_summary.append(f"... ({len(present_keys) - 10} more)")
        msg = (
            f"{trainer_name}: batch missing required keys {missing_keys}. "
            f"Present keys: {present_summary}. "
            f"Check that your DataModule uses the correct collator."
        )
        super().__init__(msg)


__all__ = ["BatchValidationError", "LightTrainError"]
