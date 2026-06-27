"""Coverage-completion tests for ``lighttrain.utils.seed``.

Pins and exercises every branch left uncovered after tests/utils/test_seed.py:

* **ImportError branches for numpy** (lines 25, 47, 68): when numpy is
  absent, seed_everything / rng_state / restore_rng_state skip numpy silently.
* **ImportError branches for torch** (lines 34, 55, 77): same for torch.
* **CUDA branches** (lines 33, 54, 76): seed_everything / rng_state /
  restore_rng_state call the torch.cuda helpers when CUDA is reported as
  available — verified via mocking cuda.is_available when CUDA is absent, or
  directly on systems where CUDA is available.
* **numpy happy-path** (lines 24, 46, 67): np.random.seed / get_state /
  set_state are actually called, confirmed by spy.
* **rng_state returns expected keys** when torch is absent (only 'python').
* **restore_rng_state skips cuda key** when is_available() is False.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reimport_seed():
    """Return a freshly-bound reference to the three public functions.

    This is NOT reimporting the module; the functions already use late
    ``import`` inside their bodies, so patching sys.modules before the call
    is sufficient to steer each branch.
    """
    from lighttrain.utils.seed import restore_rng_state, rng_state, seed_everything
    return seed_everything, rng_state, restore_rng_state


# ---------------------------------------------------------------------------
# seed_everything – ImportError branches
# ---------------------------------------------------------------------------

class TestSeedEverythingImportErrorBranches:
    """Verify the silent-skip behavior when libraries are absent."""

    def test_invariant_numpy_import_error_skipped_silently(self):
        """seed_everything does not raise if numpy is unavailable (line 25).

        Patch numpy out of sys.modules then call seed_everything; must
        return the clamped seed without error.
        """
        seed_everything, _, _ = _reimport_seed()
        with patch.dict(sys.modules, {"numpy": None}):
            result = seed_everything(7)
        assert result == 7

    def test_invariant_torch_import_error_skipped_silently(self):
        """seed_everything does not raise if torch is unavailable (line 34).

        Replace torch in sys.modules with None to force ImportError inside
        the try block.
        """
        seed_everything, _, _ = _reimport_seed()
        # Also patch torch sub-modules so none leak through.
        null_torch = {k: None for k in list(sys.modules) if k.startswith("torch")}
        with patch.dict(sys.modules, null_torch):
            result = seed_everything(99)
        assert result == 99

    def test_pin_seed_everything_returns_clamped_seed_without_numpy(self):
        """Pin: return value is the 32-bit clamped seed even without numpy."""
        seed_everything, _, _ = _reimport_seed()
        with patch.dict(sys.modules, {"numpy": None}):
            result = seed_everything(2**33 + 1)
        assert result == 1  # low 32 bits of (2**33 + 1)

    def test_pin_seed_everything_returns_clamped_seed_without_torch(self):
        """Pin: return value is the 32-bit clamped seed even without torch."""
        seed_everything, _, _ = _reimport_seed()
        null_torch = {k: None for k in list(sys.modules) if k.startswith("torch")}
        with patch.dict(sys.modules, null_torch):
            result = seed_everything(2**33 + 3)
        assert result == 3


# ---------------------------------------------------------------------------
# seed_everything – numpy happy path (line 24)
# ---------------------------------------------------------------------------

class TestSeedEverythingNumpyHappyPath:
    """Verify np.random.seed is actually called with the right value."""

    def test_invariant_numpy_seed_called_with_clamped_value(self):
        """np.random.seed receives the clamped integer (line 24).

        Use a spy on np.random.seed to confirm the exact argument.
        """
        import numpy as np
        seed_everything, _, _ = _reimport_seed()
        with patch.object(np.random, "seed") as spy:
            seed_everything(42)
            spy.assert_called_once_with(42)

    def test_invariant_numpy_seed_receives_clamped_large_seed(self):
        """np.random.seed receives the low 32 bits for oversized seed."""
        import numpy as np
        seed_everything, _, _ = _reimport_seed()
        with patch.object(np.random, "seed") as spy:
            seed_everything(2**40 + 99)
            spy.assert_called_once_with(99)


# ---------------------------------------------------------------------------
# seed_everything – CUDA branch (line 33)
# ---------------------------------------------------------------------------

class TestSeedEverythingCudaBranch:
    """Cover the torch.cuda.manual_seed_all call (line 33).

    Implementation note: torch.manual_seed() internally calls
    cuda.manual_seed_all() via the C extension regardless of our Python mock
    (that accounts for 1 baseline call).  seed_everything's explicit
    ``torch.cuda.manual_seed_all(seed)`` on line 33 adds a second call when
    cuda.is_available() returns True.  We measure the *delta* against the
    baseline to isolate the branch under test.
    """

    def test_invariant_cuda_manual_seed_all_called_when_cuda_available(self):
        """When cuda.is_available() is True, the explicit manual_seed_all(seed)
        call on line 33 fires, producing one *extra* call beyond the baseline
        that torch.manual_seed() already emits.
        """
        import torch
        seed_everything, _, _ = _reimport_seed()
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "manual_seed_all") as cuda_spy:
            seed_everything(11)
        # baseline = 1 (from torch.manual_seed internals) + 1 (line 33) = 2
        assert cuda_spy.call_count == 2
        assert call(11) in cuda_spy.call_args_list

    def test_pin_cuda_manual_seed_all_not_called_explicitly_when_unavailable(self):
        """Pin: when cuda.is_available() is False the explicit branch on line 33
        is skipped, so only the C-level call from torch.manual_seed fires (1 call).
        """
        import torch
        seed_everything, _, _ = _reimport_seed()
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.cuda, "manual_seed_all") as cuda_spy:
            seed_everything(11)
        # Only 1 call: the baseline from torch.manual_seed's own C internals.
        assert cuda_spy.call_count == 1


# ---------------------------------------------------------------------------
# rng_state – ImportError branches
# ---------------------------------------------------------------------------

class TestRngStateImportErrorBranches:
    """When numpy or torch is absent, rng_state omits those keys."""

    def test_invariant_numpy_absent_rng_state_has_no_numpy_key(self):
        """rng_state omits 'numpy' key when numpy import fails (line 47)."""
        _, rng_state, _ = _reimport_seed()
        with patch.dict(sys.modules, {"numpy": None}):
            state = rng_state()
        assert "numpy" not in state
        assert "python" in state

    def test_invariant_torch_absent_rng_state_has_no_torch_key(self):
        """rng_state omits 'torch' key when torch import fails (line 55)."""
        _, rng_state, _ = _reimport_seed()
        null_torch = {k: None for k in list(sys.modules) if k.startswith("torch")}
        with patch.dict(sys.modules, null_torch):
            state = rng_state()
        assert "torch" not in state
        assert "python" in state

    def test_pin_rng_state_only_python_key_when_both_absent(self):
        """Pin: only 'python' key survives when both numpy and torch are absent."""
        _, rng_state, _ = _reimport_seed()
        null_mods = {"numpy": None}
        null_mods.update({k: None for k in list(sys.modules) if k.startswith("torch")})
        with patch.dict(sys.modules, null_mods):
            state = rng_state()
        assert set(state.keys()) == {"python"}


# ---------------------------------------------------------------------------
# rng_state – numpy happy path (line 46)
# ---------------------------------------------------------------------------

class TestRngStateNumpyHappyPath:
    """np.random.get_state is called and its result stored (line 46)."""

    def test_invariant_numpy_state_captured_in_rng_state(self):
        """'numpy' key in rng_state() output contains the np.random state.

        Spy on get_state to confirm it's invoked.
        """
        import numpy as np
        _, rng_state, _ = _reimport_seed()
        sentinel = object()
        with patch.object(np.random, "get_state", return_value=sentinel) as spy:
            state = rng_state()
        spy.assert_called_once()
        assert state["numpy"] is sentinel


# ---------------------------------------------------------------------------
# rng_state – CUDA branch (line 54)
# ---------------------------------------------------------------------------

class TestRngStateCudaBranch:
    """Cover torch.cuda.get_rng_state_all (line 54)."""

    def test_invariant_cuda_rng_state_captured_when_available(self):
        """'torch_cuda' key appears in rng_state() when CUDA is available."""
        import torch
        _, rng_state, _ = _reimport_seed()
        fake_cuda_state = [MagicMock()]
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_rng_state_all", return_value=fake_cuda_state):
            state = rng_state()
        assert "torch_cuda" in state
        assert state["torch_cuda"] is fake_cuda_state

    def test_pin_cuda_rng_state_absent_when_cuda_unavailable(self):
        """Pin: 'torch_cuda' key absent from rng_state() when CUDA is off."""
        import torch
        _, rng_state, _ = _reimport_seed()
        with patch.object(torch.cuda, "is_available", return_value=False):
            state = rng_state()
        assert "torch_cuda" not in state


# ---------------------------------------------------------------------------
# restore_rng_state – numpy ImportError branch (line 68)
# ---------------------------------------------------------------------------

class TestRestoreRngStateNumpyImportErrorBranch:
    """Verify the numpy ImportError-pass branch in restore_rng_state (line 68)."""

    def test_invariant_numpy_import_error_silenced_in_restore(self):
        """restore_rng_state does not raise when numpy is absent but state
        contains a 'numpy' key (line 68).
        """
        _, _, restore_rng_state = _reimport_seed()
        state = {"numpy": "some_state"}
        with patch.dict(sys.modules, {"numpy": None}):
            restore_rng_state(state)  # must not raise

    def test_pin_restore_rng_state_skips_numpy_when_import_fails(self):
        """Pin: python RNG is still restored even when numpy import fails."""
        import random as _random
        _, rng_state, restore_rng_state = _reimport_seed()
        from lighttrain.utils.seed import seed_everything
        seed_everything(123)
        state = rng_state()  # capture full state (with numpy)
        # advance python random
        _ = _random.random()
        _random.random()
        # restore with numpy absent; python state should still be restored
        with patch.dict(sys.modules, {"numpy": None}):
            restore_rng_state(state)
        # first draw should match state-captured position (= old_val_after_advance
        # is at +2; state was at 0, so next draw = position 1)
        # Simpler check: re-seed, capture again, advance, restore, compare
        seed_everything(321)
        state2 = rng_state()
        a = _random.random()
        # Corrupt python state
        for _ in range(10):
            _random.random()
        # Restore with numpy disabled: python RNG should still restore
        with patch.dict(sys.modules, {"numpy": None}):
            restore_rng_state(state2)
        b = _random.random()
        assert a == b


# ---------------------------------------------------------------------------
# restore_rng_state – numpy happy path (line 67)
# ---------------------------------------------------------------------------

class TestRestoreRngStateNumpyHappyPath:
    """np.random.set_state is called (line 67)."""

    def test_invariant_numpy_set_state_called_on_restore(self):
        """restore_rng_state calls np.random.set_state with the stored value."""
        import numpy as np
        _, rng_state, restore_rng_state = _reimport_seed()
        from lighttrain.utils.seed import seed_everything
        seed_everything(0)
        state = rng_state()
        with patch.object(np.random, "set_state") as spy:
            restore_rng_state(state)
        spy.assert_called_once_with(state["numpy"])


# ---------------------------------------------------------------------------
# restore_rng_state – torch ImportError branch (line 77)
# ---------------------------------------------------------------------------

class TestRestoreRngStateTorchImportErrorBranch:
    """Verify the torch ImportError-pass branch in restore_rng_state (line 77)."""

    def test_invariant_torch_import_error_silenced_in_restore(self):
        """restore_rng_state does not raise when torch is absent and 'torch'
        key exists in state (line 77).
        """
        _, _, restore_rng_state = _reimport_seed()
        import torch
        fake_torch_state = torch.get_rng_state()  # real bytes tensor
        state = {"torch": fake_torch_state}
        null_torch = {k: None for k in list(sys.modules) if k.startswith("torch")}
        with patch.dict(sys.modules, null_torch):
            restore_rng_state(state)  # must not raise


# ---------------------------------------------------------------------------
# restore_rng_state – CUDA branch (line 76)
# ---------------------------------------------------------------------------

class TestRestoreRngStateCudaBranch:
    """Cover torch.cuda.set_rng_state_all (line 76)."""

    def test_invariant_cuda_state_restored_when_available_and_key_present(self):
        """restore_rng_state calls cuda.set_rng_state_all when both
        'torch_cuda' key is present and cuda.is_available() is True (line 76).
        """
        import torch
        _, _, restore_rng_state = _reimport_seed()
        fake_cuda_state = [MagicMock()]
        state = {
            "torch": torch.get_rng_state(),
            "torch_cuda": fake_cuda_state,
        }
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "set_rng_state_all") as cuda_spy:
            restore_rng_state(state)
        cuda_spy.assert_called_once_with(fake_cuda_state)

    def test_pin_cuda_state_not_restored_when_unavailable(self):
        """Pin: cuda.set_rng_state_all not called when cuda.is_available() is False."""
        import torch
        _, _, restore_rng_state = _reimport_seed()
        fake_cuda_state = [MagicMock()]
        state = {
            "torch": torch.get_rng_state(),
            "torch_cuda": fake_cuda_state,
        }
        with patch.object(torch.cuda, "is_available", return_value=False), \
             patch.object(torch.cuda, "set_rng_state_all") as cuda_spy:
            restore_rng_state(state)
        cuda_spy.assert_not_called()

    def test_pin_cuda_state_not_restored_when_key_absent(self):
        """Pin: cuda.set_rng_state_all not called when 'torch_cuda' key is absent."""
        import torch
        _, _, restore_rng_state = _reimport_seed()
        state = {"torch": torch.get_rng_state()}  # no 'torch_cuda'
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "set_rng_state_all") as cuda_spy:
            restore_rng_state(state)
        cuda_spy.assert_not_called()


# ---------------------------------------------------------------------------
# Edge-case / invariant combination tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Miscellaneous edge-case invariants."""

    def test_invariant_seed_zero_accepted(self):
        """seed_everything(0) returns 0 and does not raise."""
        from lighttrain.utils.seed import seed_everything
        assert seed_everything(0) == 0

    def test_invariant_seed_max_32bit_accepted(self):
        """seed_everything(2**32 - 1) returns 2**32 - 1 (all ones, in range)."""
        from lighttrain.utils.seed import seed_everything
        assert seed_everything(2**32 - 1) == 2**32 - 1

    def test_invariant_seed_float_truncated_to_int(self):
        """seed_everything(3.9) applies int() before masking — returns 3."""
        from lighttrain.utils.seed import seed_everything
        assert seed_everything(3.9) == 3  # type: ignore[arg-type]

    def test_invariant_rng_state_returns_dict(self):
        """rng_state() always returns a dict."""
        from lighttrain.utils.seed import rng_state
        state = rng_state()
        assert isinstance(state, dict)

    def test_invariant_restore_empty_dict_is_noop(self):
        """restore_rng_state({}) completes without error (all branches skipped)."""
        from lighttrain.utils.seed import restore_rng_state
        restore_rng_state({})

    def test_invariant_restore_unknown_keys_ignored(self):
        """restore_rng_state with unknown keys beyond 'python'/'numpy'/'torch'
        does not raise (they are simply not checked).
        """
        from lighttrain.utils.seed import restore_rng_state
        restore_rng_state({"unknown_key": "some_value"})

    def test_invariant_pythonhashseed_set_on_first_call(self, monkeypatch):
        """PYTHONHASHSEED env-var is set by seed_everything via setdefault."""
        import os

        from lighttrain.utils.seed import seed_everything
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        seed_everything(55)
        assert os.environ.get("PYTHONHASHSEED") == "55"

    def test_pin_pythonhashseed_not_overwritten_if_already_set(self, monkeypatch):
        """Pin: os.environ.setdefault means pre-existing PYTHONHASHSEED is kept."""
        import os

        from lighttrain.utils.seed import seed_everything
        monkeypatch.setenv("PYTHONHASHSEED", "999")
        seed_everything(1)
        assert os.environ["PYTHONHASHSEED"] == "999"
