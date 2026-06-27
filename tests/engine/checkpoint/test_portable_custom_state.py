"""P2a (issue #2b): custom optimizer state should be self-contained.

An optimizer that stashes a custom *object* (e.g. GaLore's ``GaLoreProjector``)
in ``optimizer.state[p]`` pickles it by reference — the checkpoint then
hard-depends on that class being importable at load time, so a vanilla process
(or ``doctor`` / ``convert-checkpoint``, which don't import ``user_modules``
first) trips on it.

The portable pattern (which the GaLore adapter follows) is to serialize the
custom state as **plain tensors + scalars** in ``state_dict()`` and rebuild the
object in ``load_state_dict()``. This test exercises that pattern with a stand-in
projector so it runs without ``galore_torch``: it proves the serialized form
carries no reference to the custom class, yet round-trips losslessly.
"""

from __future__ import annotations

import pickle

import torch

from lighttrain.engine.checkpoint.manager import CheckpointManager


class _Projector:
    """Stand-in for a custom non-tensor optimizer-state object."""

    def __init__(self, ortho_matrix: torch.Tensor, rank: int, scale: float) -> None:
        self.ortho_matrix = ortho_matrix
        self.rank = rank
        self.scale = scale


def _portable_state_dict(raw: dict) -> dict:
    """Mirror the GaLore adapter: replace _Projector objects with plain dicts."""
    sd = {"state": {}, "param_groups": raw["param_groups"]}
    for pid, st in raw["state"].items():
        st2 = dict(st)
        proj = st2.get("projector")
        if isinstance(proj, _Projector):
            st2["projector"] = {
                "__projector__": True,
                "ortho_matrix": proj.ortho_matrix,
                "rank": proj.rank,
                "scale": proj.scale,
            }
        sd["state"][pid] = st2
    return sd


def _rebuild(sd: dict) -> dict:
    for st in sd["state"].values():
        proj = st.get("projector")
        if isinstance(proj, dict) and proj.get("__projector__"):
            st["projector"] = _Projector(proj["ortho_matrix"], proj["rank"], proj["scale"])
    return sd


def _contains_instance(obj, cls) -> bool:
    if isinstance(obj, cls):
        return True
    if isinstance(obj, dict):
        return any(_contains_instance(v, cls) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_instance(v, cls) for v in obj)
    return False


def test_portable_state_dict_has_no_custom_class_reference():
    raw = {
        "state": {
            0: {"exp_avg": torch.randn(8, 4), "projector": _Projector(torch.randn(4, 8), rank=4, scale=0.25)},
        },
        "param_groups": [{"lr": 1e-3}],
    }
    # The naive form embeds the custom object (non-portable).
    assert _contains_instance(raw, _Projector)

    sd = _portable_state_dict(raw)
    # The portable form is plain tensors/scalars — no _Projector anywhere.
    assert not _contains_instance(sd, _Projector)
    # And its pickle contains no reference to the class' module path.
    blob = pickle.dumps(sd)
    assert b"_Projector" not in blob


def test_portable_state_round_trips_through_checkpoint_manager(tmp_path):
    mgr = CheckpointManager(tmp_path, keep_last_n=2)
    ortho = torch.randn(4, 8)
    raw = {
        "state": {0: {"exp_avg": torch.randn(8, 4), "projector": _Projector(ortho, rank=4, scale=0.25)}},
        "param_groups": [{"lr": 1e-3}],
    }
    target = mgr.save(step=1, state={"optimizer": _portable_state_dict(raw)}, kind="step")
    assert target is not None

    loaded = _rebuild(mgr.load(target)["optimizer"])
    proj = loaded["state"][0]["projector"]
    assert isinstance(proj, _Projector)
    assert proj.rank == 4 and proj.scale == 0.25
    torch.testing.assert_close(proj.ortho_matrix, ortho)
