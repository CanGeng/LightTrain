"""Fixtures specific to distributed-context tests.

The top-level ``tests/conftest.py`` carries trainer-oriented fixtures whose
names (``fake_dist_env``, ``tiny_model``) refer to different surfaces. To
avoid colliding with those, this conftest exposes ``dist_mock`` — a factory
that monkeypatches the ``torch.distributed`` symbols used by
``lighttrain.distributed._context`` so the manual group helpers run without
NCCL/gloo and record every call.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import pytest


@pytest.fixture
def dist_mock(monkeypatch) -> Callable[..., SimpleNamespace]:
    """Factory that installs a stand-in ``torch.distributed`` for a single test.

    Usage::

        handle = dist_mock(rank=2, world_size=8,
                           dp_group_global_ranks=[2, 3, 6, 7])
        # ... call _create_ep_groups / _create_groups_manual ...
        assert handle.calls == [[2, 3], [6, 7]]

    The returned ``handle`` exposes:
        - ``handle.calls``        list[list[int]] — members of every new_group call
        - ``handle.dp_group``     sentinel object whose ``get_process_group_ranks``
                                  returns ``dp_group_global_ranks``
        - ``handle.fake``         the SimpleNamespace patched onto torch.distributed
    """

    def _install(
        *,
        rank: int,
        world_size: int,
        dp_group_global_ranks: list[int] | None = None,
    ) -> SimpleNamespace:
        calls: list[list[int]] = []
        returns: list[object] = []

        dp_members = tuple(dp_group_global_ranks or ())
        dp_sentinel = SimpleNamespace(_members=dp_members, _kind="dp_sentinel")

        def _new_group(members):
            members_list = list(members)
            calls.append(members_list)
            obj = SimpleNamespace(_members=tuple(members_list))
            returns.append(obj)
            return obj

        def _get_process_group_ranks(group):
            if group is dp_sentinel:
                return list(dp_members)
            mem = getattr(group, "_members", None)
            if mem is None:
                raise RuntimeError(
                    "dist_mock.get_process_group_ranks: unknown group object"
                )
            return list(mem)

        def _get_rank() -> int:
            return rank

        def _get_world_size() -> int:
            return world_size

        for name, val in (
            ("new_group", _new_group),
            ("get_process_group_ranks", _get_process_group_ranks),
            ("get_rank", _get_rank),
            ("get_world_size", _get_world_size),
            ("is_initialized", lambda: True),
            ("init_process_group", lambda **_kw: None),
        ):
            monkeypatch.setattr(f"torch.distributed.{name}", val)

        return SimpleNamespace(
            calls=calls,
            new_group_returns=returns,
            dp_group=dp_sentinel,
            fake=SimpleNamespace(),
        )

    return _install
