"""Objective extensions.

The ``objective`` abstraction (``ObjectiveProfile`` in
``lighttrain.optim.architectures.profile``) is for non-standard training objectives —
the default LM/MLM paths use ``loss:``, not ``objective:``, so *all* concrete
objective implementations live here (DESIGN §3.3: protocol in core, impls in
frontier). Registered via auto-discovery; import the submodule you need:

    from lighttrain.builtin_plugins.objectives.next_token import NextTokenObjective
    from lighttrain.builtin_plugins.objectives.masked_denoising import MaskedDenoisingObjective
    from lighttrain.builtin_plugins.objectives.diffusion import DiffusionObjective
    from lighttrain.builtin_plugins.objectives.flow_matching import FlowMatchingObjective
    from lighttrain.builtin_plugins.objectives.jepa import JEPAObjective
"""
