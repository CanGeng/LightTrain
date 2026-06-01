"""Frontier generative-objective plugins.

Opt-in, specific generative paradigms (the basic ``next_token`` /
``masked_denoising`` objectives stay in ``lighttrain.objectives``). Registered
via auto-discovery; import the submodule you need:

    from lighttrain.plugins.objectives.diffusion import DiffusionObjective
    from lighttrain.plugins.objectives.flow_matching import FlowMatchingObjective
    from lighttrain.plugins.objectives.jepa import JEPAObjective
"""
