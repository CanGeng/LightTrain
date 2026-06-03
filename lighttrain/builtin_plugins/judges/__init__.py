"""Frontier judge extensions.

Opt-in judge implementations (the judge Protocol stays in
``lighttrain.protocols``; the runtime resolves judges via the ``judge``
registry category). Registered via auto-discovery; import what you need:

    from lighttrain.builtin_plugins.judges.judge import VerifierJudge, PairwiseLLMJudge
"""
