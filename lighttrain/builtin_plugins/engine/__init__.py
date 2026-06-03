"""Engine implementations.

The ``EngineProtocol`` + ``StepContext`` plumbing stay in ``lighttrain.engine``;
the concrete ``StandardEngine`` lives here (DESIGN §3.3) and is picked up by
auto-discovery (``from lighttrain.builtin_plugins.engine.standard import StandardEngine``).
"""
