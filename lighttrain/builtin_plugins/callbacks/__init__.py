"""Callback implementations.

The ``CallbackProtocol`` + ``EventBus`` (``Signal`` / ``CALLBACK_EVENTS``) stay in
``lighttrain.callbacks.base``; the concrete builtin callbacks + the invariants
callback are registered impls here (DESIGN §3.3), picked up by auto-discovery:

    from lighttrain.builtin_plugins.callbacks.builtins.ema import EMACallback
"""
