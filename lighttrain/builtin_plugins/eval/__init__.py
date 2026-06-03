"""Evaluation implementations.

The eval *framework* (``Evaluator`` / ``EvalTask`` / ``EvalReport`` +
``generation_eval`` + ``metrics``) stays in ``lighttrain.eval``; the concrete
``RegressionGate`` (registered under the ``invariant`` category) lives here
(DESIGN §3.3) and is picked up by auto-discovery.
"""
