"""Diagnostic callback implementations (registered, opt-in).

The ``@register("callback", ...)`` diagnostics (grad_flow / dead_neuron /
nan_hunter / loss_attribution / sample_preview) live here; the non-registered
diagnostic *helpers* consumed by the core trainer loop (crash_bundle /
oom_report / index_page / callback_isolation / frozen_step / nan_repro) stay in
``lighttrain.observability.diagnostics`` (DESIGN §3.3).
"""
