"""PrepGraph node implementations.

The PrepGraph *framework* (``PrepGraph`` / ``PrepRunner`` / ``PrepNode`` base /
fingerprinting / IO) stays in ``lighttrain.data.prepgraph``; the concrete
``@register("prep_node", ...)`` nodes live here (DESIGN §3.3) and are picked up
by auto-discovery.
"""
