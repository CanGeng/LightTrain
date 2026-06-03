"""Trainer implementations.

The abstract ``Trainer`` base + shared ``_primitives`` / ``_utils`` plumbing stay
in ``lighttrain.trainers``; the concrete per-paradigm trainers
(pretrain / preference / ppo / grpo / reward_model) live here (DESIGN §3.3) and
are picked up by auto-discovery:

    from lighttrain.builtin_plugins.trainers.pretrain import PretrainTrainer
"""
