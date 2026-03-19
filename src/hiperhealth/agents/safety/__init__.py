"""
title: Safety sub-package for LLM output guardrails.
"""

from hiperhealth.agents.safety.topic_guard import (
    DEFAULT_BANNED_TOPICS,
    DEFAULT_HYP_TEMPLATE,
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    THRESHOLD_MAX,
    UnsafeOutputError,
    check_output_safety,
    detect_banned_topics,
)

__all__ = [
    'DEFAULT_BANNED_TOPICS',
    'DEFAULT_HYP_TEMPLATE',
    'DEFAULT_MODEL',
    'DEFAULT_THRESHOLD',
    'THRESHOLD_MAX',
    'UnsafeOutputError',
    'check_output_safety',
    'detect_banned_topics',
]
