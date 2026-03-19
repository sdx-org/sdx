"""
title: Zero-shot MNLI topic guard for LLM output safety.
summary: |-
  Detects banned topics (self-harm, violent threats, dosing advice, etc.)
  via zero-shot classification and raises UnsafeOutputError before the
  response is persisted or returned from agents.client.chat.
"""

from __future__ import annotations

import os

from functools import lru_cache
from typing import Any

from hiperhealth.schema.clinical_outputs import LLMDiagnosis

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = 'facebook/bart-large-mnli'
DEFAULT_THRESHOLD: float = 0.8
THRESHOLD_MAX: float = 1.0
DEFAULT_HYP_TEMPLATE: str = 'This text is about {}.'
DEFAULT_BANNED_TOPICS: tuple[str, ...] = (
    'self-harm or suicide instructions',
    'violent threats',
    'medication dosing advice',
    'step-by-step treatment plans',
    'illegal drug use',
)

_ENV_PREFIX = 'HIPERHEALTH_SAFETY_'


class UnsafeOutputError(ValueError):
    """
    title: Raised when LLM output contains a banned topic.
    attributes:
      hits:
        type: list[tuple[str, float]]
        description: Detected (label, score) pairs that exceeded the threshold.
    """

    def __init__(
        self, message: str, hits: list[tuple[str, float]]
    ) -> None:
        super().__init__(message)
        self.hits = hits


def _get_device_id() -> int:
    """Return GPU device id (0) when CUDA is available, else CPU (-1)."""
    try:
        import torch  # type: ignore[import-untyped]

        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


def _env_float(name: str, default: float) -> float:
    """Parse a float env var; fall back to *default* if missing or invalid."""
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_banned_topics() -> tuple[str, ...]:
    """Return banned topics from env or fall back to defaults."""
    raw = os.getenv(f'{_ENV_PREFIX}BANNED_TOPICS', '')
    if not raw.strip():
        return DEFAULT_BANNED_TOPICS
    parsed = tuple(t.strip() for t in raw.split(';') if t.strip())
    return parsed or DEFAULT_BANNED_TOPICS


@lru_cache(maxsize=1)
def create_topic_classifier(model_name: str) -> Any:
    """
    title: Create and cache a zero-shot classification pipeline.
    parameters:
      model_name:
        type: str
        description: HuggingFace model name for zero-shot classification.
    returns:
      type: Any
      description: transformers zero-shot-classification pipeline.
    """
    from transformers import pipeline  # type: ignore[import-untyped]

    return pipeline(
        'zero-shot-classification',
        model=model_name,
        device=_get_device_id(),
    )


def detect_banned_topics(
    text: str,
    banned_topics: tuple[str, ...] | None = None,
    *,
    threshold: float | None = None,
    model_name: str | None = None,
    hypothesis_template: str | None = None,
) -> list[tuple[str, float]]:
    """
    title: Return (label, score) pairs that exceed the safety threshold.
    parameters:
      text:
        type: str
        description: Text to classify.
      banned_topics:
        type: tuple[str, ...] | None
        description: Labels to test against; defaults to DEFAULT_BANNED_TOPICS.
      threshold:
        type: float | None
        description: Confidence cut-off; env HIPERHEALTH_SAFETY_THRESHOLD.
      model_name:
        type: str | None
        description: HF model name; env HIPERHEALTH_SAFETY_MODEL.
      hypothesis_template:
        type: str | None
        description: NLI hypothesis template; env HIPERHEALTH_SAFETY_HYP_TEMPLATE.
    returns:
      type: list[tuple[str, float]]
      description: Detected hits above threshold.
    """
    effective_topics = banned_topics or _env_banned_topics()
    effective_threshold = (
        threshold
        if threshold is not None
        else _env_float(f'{_ENV_PREFIX}THRESHOLD', DEFAULT_THRESHOLD)
    )
    effective_model = model_name or os.getenv(
        f'{_ENV_PREFIX}MODEL', DEFAULT_MODEL
    )
    effective_template = hypothesis_template or os.getenv(
        f'{_ENV_PREFIX}HYP_TEMPLATE', DEFAULT_HYP_TEMPLATE
    )

    classifier = create_topic_classifier(effective_model)
    output = classifier(
        text,
        candidate_labels=list(effective_topics),
        hypothesis_template=effective_template,
    )

    return [
        (label, score)
        for label, score in zip(output['labels'], output['scores'])
        if score >= effective_threshold
    ]


def check_output_safety(obj: LLMDiagnosis) -> None:
    """
    title: Validate *obj* against banned topics; raise UnsafeOutputError on hit.
    summary: |-
      Combines summary and options into a single text block, runs zero-shot
      MNLI classification, and raises UnsafeOutputError if any banned topic
      exceeds the configured threshold.
    parameters:
      obj:
        type: LLMDiagnosis
        description: Validated LLM output to inspect.
    raises:
      UnsafeOutputError: When a banned topic is detected.
    """
    if isinstance(obj.options, dict):
        options_parts: list[str] = list(obj.options.keys())
    else:
        options_parts = list(obj.options or [])

    options_text = '\n'.join(str(p) for p in options_parts)
    combined = f'{obj.summary}\n{options_text}'.strip()
    if not combined:
        return

    hits = detect_banned_topics(combined)
    if hits:
        detail = ', '.join(f'{lbl}:{score:.2f}' for lbl, score in hits)
        raise UnsafeOutputError(
            f'Unsafe LLM output blocked — banned topics detected: {detail}',
            hits=hits,
        )
