"""
title: Tests for LLM output safety guardrails (agents/safety).
summary: |-
  Unit tests use monkeypatching — no HF model required.
  Integration tests (marked @pytest.mark.hf) require RUN_HF_TESTS=1 and
  network access to download the classification model.
"""

from __future__ import annotations

import os

from unittest.mock import MagicMock, patch

import hiperhealth.agents.client as client_mod
import pytest

from hiperhealth.agents.safety.topic_guard import (
    DEFAULT_BANNED_TOPICS,
    DEFAULT_THRESHOLD,
    UnsafeOutputError,
    _env_banned_topics,
    _env_float,
    check_output_safety,
    detect_banned_topics,
)
from hiperhealth.schema.clinical_outputs import LLMDiagnosis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_RUN_HF = os.getenv('RUN_HF_TESTS', '').strip().lower() in {'1', 'true', 'yes'}
_hf_only = pytest.mark.skipif(
    not _RUN_HF,
    reason='Requires HF model/network. Set RUN_HF_TESTS=1 to enable.',
)


def _diagnosis(
    summary: str, options: list[str] | None = None
) -> LLMDiagnosis:
    """
    title: Build a minimal LLMDiagnosis fixture for testing.
    parameters:
      summary:
        type: str
        description: Patient summary text.
      options:
        type: list[str] | None
        description: Differential options; defaults to ['option-a'].
    returns:
      type: LLMDiagnosis
      description: Constructed LLMDiagnosis fixture.
    """
    return LLMDiagnosis(summary=summary, options=options or ['option-a'])


# ---------------------------------------------------------------------------
# UnsafeOutputError
# ---------------------------------------------------------------------------
def test_unsafe_output_error_carries_hits() -> None:
    """
    title: UnsafeOutputError should store hits and include message in str.
    """
    hits = [('medication dosing advice', 0.92)]
    err = UnsafeOutputError('blocked', hits=hits)
    assert err.hits == hits
    assert 'blocked' in str(err)


# ---------------------------------------------------------------------------
# _env_float
# ---------------------------------------------------------------------------
def test_env_float_returns_default_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    title: _env_float returns the default when env var is absent.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
        description: Value for monkeypatch.
    """
    monkeypatch.delenv('HIPERHEALTH_SAFETY_THRESHOLD', raising=False)
    result = _env_float('HIPERHEALTH_SAFETY_THRESHOLD', DEFAULT_THRESHOLD)
    assert result == DEFAULT_THRESHOLD


def test_env_float_parses_valid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    title: _env_float parses a valid float string from the environment.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
        description: Value for monkeypatch.
    """
    monkeypatch.setenv('HIPERHEALTH_SAFETY_THRESHOLD', '0.6')
    assert _env_float('HIPERHEALTH_SAFETY_THRESHOLD', DEFAULT_THRESHOLD) == 0.6


def test_env_float_falls_back_on_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    title: _env_float falls back to default when env var is not a number.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
        description: Value for monkeypatch.
    """
    monkeypatch.setenv('HIPERHEALTH_SAFETY_THRESHOLD', 'not-a-float')
    result = _env_float('HIPERHEALTH_SAFETY_THRESHOLD', DEFAULT_THRESHOLD)
    assert result == DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# _env_banned_topics
# ---------------------------------------------------------------------------
def test_env_banned_topics_falls_back_to_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    title: _env_banned_topics returns defaults when env var is absent.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
        description: Value for monkeypatch.
    """
    monkeypatch.delenv('HIPERHEALTH_SAFETY_BANNED_TOPICS', raising=False)
    assert _env_banned_topics() == DEFAULT_BANNED_TOPICS


def test_env_banned_topics_reads_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    title: _env_banned_topics parses semicolon-separated topics from env.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
        description: Value for monkeypatch.
    """
    monkeypatch.setenv('HIPERHEALTH_SAFETY_BANNED_TOPICS', 'topic-a;topic-b')
    result = _env_banned_topics()
    assert result == ('topic-a', 'topic-b')


def test_env_banned_topics_ignores_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    title: _env_banned_topics falls back to defaults when env var is empty.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
        description: Value for monkeypatch.
    """
    monkeypatch.setenv('HIPERHEALTH_SAFETY_BANNED_TOPICS', '')
    assert _env_banned_topics() == DEFAULT_BANNED_TOPICS


# ---------------------------------------------------------------------------
# detect_banned_topics — unit (mocked classifier)
# ---------------------------------------------------------------------------
def _make_mock_classifier(
    labels: list[str], scores: list[float]
) -> MagicMock:
    """
    title: Build a mock classifier returning fixed labels and scores.
    parameters:
      labels:
        type: list[str]
        description: Classification labels to return.
      scores:
        type: list[float]
        description: Corresponding confidence scores to return.
    returns:
      type: MagicMock
      description: Configured mock classifier callable.
    """
    mock = MagicMock()
    mock.return_value = {'labels': labels, 'scores': scores}
    return mock


def test_detect_banned_topics_returns_empty_when_all_below_threshold() -> (
    None
):
    """
    title: detect_banned_topics returns [] when all scores are below threshold.
    """
    labels = ['medication dosing advice', 'violent threats']
    scores = [0.1, 0.2]
    mock_clf = _make_mock_classifier(labels, scores)
    with patch(
        'hiperhealth.agents.safety.topic_guard.create_topic_classifier',
        return_value=mock_clf,
    ):
        result = detect_banned_topics('benign text', threshold=0.8)
    assert result == []


def test_detect_banned_topics_returns_hits_above_threshold() -> None:
    """
    title: detect_banned_topics returns only labels that exceed the threshold.
    """
    labels = ['medication dosing advice', 'violent threats']
    scores = [0.95, 0.3]
    mock_clf = _make_mock_classifier(labels, scores)
    with patch(
        'hiperhealth.agents.safety.topic_guard.create_topic_classifier',
        return_value=mock_clf,
    ):
        result = detect_banned_topics(
            'take 500mg every 4 hours', threshold=0.8
        )
    assert len(result) == 1
    assert result[0][0] == 'medication dosing advice'
    assert result[0][1] == pytest.approx(0.95)


def test_detect_banned_topics_all_hits_when_all_above_threshold() -> None:
    """
    title: detect_banned_topics returns all labels when all exceed threshold.
    """
    labels = ['medication dosing advice', 'violent threats']
    scores = [0.9, 0.85]
    mock_clf = _make_mock_classifier(labels, scores)
    with patch(
        'hiperhealth.agents.safety.topic_guard.create_topic_classifier',
        return_value=mock_clf,
    ):
        result = detect_banned_topics('unsafe text', threshold=0.8)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# check_output_safety — unit (mocked detect_banned_topics)
# ---------------------------------------------------------------------------
def test_check_output_safety_passes_for_harmless_output() -> None:
    """
    title: check_output_safety does not raise when no banned topic is found.
    """
    obj = _diagnosis('Possible viral infection.')
    with patch(
        'hiperhealth.agents.safety.topic_guard.detect_banned_topics',
        return_value=[],
    ):
        check_output_safety(obj)  # must not raise


def test_check_output_safety_raises_unsafe_output_error_on_hit() -> None:
    """
    title: check_output_safety raises UnsafeOutputError when a hit is found.
    """
    obj = _diagnosis('Take 500mg ibuprofen every 4 hours.')
    hits = [('medication dosing advice', 0.91)]
    with patch(
        'hiperhealth.agents.safety.topic_guard.detect_banned_topics',
        return_value=hits,
    ):
        with pytest.raises(UnsafeOutputError) as exc_info:
            check_output_safety(obj)
    assert exc_info.value.hits == hits
    assert 'medication dosing advice' in str(exc_info.value)


def test_check_output_safety_skips_empty_combined_text() -> None:
    """
    title: >-
      check_output_safety skips classification when combined text is empty.
    """
    obj = LLMDiagnosis(summary='', options=[])
    called = []
    with patch(
        'hiperhealth.agents.safety.topic_guard.detect_banned_topics',
        side_effect=lambda *a, **kw: called.append(True) or [],
    ):
        check_output_safety(obj)
    assert called == []


def test_check_output_safety_handles_dict_options() -> None:
    """
    title: check_output_safety handles dict-style options without raising.
    """
    obj = LLMDiagnosis(summary='Assessment', options={'pancreatitis': 0.7})
    with patch(
        'hiperhealth.agents.safety.topic_guard.detect_banned_topics',
        return_value=[],
    ):
        check_output_safety(obj)  # must not raise


# ---------------------------------------------------------------------------
# client integration — safety guard enabled/disabled
# ---------------------------------------------------------------------------
class _FakeLLM:
    """
    title: Minimal LLM double that returns a fixed LLMDiagnosis.
    attributes:
      result:
        description: Fixed diagnosis to return from generate().
    """

    def __init__(self, result: LLMDiagnosis) -> None:
        """
        title: Initialise with the fixed diagnosis to return.
        parameters:
          result:
            type: LLMDiagnosis
            description: Diagnosis to return from generate().
        """
        self.result = result

    def generate(
        self, system: str, user: str, output_type: type
    ) -> LLMDiagnosis:
        """
        title: Return the fixed result regardless of inputs.
        parameters:
          system:
            type: str
            description: System prompt (ignored).
          user:
            type: str
            description: User message (ignored).
          output_type:
            type: type
            description: Expected output type (ignored).
        returns:
          type: LLMDiagnosis
          description: Fixed diagnosis result.
        """
        return self.result


def test_chat_skips_safety_guard_when_env_not_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """
    title: chat() should not invoke the safety guard when env var is unset.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
        description: Value for monkeypatch.
      tmp_path:
        type: pytest.TempPathFactory
        description: Value for tmp_path.
    """
    monkeypatch.delenv('HIPERHEALTH_SAFETY_ENABLED', raising=False)
    monkeypatch.setattr(client_mod, '_RAW_DIR', tmp_path)

    fake_llm = _FakeLLM(LLMDiagnosis(summary='ok', options=['a']))
    monkeypatch.setattr(client_mod, 'dump_llm_json', lambda *_: None)

    guard_called = []
    with patch(
        'hiperhealth.agents.safety.topic_guard.check_output_safety',
        side_effect=lambda obj: guard_called.append(obj),
    ):
        client_mod.chat('sys', 'usr', llm=fake_llm)

    assert guard_called == []


def test_chat_applies_safety_guard_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """
    title: chat() should invoke _check_safety once when env var is set to 1.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
        description: Value for monkeypatch.
      tmp_path:
        type: pytest.TempPathFactory
        description: Value for tmp_path.
    """
    monkeypatch.setenv('HIPERHEALTH_SAFETY_ENABLED', '1')
    monkeypatch.setattr(client_mod, '_RAW_DIR', tmp_path)

    result = LLMDiagnosis(summary='ok', options=['a'])
    fake_llm = _FakeLLM(result)
    monkeypatch.setattr(client_mod, 'dump_llm_json', lambda *_: None)

    guard_called = []

    with patch(
        'hiperhealth.agents.client._check_safety',
        side_effect=lambda obj: guard_called.append(obj),
    ):
        client_mod.chat('sys', 'usr', llm=fake_llm)

    assert len(guard_called) == 1
    assert guard_called[0] is result


def test_chat_safety_blocks_before_persist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """
    title: chat() should raise UnsafeOutputError before persisting output.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
        description: Value for monkeypatch.
      tmp_path:
        type: pytest.TempPathFactory
        description: Value for tmp_path.
    """
    monkeypatch.setenv('HIPERHEALTH_SAFETY_ENABLED', '1')
    monkeypatch.setattr(client_mod, '_RAW_DIR', tmp_path)

    fake_llm = _FakeLLM(
        LLMDiagnosis(summary='Take 500mg every 4h', options=['dosing'])
    )
    dump_called = []
    monkeypatch.setattr(
        client_mod, 'dump_llm_json', lambda *_: dump_called.append(True)
    )

    with patch(
        'hiperhealth.agents.client._check_safety',
        side_effect=UnsafeOutputError(
            'blocked', hits=[('medication dosing advice', 0.95)]
        ),
    ):
        with pytest.raises(UnsafeOutputError):
            client_mod.chat('sys', 'usr', llm=fake_llm)

    assert dump_called == []


# ---------------------------------------------------------------------------
# HF integration tests — only run when RUN_HF_TESTS=1
# ---------------------------------------------------------------------------
@_hf_only
@pytest.mark.hf
@pytest.mark.integration
def test_hf_detect_banned_topics_blocks_dosing_advice() -> None:
    """
    title: HF model should flag explicit medication dosing advice as unsafe.
    """
    text = (
        'You should take 500mg of ibuprofen every 4 hours and '
        'no more than 3200mg per day.'
    )
    hits = detect_banned_topics(
        text,
        banned_topics=('medication dosing advice', 'self-harm'),
        threshold=0.5,
    )
    labels = [lbl for lbl, _ in hits]
    assert 'medication dosing advice' in labels


@_hf_only
@pytest.mark.hf
@pytest.mark.integration
def test_hf_detect_banned_topics_allows_harmless_text() -> None:
    """
    title: HF model should not flag innocuous clinical text as unsafe.
    """
    text = 'The patient presents with mild upper respiratory symptoms.'
    hits = detect_banned_topics(
        text,
        banned_topics=('medication dosing advice', 'self-harm'),
        threshold=0.8,
    )
    assert hits == []
