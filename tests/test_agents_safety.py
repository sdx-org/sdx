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

import pytest

import hiperhealth.agents.client as client_mod

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


def _diagnosis(summary: str, options: list[str] | None = None) -> LLMDiagnosis:
    return LLMDiagnosis(summary=summary, options=options or ['option-a'])


# ---------------------------------------------------------------------------
# UnsafeOutputError
# ---------------------------------------------------------------------------
def test_unsafe_output_error_carries_hits() -> None:
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
    monkeypatch.delenv('HIPERHEALTH_SAFETY_THRESHOLD', raising=False)
    result = _env_float('HIPERHEALTH_SAFETY_THRESHOLD', DEFAULT_THRESHOLD)
    assert result == DEFAULT_THRESHOLD


def test_env_float_parses_valid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HIPERHEALTH_SAFETY_THRESHOLD', '0.6')
    assert _env_float('HIPERHEALTH_SAFETY_THRESHOLD', DEFAULT_THRESHOLD) == 0.6


def test_env_float_falls_back_on_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('HIPERHEALTH_SAFETY_THRESHOLD', 'not-a-float')
    result = _env_float('HIPERHEALTH_SAFETY_THRESHOLD', DEFAULT_THRESHOLD)
    assert result == DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# _env_banned_topics
# ---------------------------------------------------------------------------
def test_env_banned_topics_falls_back_to_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('HIPERHEALTH_SAFETY_BANNED_TOPICS', raising=False)
    assert _env_banned_topics() == DEFAULT_BANNED_TOPICS


def test_env_banned_topics_reads_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('HIPERHEALTH_SAFETY_BANNED_TOPICS', 'topic-a;topic-b')
    result = _env_banned_topics()
    assert result == ('topic-a', 'topic-b')


def test_env_banned_topics_ignores_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('HIPERHEALTH_SAFETY_BANNED_TOPICS', '')
    assert _env_banned_topics() == DEFAULT_BANNED_TOPICS


# ---------------------------------------------------------------------------
# detect_banned_topics — unit (mocked classifier)
# ---------------------------------------------------------------------------
def _make_mock_classifier(labels: list[str], scores: list[float]) -> MagicMock:
    mock = MagicMock()
    mock.return_value = {'labels': labels, 'scores': scores}
    return mock


def test_detect_banned_topics_returns_empty_when_all_below_threshold() -> None:
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
    obj = _diagnosis('Possible viral infection.')
    with patch(
        'hiperhealth.agents.safety.topic_guard.detect_banned_topics',
        return_value=[],
    ):
        check_output_safety(obj)  # must not raise


def test_check_output_safety_raises_unsafe_output_error_on_hit() -> None:
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
    obj = LLMDiagnosis(summary='', options=[])
    called = []
    with patch(
        'hiperhealth.agents.safety.topic_guard.detect_banned_topics',
        side_effect=lambda *a, **kw: called.append(True) or [],
    ):
        check_output_safety(obj)
    assert called == []


def test_check_output_safety_handles_dict_options() -> None:
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
    def __init__(self, result: LLMDiagnosis) -> None:
        self.result = result

    def generate(
        self, system: str, user: str, output_type: type
    ) -> LLMDiagnosis:
        return self.result


def test_chat_skips_safety_guard_when_env_not_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
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
    text = 'The patient presents with mild upper respiratory symptoms.'
    hits = detect_banned_topics(
        text,
        banned_topics=('medication dosing advice', 'self-harm'),
        threshold=0.8,
    )
    assert hits == []
