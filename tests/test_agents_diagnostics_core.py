"""
title: Unit tests for diagnostics prompts and payload encoding.
"""

from __future__ import annotations

import json

import hiperhealth.skills.diagnostics.core as diag_mod
import pytest

from hiperhealth.schema.clinical_outputs import LLMDiagnosis


@pytest.fixture
def chat_spy(monkeypatch):
    """
    title: Capture calls made to diagnostics chat backend.
    parameters:
      monkeypatch:
        description: Value for monkeypatch.
    """
    calls: list[dict[str, str | None]] = []

    def _fake_chat(system: str, user: str, *, session_id: str | None = None):
        calls.append(
            {'system': system, 'user': user, 'session_id': session_id}
        )
        return LLMDiagnosis(summary='done', options=['x'])

    monkeypatch.setattr(diag_mod, 'chat', _fake_chat)
    return calls


def test_differential_uses_language_prompt_and_utf8_json(chat_spy):
    """
    title: >-
      Differential should pick requested language prompt and UTF-8 payload.
    parameters:
      chat_spy:
        description: Value for chat_spy.
    """
    patient = {'symptoms': 'dor no coração'}

    out = diag_mod.differential(patient, language='pt', session_id='abc')

    assert out.summary == 'done'
    assert chat_spy[0]['system'] == diag_mod._DIAG_PROMPTS['pt']
    assert chat_spy[0]['session_id'] == 'abc'
    assert '"dor no coração"' in str(chat_spy[0]['user'])
    assert '\\u00e7' not in str(chat_spy[0]['user'])


def test_differential_falls_back_to_english_prompt(chat_spy):
    """
    title: Unknown language should fallback to English diagnosis prompt.
    parameters:
      chat_spy:
        description: Value for chat_spy.
    """
    diag_mod.differential({'age': 40}, language='xx')
    assert chat_spy[0]['system'] == diag_mod._DIAG_PROMPTS['en']


def test_exams_uses_language_prompt_and_json_array(chat_spy):
    """
    title: Exam suggestions should encode selected diagnoses as JSON list.
    parameters:
      chat_spy:
        description: Value for chat_spy.
    """
    selected = ['Condition A', 'Condition B']

    diag_mod.exams(selected, language='es', session_id='sid-2')

    assert chat_spy[0]['system'] == diag_mod._EXAM_PROMPTS['es']
    assert chat_spy[0]['user'] == json.dumps(selected, ensure_ascii=False)
    assert chat_spy[0]['session_id'] == 'sid-2'


def test_exams_falls_back_to_english_prompt(chat_spy):
    """
    title: Unknown language should fallback to English exam prompt.
    parameters:
      chat_spy:
        description: Value for chat_spy.
    """
    diag_mod.exams(['A'], language='zz')
    assert chat_spy[0]['system'] == diag_mod._EXAM_PROMPTS['en']
