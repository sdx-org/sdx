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


def test_differential_uses_output_language_instruction_and_utf8_json(chat_spy):
    """
    title: >-
      Differential should use English task instructions, a localized output-
      language instruction, and UTF-8 payload encoding.
    parameters:
      chat_spy:
        description: Value for chat_spy.
    """
    patient = {'symptoms': 'dor no coração'}

    out = diag_mod.differential(patient, language='pt', session_id='abc')

    assert out.summary == 'done'
    assert 'Return a JSON object with keys' in str(chat_spy[0]['system'])
    assert 'Write all natural-language string values in Portuguese.' in str(
        chat_spy[0]['system']
    )
    assert chat_spy[0]['session_id'] == 'abc'
    assert '"dor no coração"' in str(chat_spy[0]['user'])
    assert '\\u00e7' not in str(chat_spy[0]['user'])


def test_differential_falls_back_to_english_output(chat_spy):
    """
    title: Unknown language should fallback to English output instructions.
    parameters:
      chat_spy:
        description: Value for chat_spy.
    """
    diag_mod.differential({'age': 40}, language='xx')
    assert 'Write all natural-language string values in English.' in str(
        chat_spy[0]['system']
    )


def test_exams_uses_output_language_instruction_and_json_array(chat_spy):
    """
    title: >-
      Exam suggestions should encode selected diagnoses as JSON list and
      request localized natural-language output.
    parameters:
      chat_spy:
        description: Value for chat_spy.
    """
    selected = ['Condition A', 'Condition B']

    diag_mod.exams(selected, language='es', session_id='sid-2')

    assert 'Given the selected diagnoses' in str(chat_spy[0]['system'])
    assert 'Write all natural-language string values in Spanish.' in str(
        chat_spy[0]['system']
    )
    assert chat_spy[0]['user'] == json.dumps(selected, ensure_ascii=False)
    assert chat_spy[0]['session_id'] == 'sid-2'


def test_exams_falls_back_to_english_output(chat_spy):
    """
    title: Unknown language should fallback to English output instructions.
    parameters:
      chat_spy:
        description: Value for chat_spy.
    """
    diag_mod.exams(['A'], language='zz')
    assert 'Write all natural-language string values in English.' in str(
        chat_spy[0]['system']
    )
