"""Unit tests for shared structured-LLM client helpers."""

from __future__ import annotations

from types import SimpleNamespace

import hiperhealth.agents.client as client_mod
import pytest

from hiperhealth.schema.clinical_outputs import LLMDiagnosis


class _FakeLLM:
    """Minimal structured LLM double used by client tests."""

    def __init__(self, result: LLMDiagnosis) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def generate(self, system: str, user: str, output_type):
        """Return a fixed validated payload and capture call metadata."""
        self.calls.append(
            {'system': system, 'user': user, 'output_type': output_type}
        )
        return self.result


def test_dump_llm_json_uses_given_session_id(tmp_path, monkeypatch):
    """Generated dump filename should include provided session id."""
    monkeypatch.setattr(client_mod, '_RAW_DIR', tmp_path)

    client_mod.dump_llm_json('{"ok": true}', sid='session-1')

    files = list(tmp_path.glob('*.json'))
    assert len(files) == 1
    assert files[0].name.endswith('_session-1.json')
    assert files[0].read_text(encoding='utf-8') == '{"ok": true}'


def test_dump_llm_json_generates_uuid_suffix_when_sid_is_none(
    tmp_path, monkeypatch
):
    """Without session id, dump should use first 8 chars of UUID."""
    monkeypatch.setattr(client_mod, '_RAW_DIR', tmp_path)
    monkeypatch.setattr(
        client_mod.uuid,
        'uuid4',
        lambda: SimpleNamespace(hex='cafebabedeadbeef'),
    )

    client_mod.dump_llm_json('{}', sid=None)

    files = list(tmp_path.glob('*.json'))
    assert len(files) == 1
    assert files[0].name.endswith('_cafebabe.json')


def test_chat_returns_validated_llm_diagnosis(monkeypatch):
    """chat() should call the structured LLM and persist normalized JSON."""
    fake_llm = _FakeLLM(LLMDiagnosis(summary='ok', options=['a']))
    dumped = {}
    monkeypatch.setattr(
        client_mod,
        'dump_llm_json',
        lambda text, sid: dumped.update({'text': text, 'sid': sid}),
    )

    out = client_mod.chat('sys', 'usr', session_id='sid-1', llm=fake_llm)

    assert out.summary == 'ok'
    assert out.options == ['a']
    assert dumped == {
        'text': '{"summary":"ok","options":["a"]}',
        'sid': 'sid-1',
    }
    assert fake_llm.calls == [
        {'system': 'sys', 'user': 'usr', 'output_type': LLMDiagnosis}
    ]


def test_chat_raises_library_exception_on_invalid_llm_json(monkeypatch):
    """Invalid LLM payload should raise a library-level validation error."""
    monkeypatch.setattr(client_mod, 'dump_llm_json', lambda *_: None)

    class _InvalidLLM:
        def generate(self, *_args, **_kwargs):
            return LLMDiagnosis.model_validate_json('{"summary":"only"}')

    with pytest.raises(client_mod.LLMResponseValidationError) as exc:
        client_mod.chat('system', 'user', llm=_InvalidLLM())

    assert 'LLM response is not valid LLMDiagnosis' in str(exc.value)
