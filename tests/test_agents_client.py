"""Unit tests for shared OpenAI client helpers."""

from __future__ import annotations

from types import SimpleNamespace

import hiperhealth.agents.client as client_mod
import pytest


class _FakeCompletions:
    """Mock for OpenAI `chat.completions` endpoint."""

    def __init__(self, content: str | None) -> None:
        self._content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        """Return fake API response and keep call payload."""
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


def _patch_client(monkeypatch: pytest.MonkeyPatch, content: str | None):
    """Patch module-level OpenAI client with deterministic mock."""
    completions = _FakeCompletions(content=content)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    monkeypatch.setattr(client_mod, '_client', fake_client)
    return completions


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
    """chat() should call OpenAI endpoint and validate JSON output."""
    completions = _patch_client(
        monkeypatch, content='{"summary":"ok","options":["a"]}'
    )
    dumped = {}
    monkeypatch.setattr(
        client_mod,
        'dump_llm_json',
        lambda text, sid: dumped.update({'text': text, 'sid': sid}),
    )

    out = client_mod.chat('sys', 'usr', session_id='sid-1')

    assert out.summary == 'ok'
    assert out.options == ['a']
    assert dumped == {
        'text': '{"summary":"ok","options":["a"]}',
        'sid': 'sid-1',
    }

    assert len(completions.calls) == 1
    call = completions.calls[0]
    assert call['model'] == client_mod._MODEL_NAME
    assert call['response_format'] == {'type': 'json_object'}
    assert call['messages'] == [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': 'usr'},
    ]


def test_chat_raises_library_exception_on_invalid_llm_json(monkeypatch):
    """Invalid LLM payload should raise a library-level validation error."""
    _patch_client(monkeypatch, content=None)
    monkeypatch.setattr(client_mod, 'dump_llm_json', lambda *_: None)

    with pytest.raises(client_mod.LLMResponseValidationError) as exc:
        client_mod.chat('system', 'user')

    assert 'LLM response is not valid LLMDiagnosis' in str(exc.value)
