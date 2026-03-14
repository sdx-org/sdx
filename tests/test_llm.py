"""Tests for provider-agnostic LLM settings and adapters."""

from __future__ import annotations

from hiperhealth.llm import (
    LiteLLMStructuredLLM,
    LLMSettings,
    load_diagnostics_llm_settings,
)
from hiperhealth.schema.clinical_outputs import LLMDiagnosis


def test_load_diagnostics_llm_settings_prefers_specific_env(monkeypatch):
    """Task-specific env vars should override generic and legacy ones."""
    monkeypatch.setenv('OPENAI_MODEL', 'legacy-model')
    monkeypatch.setenv('OPENAI_API_KEY', 'legacy-key')
    monkeypatch.setenv('HIPERHEALTH_LLM_PROVIDER', 'openai')
    monkeypatch.setenv(
        'HIPERHEALTH_DIAGNOSTICS_LLM_PROVIDER',
        'ollama',
    )
    monkeypatch.setenv(
        'HIPERHEALTH_DIAGNOSTICS_LLM_MODEL',
        'llama3.2:3b',
    )
    monkeypatch.setenv(
        'HIPERHEALTH_DIAGNOSTICS_LLM_BASE_URL',
        'http://localhost:11434/v1',
    )

    settings = load_diagnostics_llm_settings()

    assert settings.provider == 'ollama'
    assert settings.model == 'llama3.2:3b'
    assert settings.api_key == ''
    assert settings.api_params['base_url'] == 'http://localhost:11434/v1'


def test_litellm_structured_llm_builds_messages_and_kwargs():
    """Structured generation should map settings into LiteLLM kwargs."""
    calls: list[dict[str, object]] = []

    def _fake_completion(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            'choices': [
                {'message': {'content': '{"summary":"ok","options":["a"]}'}}
            ]
        }

    llm = LiteLLMStructuredLLM(
        LLMSettings(
            provider='ollama',
            model='llama3.2:3b',
            api_params={'base_url': 'http://localhost:11434/v1'},
        ),
        completion_fn=_fake_completion,
    )

    result = llm.generate('sys', 'usr', LLMDiagnosis)

    assert result.summary == 'ok'
    assert result.options == ['a']
    assert calls[0]['model'] == 'ollama/llama3.2:3b'
    assert calls[0]['api_base'] == 'http://localhost:11434/v1'
    messages = calls[0]['messages']
    assert isinstance(messages, list)
    assert messages[1] == {'role': 'user', 'content': 'usr'}
    assert messages[0]['role'] == 'system'
    assert 'Return only a valid JSON object' in messages[0]['content']
    assert 'LLMDiagnosis' in messages[0]['content']
