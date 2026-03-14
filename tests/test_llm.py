"""Tests for provider-agnostic LLM settings and adapters."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from hiperhealth.llm import (
    LLMSettings,
    RagoStructuredLLM,
    load_diagnostics_llm_settings,
)
from hiperhealth.schema.clinical_outputs import LLMDiagnosis


class _FakeGeneration:
    """Capture Rago generation configuration and return fixed content."""

    init_calls: ClassVar[list[dict[str, Any]]] = []
    generate_calls: ClassVar[list[dict[str, Any]]] = []
    result: ClassVar[Any] = LLMDiagnosis(summary='ok', options=['a'])

    def __init__(self, **kwargs: Any) -> None:
        self.__class__.init_calls.append(kwargs)

    def generate(self, query: str, data: list[str]) -> Any:
        self.__class__.generate_calls.append({'query': query, 'data': data})
        return self.__class__.result


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


def test_rago_structured_llm_maps_ollama_to_openai_backend():
    """Structured Ollama calls should use Rago's OpenAI-compatible backend."""
    _FakeGeneration.init_calls.clear()
    _FakeGeneration.generate_calls.clear()
    _FakeGeneration.result = '{"summary":"ok","options":["a"]}'

    llm = RagoStructuredLLM(
        LLMSettings(provider='ollama', model='llama3.2:3b'),
        generation_factory=_FakeGeneration,
    )

    result = llm.generate('sys', 'usr', LLMDiagnosis)

    assert result.summary == 'ok'
    assert result.options == ['a']
    assert _FakeGeneration.init_calls[0]['backend'] == 'ollama-openai'
    assert _FakeGeneration.init_calls[0]['prompt_template'] == '{query}'
    assert _FakeGeneration.generate_calls[0] == {
        'query': 'usr',
        'data': [],
    }


def test_anamnesis_backend_rejects_unsupported_provider():
    """FHIR extraction should fail for unsupported AnamnesisAI backends."""
    with pytest.raises(ValueError) as exc:
        LLMSettings(provider='gemini').to_anamnesis_backend()

    assert 'supports only the openai and ollama providers' in str(exc.value)
