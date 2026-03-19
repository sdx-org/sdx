"""
title: Tests for provider-agnostic LLM settings and adapters.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hiperhealth.llm import (
    LiteLLMStructuredLLM,
    LLMSettings,
    _clean_json_text,
    _coerce_model_output,
    _extract_message_content,
    _join_content_blocks,
    _load_api_params,
    build_structured_llm,
    load_diagnostics_llm_settings,
)
from hiperhealth.schema.clinical_outputs import LLMDiagnosis


def test_load_diagnostics_llm_settings_prefers_specific_env(monkeypatch):
    """
    title: Task-specific env vars should override generic and legacy ones.
    parameters:
      monkeypatch:
        description: Value for monkeypatch.
    """
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
    """
    title: Structured generation should map settings into LiteLLM kwargs.
    """
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


def test_with_overrides_merges_params():
    """
    title: with_overrides should merge api_params and override fields.
    """
    base = LLMSettings(
        provider='openai',
        model='gpt-4',
        temperature=0.5,
        api_params={'base_url': 'http://old'},
    )
    overridden = base.with_overrides(
        model='gpt-4o',
        temperature=0.0,
        api_params={'timeout': 30},
    )

    assert overridden.model == 'gpt-4o'
    assert overridden.temperature == 0.0
    assert overridden.provider == 'openai'
    assert overridden.api_params['base_url'] == 'http://old'
    assert overridden.api_params['timeout'] == 30


def test_to_litellm_model_raises_when_no_model():
    """
    title: to_litellm_model should raise ValueError when no model set.
    """
    settings = LLMSettings(provider='openai', model='', engine='')
    with pytest.raises(ValueError, match='LLM model is required'):
        settings.to_litellm_model()


def test_to_litellm_model_returns_as_is_when_slash_present():
    """
    title: Model names with / should be returned unchanged.
    """
    settings = LLMSettings(model='openai/gpt-4o')
    assert settings.to_litellm_model() == 'openai/gpt-4o'


def test_to_litellm_model_uses_engine_fallback():
    """
    title: engine should be used if model is empty.
    """
    settings = LLMSettings(provider='openai', model='', engine='davinci-002')
    assert settings.to_litellm_model() == 'openai/davinci-002'


def test_coerce_model_output_from_instance():
    """
    title: Already correct type should be returned as-is.
    """
    obj = LLMDiagnosis(summary='ok', options=['a'])
    result = _coerce_model_output(obj, LLMDiagnosis)
    assert result is obj


def test_coerce_model_output_from_dict():
    """
    title: Dict should be validated into the Pydantic model.
    """
    result = _coerce_model_output(
        {'summary': 'ok', 'options': ['a']},
        LLMDiagnosis,
    )
    assert isinstance(result, LLMDiagnosis)
    assert result.summary == 'ok'


def test_coerce_model_output_from_string():
    """
    title: JSON string should be parsed into the Pydantic model.
    """
    result = _coerce_model_output(
        '{"summary": "ok", "options": ["a"]}',
        LLMDiagnosis,
    )
    assert isinstance(result, LLMDiagnosis)
    assert result.summary == 'ok'


def test_coerce_model_output_from_fenced_json():
    """
    title: Fenced markdown JSON should be cleaned and parsed.
    """
    result = _coerce_model_output(
        '```json\n{"summary": "ok", "options": ["a"]}\n```',
        LLMDiagnosis,
    )
    assert isinstance(result, LLMDiagnosis)


def test_coerce_model_output_unsupported_type():
    """
    title: Unsupported type should raise TypeError.
    """
    with pytest.raises(TypeError, match='Unsupported structured'):
        _coerce_model_output(42, LLMDiagnosis)


def test_coerce_model_output_from_other_basemodel():
    """
    title: Another BaseModel should be converted via model_dump.
    """
    from pydantic import BaseModel

    class OtherModel(BaseModel):
        summary: str
        options: list[str]

    other = OtherModel(summary='converted', options=['x'])
    result = _coerce_model_output(other, LLMDiagnosis)
    assert isinstance(result, LLMDiagnosis)
    assert result.summary == 'converted'


def test_extract_message_content_string_passthrough():
    """
    title: String response should be returned as-is.
    """
    assert _extract_message_content('raw text') == 'raw text'


def test_extract_message_content_empty_choices():
    """
    title: Response with no choices should raise TypeError.
    """
    with pytest.raises(TypeError, match='did not include any choices'):
        _extract_message_content(SimpleNamespace(choices=[]))


def test_extract_message_content_list_blocks():
    """
    title: Multi-part content blocks should be joined.
    """
    response = {
        'choices': [
            {
                'message': {
                    'content': [
                        {'text': 'part1'},
                        'part2',
                        {'content': 'part3'},
                    ]
                }
            }
        ]
    }
    result = _extract_message_content(response)
    assert 'part1' in result
    assert 'part2' in result
    assert 'part3' in result


def test_extract_message_content_unsupported_type():
    """
    title: Unsupported content type should raise TypeError.
    """
    response = {'choices': [{'message': {'content': 12345}}]}
    with pytest.raises(TypeError, match='must be a string or dict'):
        _extract_message_content(response)


def test_extract_message_content_dict_content():
    """
    title: Dict content should be returned as-is.
    """
    response = {'choices': [{'message': {'content': {'key': 'value'}}}]}
    result = _extract_message_content(response)
    assert result == {'key': 'value'}


def test_join_content_blocks_mixed():
    """
    title: Mixed string and dict blocks should be joined.
    """
    blocks = [
        'hello',
        {'text': 'world'},
        {'content': '!'},
        {'other': 'ignored'},
    ]
    result = _join_content_blocks(blocks)
    assert result == 'hello\nworld\n!'


def test_clean_json_text_fenced():
    """
    title: Fenced json blocks should be unwrapped.
    """
    assert _clean_json_text('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_clean_json_text_plain():
    """
    title: Plain JSON should be returned unchanged.
    """
    assert _clean_json_text('{"a": 1}') == '{"a": 1}'


def test_clean_json_text_fenced_no_lang():
    """
    title: Fenced blocks without language tag should be unwrapped.
    """
    assert _clean_json_text('```\n{"a": 1}\n```') == '{"a": 1}'


def test_load_api_params_from_env(monkeypatch):
    """
    title: _load_api_params should parse JSON from env vars.
    parameters:
      monkeypatch:
        description: Value for monkeypatch.
    """
    monkeypatch.setenv(
        'HIPERHEALTH_LLM_API_PARAMS',
        '{"timeout": 30, "retries": 3}',
    )
    result = _load_api_params(('HIPERHEALTH_LLM_',))
    assert result == {'timeout': 30, 'retries': 3}


def test_load_api_params_rejects_non_dict(monkeypatch):
    """
    title: Non-dict JSON in API_PARAMS should raise ValueError.
    parameters:
      monkeypatch:
        description: Value for monkeypatch.
    """
    monkeypatch.setenv(
        'HIPERHEALTH_LLM_API_PARAMS',
        '[1, 2, 3]',
    )
    with pytest.raises(ValueError, match='must be a JSON object'):
        _load_api_params(('HIPERHEALTH_LLM_',))


def test_build_structured_llm_returns_litellm_adapter():
    """
    title: build_structured_llm should return a LiteLLMStructuredLLM.
    """
    settings = LLMSettings(provider='ollama', model='test')
    llm = build_structured_llm(settings)
    assert isinstance(llm, LiteLLMStructuredLLM)


def test_litellm_api_key_in_kwargs():
    """
    title: LLM api_key should be included in to_litellm_kwargs.
    """
    settings = LLMSettings(
        provider='openai',
        model='gpt-4',
        api_key='sk-test-key',
    )
    kwargs = settings.to_litellm_kwargs()
    assert kwargs['api_key'] == 'sk-test-key'


def test_to_litellm_kwargs_always_includes_temperature_and_max_tokens():
    """
    title: >-
      All models get temperature and max_tokens; LiteLLM drops unsupported
      params.
    """
    for model in ('o4-mini', 'gpt-4', 'llama3.2:1b'):
        settings = LLMSettings(
            provider='openai',
            model=model,
            api_key='sk-test',
        )
        kwargs = settings.to_litellm_kwargs()
        assert kwargs['temperature'] == 0.0
        assert kwargs['max_tokens'] == 4096
