"""
title: Provider-agnostic LLM settings and structured-generation adapters.
"""

from __future__ import annotations

import json
import os

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol, TypeVar, cast

from pydantic import BaseModel

TModel = TypeVar('TModel', bound=BaseModel)

_GENERIC_PREFIX = 'HIPERHEALTH_LLM_'
_DIAGNOSTICS_PREFIX = 'HIPERHEALTH_DIAGNOSTICS_LLM_'
_STRUCTURED_OUTPUT_INSTRUCTION = (
    'Return only a valid JSON object that matches the provided JSON '
    'Schema. Do not include markdown fences or explanatory text.'
)

_PROVIDER_ALIASES = {
    'ollama-openai': 'ollama',
}

_PROVIDER_API_KEY_ENV = {
    'cohere': ('COHERE_API_KEY',),
    'fireworks': ('FIREWORKS_API_KEY',),
    'gemini': ('GEMINI_API_KEY', 'GOOGLE_API_KEY'),
    'groq': ('GROQ_API_KEY',),
    'huggingface': ('HUGGINGFACE_API_KEY', 'HF_TOKEN'),
    'huggingface-inference': ('HUGGINGFACE_API_KEY', 'HF_TOKEN'),
    'openai': ('OPENAI_API_KEY',),
    'together': ('TOGETHER_API_KEY',),
}

_DEFAULT_PROVIDER_MODEL = {
    'ollama': 'llama3.2:1b',
    'openai': 'o4-mini',
}


class StructuredLLM(Protocol):
    """
    title: Minimal interface expected by hiperhealth prompt workflows.
    """

    def generate(
        self,
        system: str,
        user: str,
        output_type: type[TModel],
    ) -> TModel:
        """
        title: Generate and validate a structured response.
        parameters:
          system:
            type: str
            description: Value for system.
          user:
            type: str
            description: Value for user.
          output_type:
            type: type[TModel]
            description: Value for output_type.
        returns:
          type: TModel
          description: Return value.
        """


_CompletionFn = Callable[..., Any]


@dataclass(frozen=True)
class LLMSettings:
    """
    title: Runtime-configurable LLM settings for a single workflow.
    attributes:
      provider:
        type: str
        description: Value for provider.
      model:
        type: str
        description: Value for model.
      api_key:
        type: str
        description: Value for api_key.
      engine:
        type: str
        description: Value for engine.
      temperature:
        type: float
        description: Value for temperature.
      max_tokens:
        type: int
        description: Value for max_tokens.
      api_params:
        type: dict[str, Any]
        description: Value for api_params.
    """

    provider: str = 'openai'
    model: str = ''
    api_key: str = ''
    engine: str = ''
    temperature: float = 0.0
    max_tokens: int = 800
    api_params: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_provider(self) -> str:
        """
        title: Return a canonical lowercase provider identifier.
        returns:
          type: str
          description: Return value.
        """
        provider = self.provider.strip().lower()
        return _PROVIDER_ALIASES.get(provider, provider)

    def with_overrides(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        engine: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_params: dict[str, Any] | None = None,
    ) -> LLMSettings:
        """
        title: Return a copy with selective overrides applied.
        parameters:
          provider:
            type: str | None
            description: Value for provider.
          model:
            type: str | None
            description: Value for model.
          api_key:
            type: str | None
            description: Value for api_key.
          engine:
            type: str | None
            description: Value for engine.
          temperature:
            type: float | None
            description: Value for temperature.
          max_tokens:
            type: int | None
            description: Value for max_tokens.
          api_params:
            type: dict[str, Any] | None
            description: Value for api_params.
        returns:
          type: LLMSettings
          description: Return value.
        """
        merged_params = dict(self.api_params)
        if api_params:
            merged_params.update(api_params)

        return replace(
            self,
            provider=provider or self.provider,
            model=model if model is not None else self.model,
            api_key=api_key if api_key is not None else self.api_key,
            engine=engine if engine is not None else self.engine,
            temperature=(
                temperature if temperature is not None else self.temperature
            ),
            max_tokens=max_tokens
            if max_tokens is not None
            else self.max_tokens,
            api_params=merged_params,
        )

    def to_litellm_model(self) -> str:
        """
        title: Return the fully-qualified LiteLLM model identifier.
        returns:
          type: str
          description: Return value.
        """
        model_name = self.model or self.engine
        if not model_name:
            raise ValueError(
                'LLM model is required. Set HIPERHEALTH_*_LLM_MODEL or '
                'pass LLMSettings(model=...).'
            )
        if '/' in model_name:
            return model_name
        return f'{self.normalized_provider}/{model_name}'

    def to_litellm_kwargs(self) -> dict[str, Any]:
        """
        title: Build LiteLLM completion kwargs from the current settings.
        returns:
          type: dict[str, Any]
          description: Return value.
        """
        kwargs = dict(self.api_params)
        base_url = kwargs.pop('base_url', '')
        if base_url and 'api_base' not in kwargs:
            kwargs['api_base'] = base_url
        kwargs['model'] = self.to_litellm_model()
        kwargs['temperature'] = self.temperature
        kwargs['max_tokens'] = self.max_tokens
        if self.api_key:
            kwargs['api_key'] = self.api_key
        return kwargs


class LiteLLMStructuredLLM:
    """
    title: Structured-generation adapter built on top of LiteLLM.
    attributes:
      settings:
        description: Value for settings.
      _completion_fn:
        description: Value for _completion_fn.
    """

    def __init__(
        self,
        settings: LLMSettings,
        completion_fn: _CompletionFn | None = None,
    ) -> None:
        self.settings = settings
        self._completion_fn = completion_fn

    def _get_completion_fn(self) -> _CompletionFn:
        if self._completion_fn is not None:
            return self._completion_fn

        from litellm import completion

        return cast(_CompletionFn, completion)

    def generate(
        self,
        system: str,
        user: str,
        output_type: type[TModel],
    ) -> TModel:
        """
        title: Generate a structured response using the configured backend.
        parameters:
          system:
            type: str
            description: Value for system.
          user:
            type: str
            description: Value for user.
          output_type:
            type: type[TModel]
            description: Value for output_type.
        returns:
          type: TModel
          description: Return value.
        """
        completion_fn = self._get_completion_fn()
        response = completion_fn(
            messages=_build_messages(system, user, output_type),
            **self.settings.to_litellm_kwargs(),
        )
        result = _extract_message_content(response)
        return _coerce_model_output(result, output_type)


def build_structured_llm(
    settings: LLMSettings | None = None,
    *,
    completion_fn: _CompletionFn | None = None,
) -> StructuredLLM:
    """
    title: Build the default structured LLM adapter for hiperhealth workflows.
    parameters:
      settings:
        type: LLMSettings | None
        description: Value for settings.
      completion_fn:
        type: _CompletionFn | None
        description: Value for completion_fn.
    returns:
      type: StructuredLLM
      description: Return value.
    """
    effective_settings = settings or load_diagnostics_llm_settings()
    return LiteLLMStructuredLLM(
        settings=effective_settings,
        completion_fn=completion_fn,
    )


def load_diagnostics_llm_settings() -> LLMSettings:
    """
    title: Load diagnostics-generation settings from env variables.
    returns:
      type: LLMSettings
      description: Return value.
    """
    return load_llm_settings(
        prefixes=(_DIAGNOSTICS_PREFIX, _GENERIC_PREFIX),
        default_provider='openai',
        legacy_model_envs=('OPENAI_MODEL',),
        legacy_api_key_envs=('OPENAI_API_KEY',),
    )


def load_llm_settings(
    *,
    prefixes: tuple[str, ...] = (_GENERIC_PREFIX,),
    default_provider: str = 'openai',
    legacy_model_envs: tuple[str, ...] = (),
    legacy_api_key_envs: tuple[str, ...] = (),
) -> LLMSettings:
    """
    title: Load LLM settings from env vars, with task-specific prefixes first.
    parameters:
      prefixes:
        type: tuple[str, Ellipsis]
        description: Value for prefixes.
      default_provider:
        type: str
        description: Value for default_provider.
      legacy_model_envs:
        type: tuple[str, Ellipsis]
        description: Value for legacy_model_envs.
      legacy_api_key_envs:
        type: tuple[str, Ellipsis]
        description: Value for legacy_api_key_envs.
    returns:
      type: LLMSettings
      description: Return value.
    """
    raw_provider = (
        (
            _first_nonempty_env(_prefixed_names(prefixes, 'PROVIDER'))
            or default_provider
        )
        .strip()
        .lower()
    )
    provider = _PROVIDER_ALIASES.get(raw_provider, raw_provider)

    model_env_names = _prefixed_names(prefixes, 'MODEL')
    if provider == 'openai':
        model_env_names += legacy_model_envs
    model = _first_nonempty_env(
        model_env_names
    ) or _DEFAULT_PROVIDER_MODEL.get(
        provider,
        '',
    )

    api_key_env_names = _prefixed_names(
        prefixes, 'API_KEY'
    ) + _PROVIDER_API_KEY_ENV.get(provider, ())
    if provider == 'openai':
        api_key_env_names += legacy_api_key_envs
    api_key = _first_nonempty_env(api_key_env_names)

    engine = _first_nonempty_env(_prefixed_names(prefixes, 'ENGINE'))
    temperature = _read_float_env(
        _prefixed_names(prefixes, 'TEMPERATURE'),
        default=0.0,
    )
    max_tokens = _read_int_env(
        _prefixed_names(prefixes, 'MAX_TOKENS'),
        default=800,
    )

    api_params = _load_api_params(prefixes)
    base_url = _first_nonempty_env(_prefixed_names(prefixes, 'BASE_URL'))
    if base_url:
        api_params.setdefault('base_url', base_url)

    return LLMSettings(
        provider=provider,
        model=model,
        api_key=api_key,
        engine=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        api_params=api_params,
    )


def _coerce_model_output(
    result: Any,
    output_type: type[TModel],
) -> TModel:
    """
    title: Normalize provider outputs to the requested Pydantic model type.
    parameters:
      result:
        type: Any
        description: Value for result.
      output_type:
        type: type[TModel]
        description: Value for output_type.
    returns:
      type: TModel
      description: Return value.
    """
    if isinstance(result, output_type):
        return result
    if isinstance(result, BaseModel):
        return output_type.model_validate(result.model_dump())
    if isinstance(result, dict):
        return output_type.model_validate(result)
    if isinstance(result, str):
        return output_type.model_validate_json(_clean_json_text(result))
    raise TypeError(
        f'Unsupported structured LLM result type: {type(result).__name__}'
    )


def _build_messages(
    system: str,
    user: str,
    output_type: type[TModel],
) -> list[dict[str, str]]:
    """
    title: Build a vendor-neutral prompt for structured JSON responses.
    parameters:
      system:
        type: str
        description: Value for system.
      user:
        type: str
        description: Value for user.
      output_type:
        type: type[TModel]
        description: Value for output_type.
    returns:
      type: list[dict[str, str]]
      description: Return value.
    """
    schema = json.dumps(output_type.model_json_schema(), ensure_ascii=False)
    system_message = '\n\n'.join(
        (
            system.strip(),
            _STRUCTURED_OUTPUT_INSTRUCTION,
            f'JSON Schema:\n{schema}',
        )
    ).strip()
    return [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user},
    ]


def _extract_message_content(response: Any) -> Any:
    """
    title: Extract the first assistant message content from a LiteLLM response.
    parameters:
      response:
        type: Any
        description: Value for response.
    returns:
      type: Any
      description: Return value.
    """
    if isinstance(response, (str, BaseModel, dict)):
        if not isinstance(response, dict) or 'choices' not in response:
            return response

    choices = _get_mapping_or_attr(response, 'choices')
    if not choices:
        raise TypeError('LiteLLM response did not include any choices.')

    choice = choices[0]
    message = _get_mapping_or_attr(choice, 'message')
    content = _get_mapping_or_attr(message, 'content')

    if isinstance(content, list):
        return _join_content_blocks(content)
    if isinstance(content, (str, dict)):
        return content
    raise TypeError(
        'LiteLLM response message content must be a string or dict.'
    )


def _get_mapping_or_attr(value: Any, key: str) -> Any:
    """
    title: Read *key* from dict-like or attribute-based SDK objects.
    parameters:
      value:
        type: Any
        description: Value for value.
      key:
        type: str
        description: Value for key.
    returns:
      type: Any
      description: Return value.
    """
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _join_content_blocks(blocks: list[Any]) -> str:
    """
    title: Flatten multi-part content blocks into a single text payload.
    parameters:
      blocks:
        type: list[Any]
        description: Value for blocks.
    returns:
      type: str
      description: Return value.
    """
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
            continue
        if isinstance(block, dict):
            text = block.get('text') or block.get('content')
            if text:
                parts.append(str(text))
    return '\n'.join(part for part in parts if part)


def _clean_json_text(text: str) -> str:
    """
    title: Strip simple fenced-markdown wrappers from provider JSON responses.
    parameters:
      text:
        type: str
        description: Value for text.
    returns:
      type: str
      description: Return value.
    """
    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = cleaned[3:]
        if cleaned.startswith('json'):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
    return cleaned.strip()


def _prefixed_names(
    prefixes: tuple[str, ...],
    suffix: str,
) -> tuple[str, ...]:
    return tuple(f'{prefix}{suffix}' for prefix in prefixes)


def _load_api_params(prefixes: tuple[str, ...]) -> dict[str, Any]:
    """
    title: Load and merge JSON api params from generic to specific prefixes.
    parameters:
      prefixes:
        type: tuple[str, Ellipsis]
        description: Value for prefixes.
    returns:
      type: dict[str, Any]
      description: Return value.
    """
    params: dict[str, Any] = {}
    for prefix in reversed(prefixes):
        raw = os.getenv(f'{prefix}API_PARAMS')
        if not raw:
            continue
        loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError(
                f'{prefix}API_PARAMS must be a JSON object, got {raw!r}.'
            )
        params.update(loaded)
    return params


def _first_nonempty_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ''


def _read_float_env(names: tuple[str, ...], *, default: float) -> float:
    value = _first_nonempty_env(names)
    if not value:
        return default
    return float(value)


def _read_int_env(names: tuple[str, ...], *, default: int) -> int:
    value = _first_nonempty_env(names)
    if not value:
        return default
    return int(value)
