"""Provider-agnostic LLM settings and structured-generation adapters."""

from __future__ import annotations

import json
import os

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol, TypeVar, cast

from pydantic import BaseModel

TModel = TypeVar('TModel', bound=BaseModel)

_GENERIC_PREFIX = 'HIPERHEALTH_LLM_'
_DIAGNOSTICS_PREFIX = 'HIPERHEALTH_DIAGNOSTICS_LLM_'
_FHIR_PREFIX = 'HIPERHEALTH_FHIR_LLM_'

_PROVIDER_API_KEY_ENV = {
    'cohere': ('COHERE_API_KEY',),
    'fireworks': ('FIREWORKS_API_KEY',),
    'gemini': ('GEMINI_API_KEY', 'GOOGLE_API_KEY'),
    'groq': ('GROQ_API_KEY',),
    'huggingface': ('HUGGINGFACE_API_KEY', 'HF_TOKEN'),
    'huggingface-inference': ('HUGGINGFACE_API_KEY', 'HF_TOKEN'),
    'ollama-openai': (),
    'openai': ('OPENAI_API_KEY',),
    'together': ('TOGETHER_API_KEY',),
}

_DEFAULT_PROVIDER_MODEL = {
    'ollama': 'llama3.2:1b',
    'ollama-openai': 'llama3.2:1b',
    'openai': 'o4-mini',
}


class StructuredLLM(Protocol):
    """Minimal interface expected by hiperhealth prompt workflows."""

    def generate(
        self,
        system: str,
        user: str,
        output_type: type[TModel],
    ) -> TModel:
        """Generate and validate a structured response."""


_GenerationFactory = Callable[..., Any]


@dataclass(frozen=True)
class LLMSettings:
    """Runtime-configurable LLM settings for a single workflow."""

    provider: str = 'openai'
    model: str = ''
    api_key: str = ''
    engine: str = ''
    temperature: float = 0.0
    max_tokens: int = 800
    api_params: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_provider(self) -> str:
        """Return a canonical lowercase provider identifier."""
        return self.provider.strip().lower()

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
        """Return a copy with selective overrides applied."""
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

    def to_rago_backend(self, *, structured_output: bool = False) -> str:
        """
        Map user-facing provider names to the corresponding Rago backend.

        For structured output, Ollama must use the OpenAI-compatible backend.
        """
        provider = self.normalized_provider
        if provider == 'ollama' and structured_output:
            return 'ollama-openai'
        return provider

    def to_anamnesis_backend(self) -> str:
        """Return the backend supported by AnamnesisAI for FHIR extraction."""
        provider = self.normalized_provider
        if provider in {'openai', 'ollama'}:
            return provider
        raise ValueError(
            'AnamnesisAI currently supports only the openai and ollama '
            f'providers, got {self.provider!r}.'
        )

    def to_anamnesis_api_params(self) -> dict[str, Any]:
        """Translate generic settings to AnamnesisAI generation parameters."""
        api_params = dict(self.api_params)
        if self.model:
            api_params.setdefault('model_name', self.model)
        if self.max_tokens:
            api_params.setdefault('output_max_length', self.max_tokens)
        return api_params


class RagoStructuredLLM:
    """Structured-generation adapter built on top of Rago Generation."""

    def __init__(
        self,
        settings: LLMSettings,
        generation_factory: _GenerationFactory | None = None,
    ) -> None:
        self.settings = settings
        self._generation_factory = generation_factory

    def _get_generation_factory(self) -> _GenerationFactory:
        if self._generation_factory is not None:
            return self._generation_factory

        from rago import Generation

        return cast(_GenerationFactory, Generation)

    def generate(
        self,
        system: str,
        user: str,
        output_type: type[TModel],
    ) -> TModel:
        """Generate a structured response using the configured Rago backend."""
        generation_factory = self._get_generation_factory()
        generation = generation_factory(
            backend=self.settings.to_rago_backend(structured_output=True),
            engine=self.settings.engine,
            api_key=self.settings.api_key,
            model_name=self.settings.model,
            temperature=self.settings.temperature,
            output_max_length=self.settings.max_tokens,
            structured_output=output_type,
            system_message=system,
            prompt_template='{query}',
            api_params=dict(self.settings.api_params),
        )
        result = generation.generate(user, [])
        return _coerce_model_output(result, output_type)


def build_structured_llm(
    settings: LLMSettings | None = None,
    *,
    generation_factory: _GenerationFactory | None = None,
) -> StructuredLLM:
    """Build the default structured LLM adapter for hiperhealth workflows."""
    effective_settings = settings or load_diagnostics_llm_settings()
    return RagoStructuredLLM(
        settings=effective_settings,
        generation_factory=generation_factory,
    )


def load_diagnostics_llm_settings() -> LLMSettings:
    """Load diagnostics-generation settings from env variables."""
    return load_llm_settings(
        prefixes=(_DIAGNOSTICS_PREFIX, _GENERIC_PREFIX),
        default_provider='openai',
        legacy_model_envs=('OPENAI_MODEL',),
        legacy_api_key_envs=('OPENAI_API_KEY',),
    )


def load_fhir_llm_settings() -> LLMSettings:
    """Load FHIR-extraction settings from env variables."""
    return load_llm_settings(
        prefixes=(_FHIR_PREFIX, _GENERIC_PREFIX),
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
    """Load LLM settings from env vars, with task-specific prefixes first."""
    provider = (
        (
            _first_nonempty_env(_prefixed_names(prefixes, 'PROVIDER'))
            or default_provider
        )
        .strip()
        .lower()
    )

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
    """Normalize provider outputs to the requested Pydantic model type."""
    if isinstance(result, output_type):
        return result
    if isinstance(result, BaseModel):
        return output_type.model_validate(result.model_dump())
    if isinstance(result, str):
        return output_type.model_validate_json(_clean_json_text(result))
    raise TypeError(
        f'Unsupported structured LLM result type: {type(result).__name__}'
    )


def _clean_json_text(text: str) -> str:
    """Strip simple fenced-markdown wrappers from provider JSON responses."""
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
    """Load and merge JSON api params from generic to specific prefixes."""
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
