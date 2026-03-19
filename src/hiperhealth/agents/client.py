"""
title: Shared structured-LLM helper used by all agents.
summary: |-
  * ``chat_structured`` validates against any Pydantic model.
  * ``chat`` is a convenience wrapper that validates with ``LLMDiagnosis``.
  * Persists every normalized reply under ``data/llm_raw/<sid>_<UTC>.json``.
"""

from __future__ import annotations

import logging
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from hiperhealth.llm import (
    LLMSettings,
    StructuredLLM,
    build_structured_llm,
    load_diagnostics_llm_settings,
)
from hiperhealth.schema.clinical_outputs import LLMDiagnosis

TModel = TypeVar('TModel', bound=BaseModel)

_log = logging.getLogger(__name__)
_RAW_DIR = Path('data') / 'llm_raw'


class LLMResponseValidationError(ValueError):
    """
    title: Raised when LLM output cannot be validated.
    """


def dump_llm_json(text: str, sid: str | None) -> None:
    """
    title: Save *text* to data/llm_raw/<timestamp>_<sid>.json.
    summary: If *sid* is None, a random 8-char token is used instead.
    parameters:
      text:
        type: str
        description: Value for text.
      sid:
        type: str | None
        description: Value for sid.
    """
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    suffix = sid or uuid.uuid4().hex[:8]
    (_RAW_DIR / f'{ts}_{suffix}.json').write_text(text, encoding='utf-8')


@retry(
    retry=retry_if_exception_type((ValidationError, TypeError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    before_sleep=before_sleep_log(_log, logging.WARNING),
    reraise=True,
)
def _call_llm_structured(
    llm: StructuredLLM,
    system: str,
    user: str,
    output_type: type[TModel],
) -> TModel:
    """
    title: Call the LLM and return a validated Pydantic model.
    summary: |-
      Retries up to 3 attempts on transient validation failures
      (empty responses, malformed JSON).
    parameters:
      llm:
        type: StructuredLLM
      system:
        type: str
      user:
        type: str
      output_type:
        type: type[TModel]
    returns:
      type: TModel
    """
    return llm.generate(system, user, output_type)


def chat_structured(
    system: str,
    user: str,
    output_type: type[TModel],
    *,
    session_id: str | None = None,
    llm: StructuredLLM | None = None,
    llm_settings: LLMSettings | None = None,
) -> TModel:
    """
    title: Send prompts and return a validated Pydantic model of any type.
    parameters:
      system:
        type: str
        description: Value for system.
      user:
        type: str
        description: Value for user.
      output_type:
        type: type[TModel]
        description: Pydantic model class to validate the response against.
      session_id:
        type: str | None
        description: Value for session_id.
      llm:
        type: StructuredLLM | None
        description: Value for llm.
      llm_settings:
        type: LLMSettings | None
        description: Value for llm_settings.
    returns:
      type: TModel
      description: Return value.
    """
    effective_llm = llm or _get_llm(llm_settings)

    try:
        result = _call_llm_structured(effective_llm, system, user, output_type)
    except (ValidationError, TypeError) as exc:
        raise LLMResponseValidationError(
            f'LLM response is not valid {output_type.__name__}: {exc}'
        ) from exc

    effective_settings = llm_settings or load_diagnostics_llm_settings()
    if effective_settings.persist_raw:
        dump_llm_json(result.model_dump_json(), session_id)
    return result


def chat(
    system: str,
    user: str,
    *,
    session_id: str | None = None,
    llm: StructuredLLM | None = None,
    llm_settings: LLMSettings | None = None,
) -> LLMDiagnosis:
    """
    title: Send system / user prompts and return a validated ``LLMDiagnosis``.
    parameters:
      system:
        type: str
        description: Value for system.
      user:
        type: str
        description: Value for user.
      session_id:
        type: str | None
        description: Value for session_id.
      llm:
        type: StructuredLLM | None
        description: Value for llm.
      llm_settings:
        type: LLMSettings | None
        description: Value for llm_settings.
    returns:
      type: LLMDiagnosis
      description: Return value.
    """
    return chat_structured(
        system,
        user,
        LLMDiagnosis,
        session_id=session_id,
        llm=llm,
        llm_settings=llm_settings,
    )


def _get_llm(llm_settings: LLMSettings | None) -> StructuredLLM:
    """
    title: Resolve the structured LLM adapter for the current request.
    parameters:
      llm_settings:
        type: LLMSettings | None
        description: Value for llm_settings.
    returns:
      type: StructuredLLM
      description: Return value.
    """
    settings = llm_settings or load_diagnostics_llm_settings()
    return build_structured_llm(settings)
