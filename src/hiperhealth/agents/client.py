"""
Shared structured-LLM helper used by all agents.

* Validates with ``LLMDiagnosis``.
* Persists every normalized reply under ``data/llm_raw/<sid>_<UTC>.json``.
"""

from __future__ import annotations

import uuid

from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from hiperhealth.llm import (
    LLMSettings,
    StructuredLLM,
    build_structured_llm,
    load_diagnostics_llm_settings,
)
from hiperhealth.schema.clinical_outputs import LLMDiagnosis

_RAW_DIR = Path('data') / 'llm_raw'
_RAW_DIR.mkdir(parents=True, exist_ok=True)


class LLMResponseValidationError(ValueError):
    """Raised when LLM output cannot be validated as LLMDiagnosis."""


def dump_llm_json(text: str, sid: str | None) -> None:
    """
    Save *text* to data/llm_raw/<timestamp>_<sid>.json.

    If *sid* is None, a random 8-char token is used instead.
    """
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    suffix = sid or uuid.uuid4().hex[:8]
    (_RAW_DIR / f'{ts}_{suffix}.json').write_text(text, encoding='utf-8')


def chat(
    system: str,
    user: str,
    *,
    session_id: str | None = None,
    llm: StructuredLLM | None = None,
    llm_settings: LLMSettings | None = None,
) -> LLMDiagnosis:
    """Send system / user prompts and return a validated ``LLMDiagnosis``."""
    effective_llm = llm or _get_llm(llm_settings)

    try:
        result = effective_llm.generate(system, user, LLMDiagnosis)
    except ValidationError as exc:
        raise LLMResponseValidationError(
            f'LLM response is not valid LLMDiagnosis: {exc}'
        ) from exc

    dump_llm_json(result.model_dump_json(), session_id)
    return result


def _get_llm(llm_settings: LLMSettings | None) -> StructuredLLM:
    """Resolve the structured LLM adapter for the current request."""
    settings = llm_settings or load_diagnostics_llm_settings()
    return build_structured_llm(settings)
