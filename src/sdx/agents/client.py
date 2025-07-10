"""
Shared OpenRouter/Mistral helper used by all agents.

* Forces JSON responses (`response_format={"type": "json_object"}`).
* Validates with ``LLMDiagnosis.from_llm``.
* Persists every raw reply under ``data/llm_raw/<sid>_<UTC>.json``.
"""

from __future__ import annotations

import os
import uuid

from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException
from openai import OpenAI
from pydantic import ValidationError

from sdx.schema.clinical_outputs import LLMDiagnosis

load_dotenv(Path(__file__).parents[3] / '.envs' / '.env')

# OpenRouter configuration
_OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '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')
_MODEL_NAME = os.getenv('OPENROUTER_MODEL', 'mistralai/mistral-small-3.2-24b-instruct:free')
_SITE_URL = os.getenv('SITE_URL', 'https://telehealthcareai.com')
_SITE_NAME = os.getenv('SITE_NAME', 'TeleHealthCareAI')

# Initialize OpenRouter client
_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=_OPENROUTER_API_KEY,
)

_RAW_DIR = Path('data') / 'llm_raw'
_RAW_DIR.mkdir(parents=True, exist_ok=True)


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
) -> LLMDiagnosis:
    """Send system / user prompts and return a validated ``LLMDiagnosis``."""
    rsp = _client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": _SITE_URL,
            "X-Title": _SITE_NAME,
        },
        extra_body={},
        model=_MODEL_NAME,
        response_format={'type': 'json_object'},
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
    )

    raw: str = rsp.choices[0].message.content or '{}'
    dump_llm_json(raw, session_id)

    try:
        return LLMDiagnosis.from_llm(raw)
    except ValidationError as exc:
        raise HTTPException(
            422, f'LLM response is not valid LLMDiagnosis: {exc}'
        ) from exc
