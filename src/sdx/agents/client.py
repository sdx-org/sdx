"""
Shared OpenAI helper used by all agents.

* Forces JSON responses with `response_format={"type": "json_object"}`.
* Delegates fence-stripping and Pydantic validation to
  `LLMDiagnosis.from_llm`.
"""

from __future__ import annotations

import os

from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException
from openai import OpenAI
from pydantic import ValidationError

from sdx.schema.clinical_outputs import LLMDiagnosis

load_dotenv(Path(__file__).parents[3] / '.envs' / '.env')

_MODEL_NAME = os.getenv('OPENAI_MODEL', 'o4-mini-2025-04-16')
_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', ''))


def chat(system: str, user: str) -> LLMDiagnosis:
    """Send a system/user prompt and return a validated ``LLMDiagnosis``."""
    rsp = _client.chat.completions.create(
        model=_MODEL_NAME,
        response_format={'type': 'json_object'},
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
    )

    raw: str = rsp.choices[0].message.content or '{}'

    try:
        return LLMDiagnosis.from_llm(raw)
    except ValidationError as exc:  # JSON syntax or schema error
        raise HTTPException(
            422, f'LLM response is not valid LLMDiagnosis: {exc}'
        ) from exc
