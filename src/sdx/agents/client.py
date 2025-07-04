"""Shared OpenAI helper for all agents."""

from __future__ import annotations

import json
import os

from pathlib import Path
from typing import Any, Dict, cast

from dotenv import load_dotenv
from openai import OpenAI

# load environment once (root/.envs/.env)
load_dotenv(Path(__file__).parents[3] / '.envs' / '.env')

_MODEL_NAME = 'o4-mini-2025-04-16'
_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', ''))


def chat(system: str, user: str) -> Dict[str, Any]:
    """Return parsed JSON from the first chat completion.

    Parameters
    ----------
    system
        System role instructions.
    user
        User prompt (plain text or JSON string).

    Raises
    ------
    ValueError
        If the model does not return valid JSON.
    """
    rsp = _client.chat.completions.create(
        model=_MODEL_NAME,
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
    )
    try:
        return cast(
            Dict[str, Any], json.loads(rsp.choices[0].message.content or '{}')
        )
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise ValueError('OpenAI response is not valid JSON') from exc
