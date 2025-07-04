"""Diagnostic-related LLM utilities."""

from __future__ import annotations

import json

from typing import Any, Dict, List

from sdx.agents.client import chat

_DIAG_PROMPT = (
    'You are an experienced physician assistant. '
    "Return a JSON object with keys 'summary' (two sentences) and "
    "'options' (array of differential diagnoses) given the patient data."
)

_EXAM_PROMPT = (
    'You are an experienced physician assistant. '
    "Given the selected diagnoses, return JSON with keys 'summary' and "
    "'options' (max 10 exam/procedure names)."
)


def differential(patient: Dict[str, Any]) -> Dict[str, Any]:
    """Return summary + list of differential diagnoses."""
    return chat(_DIAG_PROMPT, json.dumps(patient, ensure_ascii=False))


def exams(selected_dx: List[str]) -> Dict[str, Any]:
    """Return summary + list of suggested examinations."""
    return chat(_EXAM_PROMPT, json.dumps(selected_dx, ensure_ascii=False))
