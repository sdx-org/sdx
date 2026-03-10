"""Core types for the SDX module."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class FieldSpec(BaseModel):
    """UI-agnostic input field description."""

    id: str
    label: str
    kind: Literal[
        'text', 'number', 'select', 'multiselect', 'date', 'file'
    ] = 'text'
    required: bool = False
    options: list[str] = Field(default_factory=list)


class RequestSpec(BaseModel):
    """A request for more input."""

    message: str
    fields: list[FieldSpec] = Field(default_factory=list)


class StepResult(BaseModel):
    """Outcome emitted by a step."""

    status: Literal['more', 'route', 'done', 'abort'] = 'more'
    request: Optional[RequestSpec] = None
    next_step: Optional[str] = None
    notices: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class Prompt(BaseModel):
    """LLM prompt contract."""

    system: str
    user: str
    format: Literal['json', 'text'] = 'json'
    schema_hint: Optional[dict[str, Any]] = None


class LLMResponse(BaseModel):
    """Normalized LLM response."""

    raw: str
    parsed: Optional[dict[str, Any]] = None


class Context(BaseModel):
    """Conversation and patient context."""

    session_id: str
    patient: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    step_cursor: Optional[str] = None
    audit: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
