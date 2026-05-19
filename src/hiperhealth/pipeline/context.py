"""
title: Pipeline context — the serializable data carrier between stages.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class AuditEntry(BaseModel):
    """
    title: A single audit record produced each time a skill hook runs.
    attributes:
      stage:
        type: str
      skill_name:
        type: str
      hook:
        type: str
      timestamp:
        type: datetime
      metadata:
        type: dict[str, Any]
    """

    stage: str
    skill_name: str
    hook: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineContext(BaseModel):
    """
    title: >-
      Serializable context that flows between independently executed stages.
    summary: |-
      Callers can serialize this to JSON between invocations that may
      happen hours or days apart, by different actors.
    attributes:
      patient:
        type: dict[str, Any]
      language:
        type: str
      session_id:
        type: str | None
      results:
        type: dict[str, Any]
      audit:
        type: list[AuditEntry]
      extras:
        type: dict[str, Any]
    """

    patient: dict[str, Any] = Field(default_factory=dict)
    language: str = 'en'
    session_id: str | None = None
    results: dict[str, Any] = Field(default_factory=dict)
    audit: list[AuditEntry] = Field(default_factory=list)
    extras: dict[str, Any] = Field(default_factory=dict)


def get_skill_ui_values(
    ctx: PipelineContext,
    skill_id: str,
    view_id: str | None = None,
) -> dict[str, Any]:
    """
    title: Return skill-specific UI values stored in pipeline context extras.
    parameters:
      ctx:
        type: PipelineContext
      skill_id:
        type: str
      view_id:
        type: str | None
    returns:
      type: dict[str, Any]
    """
    skill_ui = ctx.extras.get('skill_ui', {})
    if not isinstance(skill_ui, dict):
        return {}
    skill_values = skill_ui.get(skill_id, {})
    if not isinstance(skill_values, dict):
        return {}
    if view_id is None:
        return dict(skill_values)
    view_values = skill_values.get(view_id, {})
    if not isinstance(view_values, dict):
        return {}
    return dict(view_values)
