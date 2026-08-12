"""
title: Pydantic schemas for the execution pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class LifecycleEvent(str, Enum):
    CLINICAL_DATA_SET = 'clinical_data_set'
    ANSWERS_PROVIDED = 'answers_provided'
    SKILL_UI_DATA_PROVIDED = 'skill_ui_data_provided'
    STAGE_STARTED = 'stage_started'
    STAGE_COMPLETED = 'stage_completed'
    CHECK_REQUIREMENTS_STARTED = 'check_requirements_started'
    CHECK_REQUIREMENTS_COMPLETED = 'check_requirements_completed'
    INQUIRIES_RAISED = 'inquiries_raised'
    SKILL_STARTED = 'skill_started'
    SKILL_COMPLETED = 'skill_completed'
    SKILL_FAILED = 'skill_failed'
    SKILL_RESULT_RECORDED = 'skill_result_recorded'


class PromptFragment(BaseModel):
    """
    title: A piece of text contributed by a skill for the final prompt.
    attributes:
      stage:
        type: str
      skill_name:
        type: str
      title:
        type: str
      content:
        type: str
      priority:
        type: int
      include_in_final_prompt:
        type: bool
    """

    model_config = ConfigDict(extra='forbid')

    stage: str
    skill_name: str
    title: str
    content: str
    priority: int = 100
    include_in_final_prompt: bool = True


class SkillResult(BaseModel):
    """
    title: The standardized result of a single skill execution.
    attributes:
      stage:
        type: str
      skill_name:
        type: str
      status:
        type: Literal[succeeded, failed, skipped]
      summary:
        type: str
      data:
        type: dict[str, Any]
      prompt_fragments:
        type: list[PromptFragment] | None
    """

    model_config = ConfigDict(extra='forbid')

    stage: str
    skill_name: str
    status: Literal['succeeded', 'failed', 'skipped']
    summary: str = ''
    data: dict[str, Any] = {}
    prompt_fragments: list[PromptFragment] | None = None


class ExecutionStep(BaseModel):
    """
    title: Event log record tracking step-level pipeline execution.
    attributes:
      run_id:
        type: str
      stage:
        type: str
      skill_name:
        type: str
      hook:
        type: Literal[pre, execute, post]
      attempt:
        type: int
      input_hash:
        type: str
      status:
        type: Literal[started, completed, failed, skipped]
      error_data:
        type: dict[str, Any] | None
    """

    model_config = ConfigDict(extra='forbid')

    run_id: str
    stage: str
    skill_name: str
    hook: Literal['pre', 'execute', 'post']
    attempt: int = 1
    input_hash: str
    status: Literal['started', 'completed', 'failed', 'skipped']
    error_data: dict[str, Any] | None = None


class AgentStep(BaseModel):
    """
    title: Sub-step tracked by complex multi-agent skills.
    attributes:
      name:
        type: str
      description:
        type: str
    """

    model_config = ConfigDict(extra='forbid')

    name: str
    description: str


class AgentStepResult(BaseModel):
    """
    title: The outcome of an internal sub-step inside an agent skill.
    attributes:
      name:
        type: str
      status:
        type: Literal[succeeded, failed, skipped]
      data:
        type: dict[str, Any]
    """

    model_config = ConfigDict(extra='forbid')

    name: str
    status: Literal['succeeded', 'failed', 'skipped']
    data: dict[str, Any] = {}
