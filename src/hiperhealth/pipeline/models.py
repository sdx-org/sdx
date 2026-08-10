"""
title: Pydantic schemas for the execution pipeline.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


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
      prompt_fragment:
        type: list[PromptFragment] | None
    """

    stage: str
    skill_name: str
    status: Literal['succeeded', 'failed', 'skipped']
    summary: str = ''
    data: dict[str, Any] = {}
    prompt_fragment: list[PromptFragment] | None = None


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

    name: str
    status: Literal['succeeded', 'failed', 'skipped']
    data: dict[str, Any] = {}
