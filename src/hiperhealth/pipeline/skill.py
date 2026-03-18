"""
title: Skill interface and base class for pipeline plugins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from hiperhealth.pipeline.context import PipelineContext


@dataclass(frozen=True)
class SkillMetadata:
    """
    title: Declarative metadata for a skill.
    attributes:
      name:
        type: str
      version:
        type: str
      stages:
        type: tuple[str, Ellipsis]
      description:
        type: str
    """

    name: str
    version: str = '0.1.0'
    stages: tuple[str, ...] = ()
    description: str = ''


@runtime_checkable
class Skill(Protocol):
    """
    title: Protocol that every skill must satisfy.
    attributes:
      metadata:
        type: SkillMetadata
    """

    metadata: SkillMetadata

    def pre(self, stage: str, ctx: PipelineContext) -> PipelineContext: ...

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext: ...

    def post(self, stage: str, ctx: PipelineContext) -> PipelineContext: ...


class BaseSkill:
    """
    title: Convenience base class with no-op hook defaults.
    summary: Subclass and override only the hooks you need.
    attributes:
      metadata:
        type: SkillMetadata
    """

    metadata: SkillMetadata

    def __init__(self, metadata: SkillMetadata) -> None:
        self.metadata = metadata

    def pre(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        """
        title: Called before the stage's main execution.
        parameters:
          stage:
            type: str
          ctx:
            type: PipelineContext
        returns:
          type: PipelineContext
        """
        return ctx

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        """
        title: The main execution hook for a stage.
        parameters:
          stage:
            type: str
          ctx:
            type: PipelineContext
        returns:
          type: PipelineContext
        """
        return ctx

    def post(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        """
        title: Called after the stage's main execution.
        parameters:
          stage:
            type: str
          ctx:
            type: PipelineContext
        returns:
          type: PipelineContext
        """
        return ctx
