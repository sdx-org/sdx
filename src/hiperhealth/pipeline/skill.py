"""
title: Skill interface and base class for pipeline plugins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from hiperhealth.pipeline.context import PipelineContext

if TYPE_CHECKING:
    from hiperhealth.pipeline.session import Inquiry


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

    def check_requirements(
        self, stage: str, ctx: PipelineContext
    ) -> list[Inquiry]: ...

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

    def check_requirements(
        self, stage: str, ctx: PipelineContext
    ) -> list[Inquiry]:
        """
        title: Determine what information is needed before execution.
        summary: |-
          Override to return a list of Inquiry objects describing
          what additional data the skill needs.  The default
          implementation returns an empty list (no extra data needed).
          Inquiries use three priority levels:
          - required: must have before this stage can run
          - supplementary: improves results, available now
          - deferred: only available after a future pipeline step
        parameters:
          stage:
            type: str
          ctx:
            type: PipelineContext
        returns:
          type: list[Inquiry]
        """
        return []

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
