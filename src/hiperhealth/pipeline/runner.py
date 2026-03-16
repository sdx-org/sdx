"""
title: StageRunner — executes pipeline stages independently.
"""

from __future__ import annotations

from typing import Any

from hiperhealth.pipeline.context import AuditEntry, PipelineContext
from hiperhealth.pipeline.skill import Skill


class StageRunner:
    """
    title: Executes one or more pipeline stages with registered skills.
    summary: |-
      Each stage can be run independently, at any time, by any actor.
      The primary API is ``run()`` for single-stage execution.
      ``run_many()`` is a convenience for sequential batch execution.
    attributes:
      _skills:
        type: list[Skill]
    """

    def __init__(self, skills: list[Skill] | None = None) -> None:
        self._skills: list[Skill] = sorted(
            skills or [],
            key=lambda s: s.metadata.priority,
        )

    def install(self, skill: Skill) -> None:
        """
        title: Add a skill at runtime.
        parameters:
          skill:
            type: Skill
        """
        self._skills.append(skill)
        self._skills.sort(key=lambda s: s.metadata.priority)

    @property
    def skills(self) -> list[Skill]:
        """
        title: Return the list of installed skills.
        returns:
          type: list[Skill]
        """
        return list(self._skills)

    def run(
        self,
        stage: str,
        ctx: PipelineContext,
        **kwargs: Any,
    ) -> PipelineContext:
        """
        title: Run a single stage. This is the primary API.
        summary: |-
          Extra keyword arguments (e.g. ``llm``, ``llm_settings``)
          are stored in ``ctx.extras['_run_kwargs']`` so skills can
          access them.
        parameters:
          stage:
            type: str
          ctx:
            type: PipelineContext
          kwargs:
            type: Any
            variadic: keyword
        returns:
          type: PipelineContext
        """
        ctx.extras['_run_kwargs'] = kwargs
        return self._run_stage(stage, ctx)

    def run_many(
        self,
        stages: list[str],
        ctx: PipelineContext,
        **kwargs: Any,
    ) -> PipelineContext:
        """
        title: Run multiple stages sequentially.
        parameters:
          stages:
            type: list[str]
          ctx:
            type: PipelineContext
          kwargs:
            type: Any
            variadic: keyword
        returns:
          type: PipelineContext
        """
        for stage in stages:
            ctx = self.run(stage, ctx, **kwargs)
        return ctx

    def _run_stage(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        relevant = [s for s in self._skills if stage in s.metadata.stages]

        for skill in relevant:
            ctx = skill.pre(stage, ctx)
            ctx.audit.append(
                AuditEntry(
                    stage=stage,
                    skill_name=skill.metadata.name,
                    hook='pre',
                )
            )

        for skill in relevant:
            ctx = skill.execute(stage, ctx)
            ctx.audit.append(
                AuditEntry(
                    stage=stage,
                    skill_name=skill.metadata.name,
                    hook='execute',
                )
            )

        for skill in relevant:
            ctx = skill.post(stage, ctx)
            ctx.audit.append(
                AuditEntry(
                    stage=stage,
                    skill_name=skill.metadata.name,
                    hook='post',
                )
            )

        return ctx
