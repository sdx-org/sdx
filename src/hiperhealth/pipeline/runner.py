"""
title: StageRunner — executes pipeline stages independently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hiperhealth.pipeline.context import AuditEntry, PipelineContext
from hiperhealth.pipeline.skill import Skill

if TYPE_CHECKING:
    from hiperhealth.pipeline.registry import SkillRegistry


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
      _registry:
        description: Value for _registry.
    """

    def __init__(
        self,
        skills: list[Skill] | None = None,
        registry: SkillRegistry | None = None,
    ) -> None:
        self._skills: list[Skill] = list(skills or [])
        self._registry = registry

    def register(self, name: str, index: int | None = None) -> None:
        """
        title: Load a skill from the registry by name and activate it.
        summary: |-
          Looks up the skill in the attached SkillRegistry,
          instantiates it, and adds it to the execution list.
          Pass ``index`` to control execution order.
        parameters:
          name:
            type: str
          index:
            type: int | None
        """
        if self._registry is None:
            from hiperhealth.pipeline.registry import SkillRegistry

            self._registry = SkillRegistry()
        skill = self._registry.load(name)
        self._add_skill(skill, index=index)

    def _add_skill(self, skill: Skill, index: int | None = None) -> None:
        if index is not None:
            self._skills.insert(index, skill)
        else:
            self._skills.append(skill)

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
