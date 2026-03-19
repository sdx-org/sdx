"""
title: StageRunner — executes pipeline stages independently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hiperhealth.pipeline.context import AuditEntry, PipelineContext
from hiperhealth.pipeline.session import Inquiry
from hiperhealth.pipeline.skill import Skill

if TYPE_CHECKING:
    from hiperhealth.pipeline.registry import SkillRegistry
    from hiperhealth.pipeline.session import Session


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

    # ── Session-aware methods ──────────────────────────────────────

    def check_requirements(
        self,
        stage: str,
        session: Session,
        **kwargs: Any,
    ) -> list[Inquiry]:
        """
        title: Ask relevant skills what information they need.
        summary: |-
          Builds a PipelineContext from the session, calls
          ``skill.check_requirements()`` for every skill registered
          on the given stage, and records events in the session file.
          Inquiries are returned with three priority levels:
          - required: must have before this stage can run
          - supplementary: improves results, available now
          - deferred: only available after a future pipeline step
        parameters:
          stage:
            type: str
          session:
            type: Session
          kwargs:
            type: Any
            variadic: keyword
        returns:
          type: list[Inquiry]
        """
        ctx = session.to_context()
        ctx.extras['_run_kwargs'] = kwargs
        session.record_event('check_requirements_started', stage=stage)

        relevant = [s for s in self._skills if stage in s.metadata.stages]
        all_inquiries: list[Inquiry] = []

        for skill in relevant:
            inquiries = skill.check_requirements(stage, ctx)
            if inquiries:
                session.record_event(
                    'inquiries_raised',
                    stage=stage,
                    skill_name=skill.metadata.name,
                    data={
                        'inquiries': [i.model_dump() for i in inquiries],
                    },
                )
                all_inquiries.extend(inquiries)

        session.record_event(
            'check_requirements_completed',
            stage=stage,
            data={'total_inquiries': len(all_inquiries)},
        )
        return all_inquiries

    def run_session(
        self,
        stage: str,
        session: Session,
        **kwargs: Any,
    ) -> Session:
        """
        title: Execute a stage using the session file.
        summary: |-
          Builds a PipelineContext from the session, runs the stage
          with the existing ``run()`` method, then writes results
          back to the session parquet.
        parameters:
          stage:
            type: str
          session:
            type: Session
          kwargs:
            type: Any
            variadic: keyword
        returns:
          type: Session
        """
        ctx = session.to_context()
        session.record_event('stage_started', stage=stage)
        ctx = self.run(stage, ctx, **kwargs)
        session.update_from_context(stage, ctx)
        return session
