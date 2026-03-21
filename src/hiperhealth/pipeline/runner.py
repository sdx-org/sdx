"""
title: StageRunner — executes pipeline stages independently.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator
from contextlib import contextmanager
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
      _disabled_skill_names:
        type: set[str]
    """

    def __init__(
        self,
        skills: list[Skill] | None = None,
        registry: SkillRegistry | None = None,
    ) -> None:
        """
        title: Initialize the stage runner with optional skills.
        parameters:
          skills:
            type: list[Skill] | None
          registry:
            type: SkillRegistry | None
        """
        self._skills: list[Skill] = list(skills or [])
        self._registry = registry
        self._disabled_skill_names: set[str] = set()

    def register(self, name: str, index: int | None = None) -> None:
        """
        title: Load a skill from the registry by name and activate it.
        summary: |-
          Looks up the skill in the attached SkillRegistry using either a
          built-in name, a canonical channel skill id such as
          ``tm.ayurveda``, or a legacy installed skill name; then it
          instantiates the skill and adds it to the execution list.
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
        """
        title: Insert a loaded skill into the execution order.
        parameters:
          skill:
            type: Skill
          index:
            type: int | None
        """
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

    @contextmanager
    def disabled(
        self,
        skill_names: str | Collection[str],
    ) -> Iterator[None]:
        """
        title: Temporarily disable one or more registered skills.
        summary: |-
          Disabled skills stay registered and installed, but are skipped
          during runner operations while the context is active.
        parameters:
          skill_names:
            type: str | Collection[str]
        returns:
          type: Iterator[None]
        """
        previous = set(self._disabled_skill_names)
        self._disabled_skill_names.update(
            self._normalize_skill_names(skill_names)
        )
        try:
            yield
        finally:
            self._disabled_skill_names = previous

    def run(
        self,
        stage: str,
        ctx: PipelineContext,
        *,
        disabled_skills: str | Collection[str] | None = None,
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
          disabled_skills:
            type: str | Collection[str] | None
          kwargs:
            type: Any
            variadic: keyword
        returns:
          type: PipelineContext
        """
        ctx.extras['_run_kwargs'] = kwargs
        return self._run_stage(stage, ctx, disabled_skills=disabled_skills)

    def run_many(
        self,
        stages: list[str],
        ctx: PipelineContext,
        *,
        disabled_skills: str | Collection[str] | None = None,
        **kwargs: Any,
    ) -> PipelineContext:
        """
        title: Run multiple stages sequentially.
        parameters:
          stages:
            type: list[str]
          ctx:
            type: PipelineContext
          disabled_skills:
            type: str | Collection[str] | None
          kwargs:
            type: Any
            variadic: keyword
        returns:
          type: PipelineContext
        """
        for stage in stages:
            ctx = self.run(
                stage,
                ctx,
                disabled_skills=disabled_skills,
                **kwargs,
            )
        return ctx

    def _normalize_skill_names(
        self,
        skill_names: str | Collection[str] | None,
    ) -> set[str]:
        """
        title: Normalize one or many skill names into a set.
        parameters:
          skill_names:
            type: str | Collection[str] | None
        returns:
          type: set[str]
        """
        if skill_names is None:
            return set()
        if isinstance(skill_names, str):
            return {skill_names}
        return set(skill_names)

    def _relevant_skills(
        self,
        stage: str,
        disabled_skills: str | Collection[str] | None = None,
    ) -> list[Skill]:
        """
        title: Return the registered skills that apply to a stage.
        parameters:
          stage:
            type: str
          disabled_skills:
            type: str | Collection[str] | None
        returns:
          type: list[Skill]
        """
        disabled_names = self._disabled_skill_names.union(
            self._normalize_skill_names(disabled_skills)
        )
        return [
            skill
            for skill in self._skills
            if stage in skill.metadata.stages
            and skill.metadata.name not in disabled_names
        ]

    def _run_stage(
        self,
        stage: str,
        ctx: PipelineContext,
        *,
        disabled_skills: str | Collection[str] | None = None,
    ) -> PipelineContext:
        """
        title: Execute pre, execute, and post hooks for one stage.
        parameters:
          stage:
            type: str
          ctx:
            type: PipelineContext
          disabled_skills:
            type: str | Collection[str] | None
        returns:
          type: PipelineContext
        """
        relevant = self._relevant_skills(
            stage,
            disabled_skills=disabled_skills,
        )

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
        *,
        disabled_skills: str | Collection[str] | None = None,
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
          disabled_skills:
            type: str | Collection[str] | None
          kwargs:
            type: Any
            variadic: keyword
        returns:
          type: list[Inquiry]
        """
        ctx = session.to_context()
        ctx.extras['_run_kwargs'] = kwargs
        session.record_event('check_requirements_started', stage=stage)

        relevant = self._relevant_skills(
            stage,
            disabled_skills=disabled_skills,
        )
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
        *,
        disabled_skills: str | Collection[str] | None = None,
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
          disabled_skills:
            type: str | Collection[str] | None
          kwargs:
            type: Any
            variadic: keyword
        returns:
          type: Session
        """
        ctx = session.to_context()
        session.record_event('stage_started', stage=stage)
        ctx = self.run(
            stage,
            ctx,
            disabled_skills=disabled_skills,
            **kwargs,
        )
        session.update_from_context(stage, ctx)
        return session
