"""
title: StageRunner — executes pipeline stages independently.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from hiperhealth.pipeline.context import AuditEntry, PipelineContext
from hiperhealth.pipeline.models import (
    ExecutionStep,
    LifecycleEvent,
    SkillResult,
)
from hiperhealth.pipeline.session import Inquiry
from hiperhealth.pipeline.skill import Skill

if TYPE_CHECKING:
    from hiperhealth.pipeline.registry import SkillRegistry
    from hiperhealth.pipeline.session import Session


def _compute_input_hash(ctx: PipelineContext) -> str:
    """
    title: Compute a deterministic hash of the current pipeline state.
    parameters:
      ctx:
        type: PipelineContext
    returns:
      type: str
    """
    data = {
        'patient': ctx.patient,
        'language': ctx.language,
    }
    serialized = json.dumps(data, sort_keys=True)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


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

    @property
    def registry(self) -> SkillRegistry | None:
        """
        title: Return the registry used to load named skills.
        returns:
          type: SkillRegistry | None
        """
        return self._registry

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
        resume: bool = True,
        force: bool = False,
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
          resume:
            type: bool
          force:
            type: bool
          kwargs:
            type: Any
            variadic: keyword
        returns:
          type: PipelineContext
        """
        ctx.extras['_run_kwargs'] = kwargs
        return self._run_stage(
            stage,
            ctx,
            disabled_skills=disabled_skills,
            resume=resume,
            force=force,
        )

    def run_many(
        self,
        stages: list[str],
        ctx: PipelineContext,
        *,
        disabled_skills: str | Collection[str] | None = None,
        resume: bool = True,
        force: bool = False,
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
          resume:
            type: bool
          force:
            type: bool
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
                resume=resume,
                force=force,
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
        resume: bool = True,
        force: bool = False,
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
          resume:
            type: bool
          force:
            type: bool
        returns:
          type: PipelineContext
        """
        relevant = self._relevant_skills(
            stage,
            disabled_skills=disabled_skills,
        )

        run_id = ctx.session_id or 'unknown'
        if stage not in ctx.skill_results:
            ctx.skill_results[stage] = {}

        for hook in ('pre', 'execute', 'post'):
            for skill in relevant:
                skill_name = skill.metadata.name
                input_hash = _compute_input_hash(ctx)

                all_steps = list(ctx.execution_steps)
                for step_data in ctx.extras.get('past_steps', []):
                    all_steps.append(ExecutionStep.model_validate(step_data))

                past_steps = [
                    s
                    for s in all_steps
                    if s.stage == stage
                    and s.skill_name == skill_name
                    and s.hook == hook
                ]

                attempt = 1
                if past_steps:
                    attempt = max(s.attempt for s in past_steps)
                    last_step = past_steps[-1]
                    if (
                        resume
                        and not force
                        and last_step.status == 'completed'
                    ):
                        if last_step.input_hash == input_hash:
                            ctx.execution_steps.append(
                                ExecutionStep(
                                    run_id=run_id,
                                    stage=stage,
                                    skill_name=skill_name,
                                    hook=hook,
                                    attempt=attempt,
                                    input_hash=input_hash,
                                    status='skipped',
                                )
                            )
                            continue

                    if last_step.status in ('failed', 'skipped', 'started'):
                        attempt += 1

                ctx.execution_steps.append(
                    ExecutionStep(
                        run_id=run_id,
                        stage=stage,
                        skill_name=skill_name,
                        hook=hook,
                        attempt=attempt,
                        input_hash=input_hash,
                        status='started',
                    )
                )

                if hook == 'execute':
                    _legacy_result_before = copy.deepcopy(
                        ctx.results.get(stage, {})
                    )
                else:
                    _legacy_result_before = None

                try:
                    func = getattr(skill, hook)
                    ctx = func(stage, ctx)
                    status = 'completed'
                    error_data = None
                except Exception as e:
                    status = 'failed'
                    error_data = {'error': str(e)}
                    ctx.execution_steps.append(
                        ExecutionStep(
                            run_id=run_id,
                            stage=stage,
                            skill_name=skill_name,
                            hook=hook,
                            attempt=attempt,
                            input_hash=input_hash,
                            status=status,
                            error_data=error_data,
                        )
                    )
                    ctx.skill_results[stage][skill_name] = SkillResult(
                        stage=stage,
                        skill_name=skill_name,
                        status='failed',
                        summary=str(e),
                    )
                    raise

                ctx.execution_steps.append(
                    ExecutionStep(
                        run_id=run_id,
                        stage=stage,
                        skill_name=skill_name,
                        hook=hook,
                        attempt=attempt,
                        input_hash=input_hash,
                        status=status,
                    )
                )

                ctx.audit.append(
                    AuditEntry(
                        stage=stage,
                        skill_name=skill_name,
                        hook=hook,
                    )
                )

                # Implicitly save legacy skill results to avoid overwrites
                # If a previous run failed, we should overwrite it on success.
                if hook == 'execute':
                    existing = ctx.skill_results[stage].get(skill_name)
                    if not existing or existing.status != 'succeeded':
                        current_data = ctx.results.get(stage, {})
                        if current_data != _legacy_result_before:
                            data = current_data
                            if isinstance(data, dict):
                                data = copy.deepcopy(data)
                        else:
                            data = {'legacy_result': data}

                        ctx.skill_results[stage][skill_name] = SkillResult(
                            stage=stage,
                            skill_name=skill_name,
                            status='succeeded',
                            data=data,
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
        session.record_event(
            LifecycleEvent.CHECK_REQUIREMENTS_STARTED, stage=stage
        )

        relevant = self._relevant_skills(
            stage,
            disabled_skills=disabled_skills,
        )
        all_inquiries: list[Inquiry] = []

        for skill in relevant:
            inquiries = skill.check_requirements(stage, ctx)
            if inquiries:
                session.record_event(
                    LifecycleEvent.INQUIRIES_RAISED,
                    stage=stage,
                    skill_name=skill.metadata.name,
                    data={
                        'inquiries': [i.model_dump() for i in inquiries],
                    },
                )
                all_inquiries.extend(inquiries)

        session.record_event(
            LifecycleEvent.CHECK_REQUIREMENTS_COMPLETED,
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
        resume: bool = True,
        force: bool = False,
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
          resume:
            type: bool
          force:
            type: bool
          kwargs:
            type: Any
            variadic: keyword
        returns:
          type: Session
        """
        ctx = session.to_context()
        session.record_event(LifecycleEvent.STAGE_STARTED, stage=stage)
        completed = False
        try:
            ctx = self.run(
                stage,
                ctx,
                disabled_skills=disabled_skills,
                resume=resume,
                force=force,
                **kwargs,
            )
            completed = True
        finally:
            session.update_from_context(stage, ctx, completed=completed)
        return session
