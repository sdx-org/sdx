"""
title: Tests for per-skill persistence and session replay.
"""

from __future__ import annotations

from typing import Any

from hiperhealth.pipeline import (
    BaseSkill,
    PipelineContext,
    SkillMetadata,
    Stage,
    StageRunner,
)
from hiperhealth.pipeline.session import Session


class _DummySkillA(BaseSkill):
    def __init__(self) -> None:
        """
        title: Initialize Dummy Skill A.
        """
        super().__init__(
            SkillMetadata(
                name='test.skill_a',
                stages=(Stage.DIAGNOSIS,),
            )
        )

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        """
        title: Execute Dummy Skill A.
        parameters:
          stage:
            type: str
          ctx:
            type: PipelineContext
        returns:
          type: PipelineContext
        """
        ctx.results[stage] = {'source': 'Skill A'}
        return ctx


class _DummySkillB(BaseSkill):
    def __init__(self) -> None:
        """
        title: Initialize Dummy Skill B.
        """
        super().__init__(
            SkillMetadata(
                name='test.skill_b',
                stages=(Stage.DIAGNOSIS,),
            )
        )

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        """
        title: Execute Dummy Skill B.
        parameters:
          stage:
            type: str
          ctx:
            type: PipelineContext
        returns:
          type: PipelineContext
        """
        ctx.results[stage] = {'source': 'Skill B'}
        return ctx


class _DummySkillC(BaseSkill):
    def __init__(self) -> None:
        """
        title: Initialize Dummy Skill C.
        """
        super().__init__(
            SkillMetadata(
                name='test.skill_c',
                stages=(Stage.DIAGNOSIS,),
            )
        )

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        """
        title: Execute Dummy Skill C.
        parameters:
          stage:
            type: str
          ctx:
            type: PipelineContext
        returns:
          type: PipelineContext
        """
        raise ValueError('Expected crash')


class TestSessionPersistence:
    def test_per_skill_persistence_in_session(self, tmp_path: Any) -> None:
        """
        title: Verify SkillResult objects are persisted and reconstructed.
        parameters:
          tmp_path:
            type: Any
        """
        session_path = tmp_path / 'test_persistence_session.parquet'
        session = Session.create(session_path)

        runner = StageRunner(skills=[_DummySkillA(), _DummySkillB()])
        runner.run_session(Stage.DIAGNOSIS, session)

        # Reload session to ensure we're reading from the parquet file
        reloaded_session = Session.load(session_path)

        skill_results = reloaded_session.skill_results

        assert Stage.DIAGNOSIS in skill_results

        res_a = skill_results[Stage.DIAGNOSIS]['test.skill_a']
        assert res_a.skill_name == 'test.skill_a'
        assert res_a.status == 'succeeded'
        assert res_a.data == {'source': 'Skill A'}

        res_b = skill_results[Stage.DIAGNOSIS]['test.skill_b']
        assert res_b.skill_name == 'test.skill_b'
        assert res_b.status == 'succeeded'
        assert res_b.data == {'source': 'Skill B'}

    def test_per_skill_failure_persistence_in_session(
        self, tmp_path: Any
    ) -> None:
        """
        title: Verify exceptions during execution flush failed steps.
        parameters:
          tmp_path:
            type: Any
        """
        import pytest

        session_path = tmp_path / 'test_failure_session.parquet'
        session = Session.create(session_path)

        runner = StageRunner(skills=[_DummySkillA(), _DummySkillC()])
        with pytest.raises(ValueError, match='Expected crash'):
            runner.run_session(Stage.DIAGNOSIS, session)

        reloaded_session = Session.load(session_path)

        # Verify SkillResult was created with 'failed' status
        skill_results = reloaded_session.skill_results
        assert Stage.DIAGNOSIS in skill_results

        res_a = skill_results[Stage.DIAGNOSIS]['test.skill_a']
        assert res_a.status == 'succeeded'

        res_c = skill_results[Stage.DIAGNOSIS]['test.skill_c']
        assert res_c.status == 'failed'
        assert 'Expected crash' in res_c.summary

        # Verify ExecutionSteps were flushed properly
        steps = reloaded_session.execution_steps
        assert len(steps) > 0

        c_started = [
            s
            for s in steps
            if s.skill_name == 'test.skill_c' and s.status == 'started'
        ]
        c_failed = [
            s
            for s in steps
            if s.skill_name == 'test.skill_c' and s.status == 'failed'
        ]

        assert len(c_started) == 2  # 'pre' started, 'execute' started
        assert len(c_failed) == 1  # 'execute' failed
        assert c_failed[0].error_data is not None
        assert 'Expected crash' in c_failed[0].error_data['error']

        # Verify stage_completed was NOT recorded
        assert Stage.DIAGNOSIS not in reloaded_session.stages_completed

    def test_partial_failure_and_resume(self, tmp_path: Any) -> None:
        """
        title: >-
          Verify that resume=True skips completed steps and retries failed
          ones.
        parameters:
          tmp_path:
            type: Any
        """
        import pytest

        session_path = tmp_path / 'test_resume_session.parquet'
        session = Session.create(session_path)

        runner = StageRunner(
            skills=[_DummySkillA(), _DummySkillC(), _DummySkillB()]
        )
        with pytest.raises(ValueError, match='Expected crash'):
            runner.run_session(Stage.DIAGNOSIS, session)

        # Now reload and resume
        reloaded_session = Session.load(session_path)

        class _FixedSkillC(_DummySkillC):
            def execute(
                self, stage: str, ctx: PipelineContext
            ) -> PipelineContext:
                """
                title: Execute Fixed Skill C.
                parameters:
                  stage:
                    type: str
                  ctx:
                    type: PipelineContext
                returns:
                  type: PipelineContext
                """
                ctx.results[stage] = {'source': 'Skill C'}
                return ctx

        fixed_runner = StageRunner(
            skills=[_DummySkillA(), _FixedSkillC(), _DummySkillB()]
        )
        fixed_runner.run_session(
            Stage.DIAGNOSIS, reloaded_session, resume=True
        )

        final_session = Session.load(session_path)
        skill_results = final_session.skill_results

        # Verify all three completed
        assert (
            skill_results[Stage.DIAGNOSIS]['test.skill_a'].status
            == 'succeeded'
        )
        assert (
            skill_results[Stage.DIAGNOSIS]['test.skill_c'].status
            == 'succeeded'
        )
        assert (
            skill_results[Stage.DIAGNOSIS]['test.skill_b'].status
            == 'succeeded'
        )

        # Verify A was skipped on retry
        steps_a_exec = [
            s
            for s in final_session.execution_steps
            if s.skill_name == 'test.skill_a' and s.hook == 'execute'
        ]
        assert steps_a_exec[0].status == 'started'
        assert steps_a_exec[1].status == 'completed'
        assert steps_a_exec[2].status == 'skipped'

        # Verify C was retried (attempt=1 failed, attempt=2 succeeded)
        steps_c_exec = [
            s
            for s in final_session.execution_steps
            if s.skill_name == 'test.skill_c' and s.hook == 'execute'
        ]
        assert steps_c_exec[0].status == 'started'
        assert steps_c_exec[0].attempt == 1
        assert steps_c_exec[1].status == 'failed'
        assert steps_c_exec[1].attempt == 1
        assert steps_c_exec[2].status == 'started'
        assert steps_c_exec[2].attempt == 2
        assert steps_c_exec[3].status == 'completed'
        assert steps_c_exec[3].attempt == 2
