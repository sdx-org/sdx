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
        super().__init__(
            SkillMetadata(
                name='test.skill_a',
                stages=(Stage.DIAGNOSIS,),
            )
        )

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        ctx.results[stage] = {'source': 'Skill A'}
        return ctx


class _DummySkillB(BaseSkill):
    def __init__(self) -> None:
        super().__init__(
            SkillMetadata(
                name='test.skill_b',
                stages=(Stage.DIAGNOSIS,),
            )
        )

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        ctx.results[stage] = {'source': 'Skill B'}
        return ctx


class _DummySkillC(BaseSkill):
    def __init__(self) -> None:
        super().__init__(
            SkillMetadata(
                name='test.skill_c',
                stages=(Stage.DIAGNOSIS,),
            )
        )

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        raise ValueError('Expected crash')


class TestSessionPersistence:
    def test_per_skill_persistence_in_session(self, tmp_path: Any) -> None:
        """
        title: Verify that SkillResult objects are correctly persisted and reconstructed.
        parameters:
          tmp_path:
            type: Any
        """
        session_path = tmp_path / 'test_persistence_session.parquet'
        session = Session(session_path)

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
        title: Verify that exceptions during execution flush failed steps to parquet.
        parameters:
          tmp_path:
            type: Any
        """
        import pytest

        session_path = tmp_path / 'test_failure_session.parquet'
        session = Session(session_path)

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
        
        c_started = [s for s in steps if s.skill_name == 'test.skill_c' and s.status == 'started']
        c_failed = [s for s in steps if s.skill_name == 'test.skill_c' and s.status == 'failed']
        
        assert len(c_started) == 2  # 'pre' started, 'execute' started
        assert len(c_failed) == 1   # 'execute' failed
        assert c_failed[0].error_data is not None
        assert 'Expected crash' in c_failed[0].error_data['error']

        # Verify stage_completed was NOT recorded
        assert Stage.DIAGNOSIS not in reloaded_session.stages_completed

