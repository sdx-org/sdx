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
