"""
title: Baseline tests documenting current pipeline execution behavior.
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
        title: Initialize the dummy skill A.
        """
        super().__init__(
            SkillMetadata(
                name='test.skill_a',
                stages=(Stage.DIAGNOSIS,),
            )
        )

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        """
        title: Execute dummy skill A.
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
        title: Initialize the dummy skill B.
        """
        super().__init__(
            SkillMetadata(
                name='test.skill_b',
                stages=(Stage.DIAGNOSIS,),
            )
        )

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        """
        title: Execute dummy skill B.
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


class TestStageRunnerBaseline:
    def test_overwrite_flaw_baseline(self) -> None:
        """
        title: Document the overwrite flaw where Skill B erases Skill A.
        """
        runner = StageRunner(skills=[_DummySkillA(), _DummySkillB()])
        ctx = PipelineContext()
        ctx = runner.run(Stage.DIAGNOSIS, ctx)

        # Verify that Skill A's and Skill B's results are both preserved
        assert 'test.skill_a' in ctx.skill_results[Stage.DIAGNOSIS]
        assert ctx.skill_results[Stage.DIAGNOSIS]['test.skill_a'].data == {
            'source': 'Skill A'
        }

        assert 'test.skill_b' in ctx.skill_results[Stage.DIAGNOSIS]
        assert ctx.skill_results[Stage.DIAGNOSIS]['test.skill_b'].data == {
            'source': 'Skill B'
        }

    def test_coarse_stage_logging_baseline(self, tmp_path: Any) -> None:
        """
        title: Document that the runner only logs coarse stage completion.
        parameters:
          tmp_path:
            type: Any
        """
        session_path = tmp_path / 'test_session.parquet'
        session = Session.create(session_path)

        runner = StageRunner(skills=[_DummySkillA(), _DummySkillB()])

        # Run the session
        runner.run_session(Stage.DIAGNOSIS, session)

        # We expect a 'stage_started' and a 'stage_completed' event
        event_types = [event['event_type'] for event in session.events]
        assert 'stage_started' in event_types
        assert 'stage_completed' in event_types

        # Verify that per-skill events are recorded
        assert 'skill_started' in event_types
        assert 'skill_completed' in event_types
        assert 'skill_result_recorded' in event_types
