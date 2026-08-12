"""
title: Tests for multi-agent step pipeline behavior.
"""

from __future__ import annotations

import pytest

from hiperhealth.pipeline import (
    BaseSkill,
    PipelineContext,
    SkillMetadata,
    Stage,
    StageRunner,
)
from hiperhealth.pipeline.models import AgentStep, AgentStepResult
from hiperhealth.skills.diagnostics.multi_agent_demo import (
    GutMicrobiomeAgentSkill,
)


class TestAgentSteps:
    def test_runner_executes_agent_steps(self) -> None:
        """
        title: Verify that runner successfully plans, executes, and reduces agent steps.
        """
        runner = StageRunner(skills=[GutMicrobiomeAgentSkill()])
        ctx = PipelineContext()
        ctx = runner.run(Stage.DIAGNOSIS, ctx)

        assert Stage.DIAGNOSIS in ctx.results
        res = ctx.results[Stage.DIAGNOSIS]

        # Verify the reduction actually wrote to ctx.results
        assert 'microbiome_analysis' in res
        analysis = res['microbiome_analysis']
        assert analysis['diet_score'] == 85
        assert analysis['microbiome_diversity'] == 'high'
        assert analysis['conclusion'] == 'Healthy gut microbiome.'

        # Verify agent_step_results are populated in SkillResult
        skill_res = ctx.skill_results[Stage.DIAGNOSIS][
            'hiperhealth.gut_microbiome'
        ]
        assert skill_res.agent_step_results is not None
        assert len(skill_res.agent_step_results) == 3
        assert skill_res.agent_step_results[0].name == 'diet_analysis'
        assert skill_res.agent_step_results[0].status == 'succeeded'

    def test_simple_skill_unaffected(self) -> None:
        """
        title: Verify that normal skills without plan_steps operate normally.
        """

        class NormalSkill(BaseSkill):
            def __init__(self) -> None:
                super().__init__(
                    SkillMetadata(
                        name='test.normal', stages=(Stage.DIAGNOSIS,)
                    )
                )

            def execute(
                self, stage: str, ctx: PipelineContext
            ) -> PipelineContext:
                ctx.results[stage] = {'normal_ran': True}
                return ctx

        runner = StageRunner(skills=[NormalSkill()])
        ctx = PipelineContext()
        ctx = runner.run(Stage.DIAGNOSIS, ctx)

        assert ctx.results[Stage.DIAGNOSIS].get('normal_ran') is True

        skill_res = ctx.skill_results[Stage.DIAGNOSIS]['test.normal']
        assert skill_res.status == 'succeeded'
        assert skill_res.agent_step_results is None

    def test_agent_step_failure_bubbles_up(self) -> None:
        """
        title: Verify that exceptions in execute_step bubble up and crash the run.
        """

        class FailingAgentSkill(BaseSkill):
            def __init__(self) -> None:
                super().__init__(
                    SkillMetadata(
                        name='test.fail_agent', stages=(Stage.DIAGNOSIS,)
                    )
                )

            def plan_steps(
                self, stage: str, ctx: PipelineContext
            ) -> list[AgentStep]:
                return [AgentStep(name='crash_step', description='Will crash')]

            def execute_step(
                self, stage: str, step: AgentStep, ctx: PipelineContext
            ) -> AgentStepResult:
                raise ValueError('Agent step crashed!')

        runner = StageRunner(skills=[FailingAgentSkill()])
        ctx = PipelineContext()

        with pytest.raises(ValueError, match='Agent step crashed!'):
            runner.run(Stage.DIAGNOSIS, ctx)
