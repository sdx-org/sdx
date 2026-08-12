"""
title: Example multi-agent skill demonstrating the agent-step protocol.
"""

from __future__ import annotations

from hiperhealth.pipeline.context import PipelineContext
from hiperhealth.pipeline.models import AgentStep, AgentStepResult
from hiperhealth.pipeline.skill import BaseSkill, SkillMetadata
from hiperhealth.pipeline.stages import Stage


class GutMicrobiomeAgentSkill(BaseSkill):
    """
    title: An example multi-agent skill.
    """

    def __init__(self) -> None:
        """
        title: Initialize the GutMicrobiomeAgentSkill.
        """
        super().__init__(
            SkillMetadata(
                name='hiperhealth.gut_microbiome',
                version='0.1.0',
                stages=(Stage.DIAGNOSIS,),
                description='Multi-agent gut microbiome analysis.',
            )
        )

    def plan_steps(self, stage: str, ctx: PipelineContext) -> list[AgentStep]:
        """
        title: Plan the steps for microbiome analysis.
        parameters:
          stage:
            type: str
          ctx:
            type: PipelineContext
        returns:
          type: list[AgentStep]
        """
        if stage != Stage.DIAGNOSIS:
            return []

        return [
            AgentStep(
                name='diet_analysis', description='Analyze patient diet.'
            ),
            AgentStep(name='lab_analysis', description='Analyze lab results.'),
            AgentStep(name='synthesis', description='Synthesize findings.'),
        ]

    def execute_step(
        self, stage: str, step: AgentStep, ctx: PipelineContext
    ) -> AgentStepResult:
        """
        title: Execute a planned step.
        parameters:
          stage:
            type: str
          step:
            type: AgentStep
          ctx:
            type: PipelineContext
        returns:
          type: AgentStepResult
        """
        if step.name == 'diet_analysis':
            return AgentStepResult(
                name=step.name,
                status='succeeded',
                data={'diet_score': 85, 'notes': 'High fiber diet.'},
            )
        elif step.name == 'lab_analysis':
            return AgentStepResult(
                name=step.name,
                status='succeeded',
                data={'microbiome_diversity': 'high'},
            )
        elif step.name == 'synthesis':
            return AgentStepResult(
                name=step.name,
                status='succeeded',
                data={'conclusion': 'Healthy gut microbiome.'},
            )

        return AgentStepResult(
            name=step.name,
            status='failed',
            data={'error': 'Unknown step.'},
        )

    def reduce_steps(
        self, stage: str, results: list[AgentStepResult], ctx: PipelineContext
    ) -> PipelineContext:
        """
        title: Synthesize all agent step results into the final context.
        parameters:
          stage:
            type: str
          results:
            type: list[AgentStepResult]
          ctx:
            type: PipelineContext
        returns:
          type: PipelineContext
        """
        if stage != Stage.DIAGNOSIS:
            return ctx

        # Accumulate data from all successful steps
        final_data = {}
        for res in results:
            if res.status == 'succeeded':
                final_data.update(res.data)

        # Write to ctx.results
        ctx.results[stage] = {'microbiome_analysis': final_data}
        return ctx


__all__ = ['GutMicrobiomeAgentSkill']
