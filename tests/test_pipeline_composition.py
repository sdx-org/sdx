"""
title: Tests for prompt fragment compilation and composition.
"""

from __future__ import annotations

from hiperhealth.pipeline import (
    BaseSkill,
    PipelineContext,
    SkillMetadata,
    Stage,
    StageRunner,
)
from hiperhealth.pipeline.models import PromptFragment


class _DummyFragmentSkillA(BaseSkill):
    def __init__(self) -> None:
        super().__init__(
            SkillMetadata(
                name='test.frag_a',
                stages=(Stage.DIAGNOSIS,),
            )
        )

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        ctx.results[stage] = {'source': 'Frag A'}
        return ctx

    def compile_prompt_fragment(
        self, stage: str, ctx: PipelineContext
    ) -> PromptFragment | list[PromptFragment] | None:
        if stage == Stage.DIAGNOSIS:
            return PromptFragment(
                stage=stage,
                skill_name=self.metadata.name,
                title='Fragment A',
                content='Content from A',
                priority=200,
            )
        return None


class _DummyFragmentSkillB(BaseSkill):
    def __init__(self) -> None:
        super().__init__(
            SkillMetadata(
                name='test.frag_b',
                stages=(Stage.DIAGNOSIS,),
            )
        )

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        ctx.results[stage] = {'source': 'Frag B'}
        return ctx

    def compile_prompt_fragment(
        self, stage: str, ctx: PipelineContext
    ) -> PromptFragment | list[PromptFragment] | None:
        if stage == Stage.DIAGNOSIS:
            return PromptFragment(
                stage=stage,
                skill_name=self.metadata.name,
                title='Fragment B',
                content='Content from B',
                priority=100,  # Should appear before A despite registration order
            )
        return None


class _DummyFragmentSkillC(BaseSkill):
    def __init__(self) -> None:
        super().__init__(
            SkillMetadata(
                name='test.frag_c',
                stages=(Stage.DIAGNOSIS,),
            )
        )

    def compile_prompt_fragment(
        self, stage: str, ctx: PipelineContext
    ) -> PromptFragment | list[PromptFragment] | None:
        if stage == Stage.DIAGNOSIS:
            return [
                PromptFragment(
                    stage=stage,
                    skill_name=self.metadata.name,
                    title='Fragment C1',
                    content='Content C1',
                    priority=150,
                ),
                PromptFragment(
                    stage=stage,
                    skill_name=self.metadata.name,
                    title='Fragment C2',
                    content='Hidden Content',
                    include_in_final_prompt=False,
                ),
            ]
        return None


class TestPromptComposition:
    def test_prompt_compilation_and_sorting(self) -> None:
        """
        title: Verify that fragments are collected, sorted by priority, and formatted.
        """
        runner = StageRunner(
            skills=[
                _DummyFragmentSkillA(),
                _DummyFragmentSkillB(),
                _DummyFragmentSkillC(),
            ]
        )
        ctx = PipelineContext()
        ctx = runner.run(Stage.DIAGNOSIS, ctx)

        assert Stage.DIAGNOSIS in ctx.results
        result = ctx.results[Stage.DIAGNOSIS]
        assert 'composed_prompt' in result

        prompt_text = result['composed_prompt']

        # B (100) -> C1 (150) -> A (200)
        expected = (
            '### Fragment B\n'
            'Content from B\n\n'
            '### Fragment C1\n'
            'Content C1\n\n'
            '### Fragment A\n'
            'Content from A'
        )
        assert prompt_text == expected

    def test_fragments_attached_to_skill_results(self) -> None:
        """
        title: Verify that fragments are correctly attached to SkillResult objects.
        """
        runner = StageRunner(skills=[_DummyFragmentSkillC()])
        ctx = PipelineContext()
        # Mock execution steps logic that populates skill_results for execute hook
        ctx.results[Stage.DIAGNOSIS] = {}
        ctx = runner.run(Stage.DIAGNOSIS, ctx)

        res_c = ctx.skill_results[Stage.DIAGNOSIS]['test.frag_c']
        assert res_c.prompt_fragment is not None
        assert isinstance(res_c.prompt_fragment, list)
        assert len(res_c.prompt_fragment) == 2
        assert res_c.prompt_fragment[0].title == 'Fragment C1'
        assert res_c.prompt_fragment[1].title == 'Fragment C2'
