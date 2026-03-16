"""
title: Tests for the pipeline core engine.
"""

from __future__ import annotations

from hiperhealth.pipeline import (
    BaseSkill,
    PipelineContext,
    SkillMetadata,
    Stage,
    StageRunner,
)
from hiperhealth.pipeline.context import AuditEntry


class TestStage:
    def test_stage_values(self) -> None:
        assert Stage.SCREENING == 'screening'
        assert Stage.INTAKE == 'intake'
        assert Stage.DIAGNOSIS == 'diagnosis'
        assert Stage.EXAM == 'exam'
        assert Stage.TREATMENT == 'treatment'
        assert Stage.PRESCRIPTION == 'prescription'

    def test_stage_is_str(self) -> None:
        assert isinstance(Stage.DIAGNOSIS, str)


class TestPipelineContext:
    def test_default_context(self) -> None:
        ctx = PipelineContext()
        assert ctx.patient == {}
        assert ctx.language == 'en'
        assert ctx.session_id is None
        assert ctx.results == {}
        assert ctx.audit == []
        assert ctx.extras == {}

    def test_context_with_data(self) -> None:
        ctx = PipelineContext(
            patient={'symptoms': 'headache'},
            language='pt',
            session_id='abc123',
        )
        assert ctx.patient['symptoms'] == 'headache'
        assert ctx.language == 'pt'
        assert ctx.session_id == 'abc123'

    def test_context_serialization_roundtrip(self) -> None:
        ctx = PipelineContext(
            patient={'symptoms': 'fever', 'age': 30},
            language='es',
            session_id='sess-1',
            results={'diagnosis': {'summary': 'flu'}},
            extras={'custom_key': 'custom_value'},
        )
        json_str = ctx.model_dump_json()
        restored = PipelineContext.model_validate_json(json_str)

        assert restored.patient == ctx.patient
        assert restored.language == ctx.language
        assert restored.session_id == ctx.session_id
        assert restored.results == ctx.results
        assert restored.extras == ctx.extras

    def test_audit_entry_timestamp(self) -> None:
        entry = AuditEntry(
            stage='diagnosis',
            skill_name='test_skill',
            hook='execute',
        )
        assert entry.timestamp is not None
        assert entry.metadata == {}


class TestSkillMetadata:
    def test_defaults(self) -> None:
        meta = SkillMetadata(name='test')
        assert meta.name == 'test'
        assert meta.version == '0.1.0'
        assert meta.stages == ()
        assert meta.description == ''

    def test_custom_values(self) -> None:
        meta = SkillMetadata(
            name='ayurveda',
            version='1.0.0',
            stages=(Stage.DIAGNOSIS, Stage.TREATMENT),
            description='Ayurvedic perspective',
        )
        assert meta.stages == ('diagnosis', 'treatment')


class _CounterSkill(BaseSkill):
    """
    title: A test skill that counts hook invocations.
    attributes:
      pre_count:
        description: Value for pre_count.
      execute_count:
        description: Value for execute_count.
      post_count:
        description: Value for post_count.
    """

    def __init__(
        self,
        name: str = 'counter',
        stages: tuple[str, ...] = (Stage.DIAGNOSIS,),
    ) -> None:
        super().__init__(SkillMetadata(name=name, stages=stages))
        self.pre_count = 0
        self.execute_count = 0
        self.post_count = 0

    def pre(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        self.pre_count += 1
        return ctx

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        self.execute_count += 1
        ctx.results[stage] = f'{self.metadata.name}_executed'
        return ctx

    def post(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        self.post_count += 1
        return ctx


class _PromptFragmentSkill(BaseSkill):
    """
    title: Injects a prompt fragment in the pre hook.
    """

    def __init__(self) -> None:
        super().__init__(
            SkillMetadata(
                name='fragment_injector',
                stages=(Stage.DIAGNOSIS,),
            )
        )

    def pre(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        fragments = ctx.extras.setdefault('prompt_fragments', {})
        fragments['diagnosis'] = 'Consider Ayurvedic doshas.'
        return ctx


class TestBaseSkill:
    def test_no_op_hooks(self) -> None:
        skill = BaseSkill(
            SkillMetadata(name='noop', stages=(Stage.DIAGNOSIS,))
        )
        ctx = PipelineContext()
        assert skill.pre(Stage.DIAGNOSIS, ctx) is ctx
        assert skill.execute(Stage.DIAGNOSIS, ctx) is ctx
        assert skill.post(Stage.DIAGNOSIS, ctx) is ctx


class TestStageRunner:
    def test_run_single_stage(self) -> None:
        skill = _CounterSkill()
        runner = StageRunner(skills=[skill])
        ctx = PipelineContext()

        result = runner.run(Stage.DIAGNOSIS, ctx)

        assert skill.pre_count == 1
        assert skill.execute_count == 1
        assert skill.post_count == 1
        assert result.results[Stage.DIAGNOSIS] == 'counter_executed'
        assert len(result.audit) == 3

    def test_run_ignores_irrelevant_stages(self) -> None:
        skill = _CounterSkill(stages=(Stage.DIAGNOSIS,))
        runner = StageRunner(skills=[skill])
        ctx = PipelineContext()

        result = runner.run(Stage.TREATMENT, ctx)

        assert skill.pre_count == 0
        assert skill.execute_count == 0
        assert skill.post_count == 0
        assert len(result.audit) == 0

    def test_run_many(self) -> None:
        skill = _CounterSkill(stages=(Stage.DIAGNOSIS, Stage.EXAM))
        runner = StageRunner(skills=[skill])
        ctx = PipelineContext()

        result = runner.run_many([Stage.DIAGNOSIS, Stage.EXAM], ctx)

        assert skill.pre_count == 2
        assert skill.execute_count == 2
        assert skill.post_count == 2
        assert Stage.DIAGNOSIS in result.results
        assert Stage.EXAM in result.results

    def test_registration_order(self) -> None:
        first = _CounterSkill(
            name='first',
            stages=(Stage.DIAGNOSIS,),
        )
        second = _CounterSkill(
            name='second',
            stages=(Stage.DIAGNOSIS,),
        )
        runner = StageRunner(skills=[first, second])
        ctx = PipelineContext()

        result = runner.run(Stage.DIAGNOSIS, ctx)

        # second overwrites the result (runs after first)
        assert result.results[Stage.DIAGNOSIS] == 'second_executed'
        # Both ran
        assert first.execute_count == 1
        assert second.execute_count == 1
        # Audit shows registration order
        exec_audits = [a for a in result.audit if a.hook == 'execute']
        assert exec_audits[0].skill_name == 'first'
        assert exec_audits[1].skill_name == 'second'

    def test_register_with_index(self) -> None:
        """
        title: register() should insert at the given index.
        """
        runner = StageRunner()
        runner.register('hiperhealth.diagnostics')
        runner.register('hiperhealth.privacy', index=0)

        names = [s.metadata.name for s in runner.skills]
        assert names == [
            'hiperhealth.privacy',
            'hiperhealth.diagnostics',
        ]

    def test_register_skill(self) -> None:
        """
        title: register() should load and activate a built-in skill.
        """
        runner = StageRunner()
        runner.register('hiperhealth.diagnostics')

        assert len(runner.skills) == 1
        assert runner.skills[0].metadata.name == ('hiperhealth.diagnostics')

    def test_prompt_fragments(self) -> None:
        fragment_skill = _PromptFragmentSkill()
        counter_skill = _CounterSkill()
        runner = StageRunner(skills=[fragment_skill, counter_skill])
        ctx = PipelineContext()

        result = runner.run(Stage.DIAGNOSIS, ctx)

        assert (
            result.extras['prompt_fragments']['diagnosis']
            == 'Consider Ayurvedic doshas.'
        )
        assert counter_skill.execute_count == 1

    def test_run_kwargs_in_extras(self) -> None:
        skill = _CounterSkill()
        runner = StageRunner(skills=[skill])
        ctx = PipelineContext()

        result = runner.run(Stage.DIAGNOSIS, ctx, llm='mock', llm_settings='s')

        assert result.extras['_run_kwargs']['llm'] == 'mock'
        assert result.extras['_run_kwargs']['llm_settings'] == 's'

    def test_context_serialization_between_runs(self) -> None:
        skill = _CounterSkill()
        runner = StageRunner(skills=[skill])

        # First run
        ctx = PipelineContext(patient={'symptoms': 'headache'})
        ctx = runner.run(Stage.DIAGNOSIS, ctx)

        # Serialize / deserialize (simulating persistence)
        json_str = ctx.model_dump_json()
        ctx2 = PipelineContext.model_validate_json(json_str)

        # Second run with restored context
        skill2 = _CounterSkill(name='counter2', stages=(Stage.EXAM,))
        runner2 = StageRunner(skills=[skill2])
        ctx2 = runner2.run(Stage.EXAM, ctx2)

        # Both results present
        assert Stage.DIAGNOSIS in ctx2.results
        assert Stage.EXAM in ctx2.results
        # Audit from both runs
        assert len(ctx2.audit) >= 3  # from first run
