"""
title: Tests for the parquet-backed session and assess flow.
"""

from __future__ import annotations

from pathlib import Path

from hiperhealth.pipeline import (
    BaseSkill,
    Inquiry,
    PipelineContext,
    Session,
    SkillMetadata,
    Stage,
    StageRunner,
)

# ── Test skill that raises inquiries ───────────────────────────────


class _AssessingSkill(BaseSkill):
    """
    title: A test skill that requests dietary_history and stool_analysis.
    """

    def __init__(self) -> None:
        super().__init__(
            SkillMetadata(
                name='test.assessor',
                stages=(Stage.DIAGNOSIS,),
            )
        )

    def check_requirements(
        self, stage: str, ctx: PipelineContext
    ) -> list[Inquiry]:
        inquiries: list[Inquiry] = []
        if 'dietary_history' not in ctx.patient:
            inquiries.append(
                Inquiry(
                    skill_name=self.metadata.name,
                    stage=stage,
                    field='dietary_history',
                    label='Describe your typical daily diet',
                    description='Dietary patterns affect analysis',
                    priority='required',
                    input_type='text',
                )
            )
        if 'stool_analysis' not in ctx.patient:
            inquiries.append(
                Inquiry(
                    skill_name=self.metadata.name,
                    stage=stage,
                    field='stool_analysis',
                    label='Stool analysis results',
                    priority='deferred',
                    input_type='file',
                )
            )
        return inquiries

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        ctx.results[stage] = {'assessed': True}
        return ctx


class _NoAssessSkill(BaseSkill):
    """
    title: A skill that never raises inquiries (default check_requirements).
    """

    def __init__(self) -> None:
        super().__init__(
            SkillMetadata(
                name='test.no_assess',
                stages=(Stage.DIAGNOSIS,),
            )
        )

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        ctx.results[stage] = {'plain': True}
        return ctx


# ── Inquiry model tests ───────────────────────────────────────────


class TestInquiry:
    def test_defaults(self) -> None:
        inq = Inquiry(
            skill_name='test',
            stage='diagnosis',
            field='lab_results',
            label='Lab results',
        )
        assert inq.priority == 'supplementary'
        assert inq.input_type == 'text'
        assert inq.description == ''
        assert inq.choices is None

    def test_full_fields(self) -> None:
        inq = Inquiry(
            skill_name='test',
            stage='diagnosis',
            field='pain_level',
            label='Rate your pain',
            description='1-10 scale',
            priority='required',
            input_type='choice',
            choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        )
        assert inq.priority == 'required'
        assert len(inq.choices) == 10

    def test_serialization_roundtrip(self) -> None:
        inq = Inquiry(
            skill_name='test',
            stage='diagnosis',
            field='lab_results',
            label='Lab results',
            priority='deferred',
        )
        json_str = inq.model_dump_json()
        restored = Inquiry.model_validate_json(json_str)
        assert restored == inq


# ── Session tests ─────────────────────────────────────────────────


class TestSession:
    def test_create_and_load(self, tmp_path: Path) -> None:
        path = tmp_path / 'session.parquet'
        Session.create(path)
        assert path.exists()

        loaded = Session.load(path)
        assert loaded.clinical_data == {}
        assert loaded.results == {}
        assert loaded.stages_completed == []

    def test_create_already_exists(self, tmp_path: Path) -> None:
        path = tmp_path / 'session.parquet'
        Session.create(path)
        try:
            Session.create(path)
            raise AssertionError('Expected FileExistsError')
        except FileExistsError:
            pass

    def test_load_not_found(self, tmp_path: Path) -> None:
        path = tmp_path / 'missing.parquet'
        try:
            Session.load(path)
            raise AssertionError('Expected FileNotFoundError')
        except FileNotFoundError:
            pass

    def test_set_clinical_data(self, tmp_path: Path) -> None:
        path = tmp_path / 'session.parquet'
        session = Session.create(path)
        session.set_clinical_data(
            {
                'symptoms': 'bloating',
                'age': 34,
            }
        )
        assert session.clinical_data == {
            'symptoms': 'bloating',
            'age': 34,
        }

    def test_clinical_data_persists(self, tmp_path: Path) -> None:
        path = tmp_path / 'session.parquet'
        session = Session.create(path)
        session.set_clinical_data({'symptoms': 'bloating'})

        loaded = Session.load(path)
        assert loaded.clinical_data == {'symptoms': 'bloating'}

    def test_provide_answers_merges(self, tmp_path: Path) -> None:
        path = tmp_path / 'session.parquet'
        session = Session.create(path)
        session.set_clinical_data({'symptoms': 'bloating'})
        session.provide_answers({'dietary_history': 'high carb'})

        assert session.clinical_data == {
            'symptoms': 'bloating',
            'dietary_history': 'high carb',
        }

    def test_to_context(self, tmp_path: Path) -> None:
        path = tmp_path / 'session.parquet'
        session = Session.create(path, language='pt')
        session.set_clinical_data({'symptoms': 'inchaço'})

        ctx = session.to_context()
        assert ctx.patient == {'symptoms': 'inchaço'}
        assert ctx.language == 'pt'
        assert ctx.session_id == 'session'

    def test_events_are_recorded(self, tmp_path: Path) -> None:
        path = tmp_path / 'session.parquet'
        session = Session.create(path)
        session.set_clinical_data({'symptoms': 'fatigue'})
        session.provide_answers({'diet': 'low fiber'})

        events = session.events
        assert len(events) == 2
        assert events[0]['event_type'] == 'clinical_data_set'
        assert events[1]['event_type'] == 'answers_provided'

    def test_stages_completed(self, tmp_path: Path) -> None:
        path = tmp_path / 'session.parquet'
        session = Session.create(path)
        session.set_clinical_data({'symptoms': 'fatigue'})

        skill = _NoAssessSkill()
        runner = StageRunner(skills=[skill])
        runner.run_session(Stage.DIAGNOSIS, session)

        assert Stage.DIAGNOSIS in session.stages_completed


# ── Check requirements flow tests ─────────────────────────────────


class TestCheckRequirements:
    def test_assess_returns_inquiries(self, tmp_path: Path) -> None:
        path = tmp_path / 'session.parquet'
        session = Session.create(path)
        session.set_clinical_data({'symptoms': 'bloating'})

        skill = _AssessingSkill()
        runner = StageRunner(skills=[skill])

        inquiries = runner.check_requirements(Stage.DIAGNOSIS, session)
        assert len(inquiries) == 2
        fields = {i.field for i in inquiries}
        assert fields == {'dietary_history', 'stool_analysis'}

    def test_assess_no_inquiries_when_data_present(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / 'session.parquet'
        session = Session.create(path)
        session.set_clinical_data(
            {
                'symptoms': 'bloating',
                'dietary_history': 'high carb',
                'stool_analysis': 'normal',
            }
        )

        skill = _AssessingSkill()
        runner = StageRunner(skills=[skill])

        inquiries = runner.check_requirements(Stage.DIAGNOSIS, session)
        assert len(inquiries) == 0

    def test_assess_records_events(self, tmp_path: Path) -> None:
        path = tmp_path / 'session.parquet'
        session = Session.create(path)
        session.set_clinical_data({'symptoms': 'bloating'})

        skill = _AssessingSkill()
        runner = StageRunner(skills=[skill])
        runner.check_requirements(Stage.DIAGNOSIS, session)

        event_types = [e['event_type'] for e in session.events]
        assert 'check_requirements_started' in event_types
        assert 'inquiries_raised' in event_types
        assert 'check_requirements_completed' in event_types

    def test_assess_no_inquiries_skill(self, tmp_path: Path) -> None:
        path = tmp_path / 'session.parquet'
        session = Session.create(path)
        session.set_clinical_data({'symptoms': 'bloating'})

        skill = _NoAssessSkill()
        runner = StageRunner(skills=[skill])

        inquiries = runner.check_requirements(Stage.DIAGNOSIS, session)
        assert len(inquiries) == 0

    def test_pending_inquiries_property(self, tmp_path: Path) -> None:
        path = tmp_path / 'session.parquet'
        session = Session.create(path)
        session.set_clinical_data({'symptoms': 'bloating'})

        skill = _AssessingSkill()
        runner = StageRunner(skills=[skill])
        runner.check_requirements(Stage.DIAGNOSIS, session)

        # Both pending
        assert len(session.pending_inquiries) == 2

        # Provide one answer
        session.provide_answers({'dietary_history': 'high carb'})
        assert len(session.pending_inquiries) == 1
        assert session.pending_inquiries[0].field == 'stool_analysis'

    def test_full_assess_provide_run_cycle(self, tmp_path: Path) -> None:
        path = tmp_path / 'session.parquet'
        session = Session.create(path)
        session.set_clinical_data({'symptoms': 'bloating'})

        skill = _AssessingSkill()
        runner = StageRunner(skills=[skill])

        # Step 1: Assess
        inquiries = runner.check_requirements(Stage.DIAGNOSIS, session)
        required = [i for i in inquiries if i.priority == 'required']
        assert len(required) == 1
        assert required[0].field == 'dietary_history'

        # Step 2: Provide answers
        session.provide_answers({'dietary_history': 'high carb'})

        # Step 3: Re-assess
        inquiries = runner.check_requirements(Stage.DIAGNOSIS, session)
        required = [i for i in inquiries if i.priority == 'required']
        assert len(required) == 0

        # Step 4: Run
        runner.run_session(Stage.DIAGNOSIS, session)
        assert Stage.DIAGNOSIS in session.stages_completed
        assert session.results[Stage.DIAGNOSIS] == {'assessed': True}

    def test_session_survives_reload(self, tmp_path: Path) -> None:
        path = tmp_path / 'session.parquet'

        # Day 1
        session = Session.create(path)
        session.set_clinical_data({'symptoms': 'bloating'})
        skill = _AssessingSkill()
        runner = StageRunner(skills=[skill])
        runner.check_requirements(Stage.DIAGNOSIS, session)

        # Day 2 — reload from disk
        session2 = Session.load(path)
        session2.provide_answers({'dietary_history': 'high carb'})

        # Re-assess with reloaded session
        inquiries = runner.check_requirements(Stage.DIAGNOSIS, session2)
        required = [i for i in inquiries if i.priority == 'required']
        assert len(required) == 0

        # Run
        runner.run_session(Stage.DIAGNOSIS, session2)
        assert Stage.DIAGNOSIS in session2.stages_completed

    def test_multiple_skills_assess(self, tmp_path: Path) -> None:
        path = tmp_path / 'session.parquet'
        session = Session.create(path)
        session.set_clinical_data({'symptoms': 'bloating'})

        skill1 = _AssessingSkill()
        skill2 = _NoAssessSkill()
        runner = StageRunner(skills=[skill1, skill2])

        inquiries = runner.check_requirements(Stage.DIAGNOSIS, session)
        # Only skill1 raises inquiries
        assert len(inquiries) == 2
        assert all(i.skill_name == 'test.assessor' for i in inquiries)

    def test_assess_irrelevant_stage(self, tmp_path: Path) -> None:
        path = tmp_path / 'session.parquet'
        session = Session.create(path)
        session.set_clinical_data({'symptoms': 'bloating'})

        skill = _AssessingSkill()  # only registered for DIAGNOSIS
        runner = StageRunner(skills=[skill])

        inquiries = runner.check_requirements(Stage.TREATMENT, session)
        assert len(inquiries) == 0
