"""
title: Tests for built-in skills and pipeline integration.
"""

from __future__ import annotations

import json

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hiperhealth.skills.diagnostics.core as diag_skill_mod
import pytest

from hiperhealth.pipeline import (
    PipelineContext,
    Stage,
    StageRunner,
    create_default_runner,
    discover_skills,
)
from hiperhealth.schema.clinical_outputs import LLMDiagnosis
from hiperhealth.skills.diagnostics import DiagnosticsSkill
from hiperhealth.skills.extraction import ExtractionSkill


class _FakeLLM:
    """
    title: Fake LLM that returns a fixed LLMDiagnosis.
    attributes:
      result:
        description: Value for result.
      calls:
        type: list[dict[str, object]]
        description: Value for calls.
    """

    def __init__(self, result: LLMDiagnosis) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        system: str,
        user: str,
        output_type: type[Any],
    ) -> LLMDiagnosis:
        self.calls.append({'system': system, 'user': user})
        return self.result


class TestDiagnosticsSkill:
    def test_execute_diagnosis_stage(self, monkeypatch: Any) -> None:
        """
        title: >-
          DiagnosticsSkill should call chat with patient data and store result.
        parameters:
          monkeypatch:
            type: Any
            description: Value for monkeypatch.
        """
        fake_result = LLMDiagnosis(
            summary='Possible flu', options=['Influenza', 'COVID-19']
        )
        calls: list[dict[str, Any]] = []

        def _fake_chat(
            system: str,
            user: str,
            *,
            session_id: str | None = None,
            llm: Any = None,
            llm_settings: Any = None,
        ) -> LLMDiagnosis:
            calls.append(
                {
                    'system': system,
                    'user': user,
                    'session_id': session_id,
                }
            )
            return fake_result

        monkeypatch.setattr(diag_skill_mod, 'chat', _fake_chat)

        skill = DiagnosticsSkill()
        ctx = PipelineContext(
            patient={'symptoms': 'fever, cough'},
            language='pt',
            session_id='sess-42',
        )
        runner = StageRunner(skills=[skill])
        ctx = runner.run(Stage.DIAGNOSIS, ctx)

        assert ctx.results[Stage.DIAGNOSIS] is fake_result
        assert len(calls) == 1
        assert calls[0]['session_id'] == 'sess-42'
        # Portuguese prompt should be used
        assert 'assistente médico' in calls[0]['system']
        # Patient data should be JSON-encoded
        parsed = json.loads(calls[0]['user'])
        assert parsed['symptoms'] == 'fever, cough'

    def test_execute_exam_stage_uses_diagnosis_results(
        self, monkeypatch: Any
    ) -> None:
        """
        title: Exam stage should use diagnosis options as input.
        parameters:
          monkeypatch:
            type: Any
            description: Value for monkeypatch.
        """
        exam_result = LLMDiagnosis(
            summary='Recommended exams',
            options=['Blood test', 'X-ray'],
        )
        calls: list[dict[str, Any]] = []

        def _fake_chat(
            system: str,
            user: str,
            *,
            session_id: str | None = None,
            llm: Any = None,
            llm_settings: Any = None,
        ) -> LLMDiagnosis:
            calls.append({'system': system, 'user': user})
            return exam_result

        monkeypatch.setattr(diag_skill_mod, 'chat', _fake_chat)

        skill = DiagnosticsSkill()
        ctx = PipelineContext(language='en')
        # Simulate a prior diagnosis result
        ctx.results[Stage.DIAGNOSIS] = LLMDiagnosis(
            summary='Flu suspected',
            options=['Influenza', 'Pneumonia'],
        )

        runner = StageRunner(skills=[skill])
        ctx = runner.run(Stage.EXAM, ctx)

        assert ctx.results[Stage.EXAM] is exam_result
        # Exam input should be the diagnosis options
        parsed = json.loads(calls[0]['user'])
        assert parsed == ['Influenza', 'Pneumonia']

    def test_execute_exam_without_diagnosis_is_noop(
        self, monkeypatch: Any
    ) -> None:
        """
        title: Exam stage should skip if no diagnosis results exist.
        parameters:
          monkeypatch:
            type: Any
            description: Value for monkeypatch.
        """
        calls: list[Any] = []
        monkeypatch.setattr(
            diag_skill_mod,
            'chat',
            lambda *a, **kw: calls.append(1),
        )

        skill = DiagnosticsSkill()
        ctx = PipelineContext()
        runner = StageRunner(skills=[skill])
        ctx = runner.run(Stage.EXAM, ctx)

        assert Stage.EXAM not in ctx.results
        assert len(calls) == 0

    def test_prompt_fragments_injected_in_diagnosis(
        self, monkeypatch: Any
    ) -> None:
        """
        title: Prompt fragments from other skills should be appended.
        parameters:
          monkeypatch:
            type: Any
            description: Value for monkeypatch.
        """
        calls: list[dict[str, Any]] = []

        def _fake_chat(
            system: str,
            user: str,
            **kw: Any,
        ) -> LLMDiagnosis:
            calls.append({'system': system})
            return LLMDiagnosis(summary='ok', options=['A'])

        monkeypatch.setattr(diag_skill_mod, 'chat', _fake_chat)

        skill = DiagnosticsSkill()
        ctx = PipelineContext(patient={'age': 30})
        ctx.extras['prompt_fragments'] = {
            'diagnosis': 'Consider Ayurvedic doshas.'
        }

        runner = StageRunner(skills=[skill])
        ctx = runner.run(Stage.DIAGNOSIS, ctx)

        # Prompt should include the fragment
        assert 'Consider Ayurvedic doshas.' in calls[0]['system']

    def test_prompt_fragments_injected_in_exam(self, monkeypatch: Any) -> None:
        """
        title: Prompt fragments should also work for exam stage.
        parameters:
          monkeypatch:
            type: Any
            description: Value for monkeypatch.
        """
        calls: list[dict[str, Any]] = []

        def _fake_chat(
            system: str,
            user: str,
            **kw: Any,
        ) -> LLMDiagnosis:
            calls.append({'system': system})
            return LLMDiagnosis(summary='ok', options=['X-ray'])

        monkeypatch.setattr(diag_skill_mod, 'chat', _fake_chat)

        skill = DiagnosticsSkill()
        ctx = PipelineContext()
        ctx.results[Stage.DIAGNOSIS] = LLMDiagnosis(
            summary='Flu', options=['Influenza']
        )
        ctx.extras['prompt_fragments'] = {
            'exam': 'Include traditional medicine exams.'
        }

        runner = StageRunner(skills=[skill])
        ctx = runner.run(Stage.EXAM, ctx)

        assert 'Include traditional medicine exams.' in calls[0]['system']

    def test_execute_irrelevant_stage_is_noop(self) -> None:
        """
        title: DiagnosticsSkill should not act on unrelated stages.
        """
        skill = DiagnosticsSkill()
        ctx = PipelineContext()
        runner = StageRunner(skills=[skill])
        ctx = runner.run(Stage.TREATMENT, ctx)

        assert Stage.TREATMENT not in ctx.results
        assert len(ctx.audit) == 0

    def test_exam_with_dict_options(self, monkeypatch: Any) -> None:
        """
        title: Exam should handle diagnosis options as dict (scored).
        parameters:
          monkeypatch:
            type: Any
            description: Value for monkeypatch.
        """
        calls: list[dict[str, Any]] = []

        def _fake_chat(
            system: str,
            user: str,
            **kw: Any,
        ) -> LLMDiagnosis:
            calls.append({'user': user})
            return LLMDiagnosis(summary='ok', options=['CBC'])

        monkeypatch.setattr(diag_skill_mod, 'chat', _fake_chat)

        skill = DiagnosticsSkill()
        ctx = PipelineContext()
        ctx.results[Stage.DIAGNOSIS] = LLMDiagnosis(
            summary='test',
            options={'Flu': 0.8, 'Cold': 0.6},
        )

        runner = StageRunner(skills=[skill])
        ctx = runner.run(Stage.EXAM, ctx)

        parsed = json.loads(calls[0]['user'])
        assert set(parsed) == {'Flu', 'Cold'}


class TestExtractionSkill:
    def test_extract_medical_report(self, test_data_dir: Path) -> None:
        """
        title: ExtractionSkill should extract text from PDF reports.
        parameters:
          test_data_dir:
            type: Path
            description: Value for test_data_dir.
        """
        pdf_path = test_data_dir / 'reports' / 'pdf_reports'
        pdf_files = list(pdf_path.glob('*.pdf'))
        if not pdf_files:
            pytest.skip('No PDF test files available')

        skill = ExtractionSkill()
        ctx = PipelineContext()
        ctx.extras['extraction_sources'] = {
            'medical_reports': [str(pdf_files[0])],
        }

        runner = StageRunner(skills=[skill])
        ctx = runner.run(Stage.INTAKE, ctx)

        reports = ctx.results[Stage.INTAKE]['medical_reports']
        assert len(reports) == 1
        assert 'text' in reports[0]
        assert len(reports[0]['text']) > 0
        assert reports[0]['mime_type'] == 'application/pdf'

    def test_extract_wearable_data(self, test_data_dir: Path) -> None:
        """
        title: ExtractionSkill should extract wearable data from CSV.
        parameters:
          test_data_dir:
            type: Path
            description: Value for test_data_dir.
        """
        csv_files = list((test_data_dir / 'wearable').glob('*.csv'))
        if not csv_files:
            pytest.skip('No CSV test files available')

        skill = ExtractionSkill()
        ctx = PipelineContext()
        ctx.extras['extraction_sources'] = {
            'wearable_data': [csv_files[0]],
        }

        runner = StageRunner(skills=[skill])
        ctx = runner.run(Stage.INTAKE, ctx)

        wearable = ctx.results[Stage.INTAKE]['wearable_data']
        assert len(wearable) == 1
        assert isinstance(wearable[0], list)
        assert len(wearable[0]) > 0

    def test_no_sources_produces_empty_results(self) -> None:
        """
        title: With no extraction sources, results should be empty.
        """
        skill = ExtractionSkill()
        ctx = PipelineContext()

        runner = StageRunner(skills=[skill])
        ctx = runner.run(Stage.INTAKE, ctx)

        assert ctx.results[Stage.INTAKE] == {}

    def test_ignores_non_intake_stage(self) -> None:
        """
        title: ExtractionSkill only runs on INTAKE stage.
        """
        skill = ExtractionSkill()
        ctx = PipelineContext()
        runner = StageRunner(skills=[skill])
        ctx = runner.run(Stage.DIAGNOSIS, ctx)

        assert Stage.DIAGNOSIS not in ctx.results


class TestPrivacySkill:
    def test_deidentifies_patient_data(self, monkeypatch: Any) -> None:
        """
        title: PrivacySkill should replace PII in patient data.
        parameters:
          monkeypatch:
            type: Any
            description: Value for monkeypatch.
        """

        class _StubDeidentifier:
            def __init__(self) -> None:
                self.analyzer = SimpleNamespace(
                    registry=SimpleNamespace(
                        recognizers=[],
                        get_recognizers=lambda **kw: [],
                        add_recognizer=lambda r: None,
                    )
                )
                self.anonymizer = SimpleNamespace()

            def deidentify(self, text: str) -> str:
                return '<redacted>'

        from hiperhealth.skills.privacy import PrivacySkill

        skill = PrivacySkill()
        # Replace the internal deidentifier with our stub
        skill._deidentifier = _StubDeidentifier()  # type: ignore[assignment]

        ctx = PipelineContext(
            patient={
                'symptoms': 'Patient John has headache',
                'age': 35,
                'name': 'John Doe',
            }
        )

        runner = StageRunner(skills=[skill])
        ctx = runner.run(Stage.SCREENING, ctx)

        # 'symptoms' is in _DEFAULT_KEYS_TO_DEIDENTIFY
        assert ctx.patient['symptoms'] == '<redacted>'
        # 'name' is NOT in the default keys
        assert ctx.patient['name'] == 'John Doe'
        # Non-string values unchanged
        assert ctx.patient['age'] == 35

    def test_empty_patient_is_noop(self) -> None:
        """
        title: PrivacySkill should not fail on empty patient data.
        """
        from hiperhealth.skills.privacy import PrivacySkill

        skill = PrivacySkill()
        ctx = PipelineContext(patient={})

        # execute should return ctx unchanged when patient is empty
        result = skill.execute(Stage.SCREENING, ctx)
        assert result.patient == {}


class TestCreateDefaultRunner:
    def test_creates_runner_with_built_in_skills(self) -> None:
        """
        title: >-
          create_default_runner should return a StageRunner with all 3 built-in
          skills.
        """
        runner = create_default_runner()

        skill_names = [s.metadata.name for s in runner.skills]
        assert 'hiperhealth.privacy' in skill_names
        assert 'hiperhealth.extraction' in skill_names
        assert 'hiperhealth.diagnostics' in skill_names

    def test_privacy_runs_first(self) -> None:
        """
        title: Privacy skill should be registered first in the default runner.
        """
        runner = create_default_runner()

        skill_names = [s.metadata.name for s in runner.skills]
        assert skill_names[0] == 'hiperhealth.privacy'


class TestDiscoverSkills:
    def test_discover_returns_list(self) -> None:
        """
        title: discover_skills should return a list.
        """
        skills = discover_skills()
        assert isinstance(skills, list)

    def test_discover_with_nonexistent_group(self) -> None:
        """
        title: Non-existent entry point group should return empty list.
        """
        skills = discover_skills(group='nonexistent.skills.group')
        assert skills == []


class TestBackwardCompatImports:
    """
    title: Verify that old import paths still work.
    """

    def test_agents_diagnostics_exports(self) -> None:
        from hiperhealth.agents.diagnostics.core import (
            differential,
            exams,
        )

        assert callable(differential)
        assert callable(exams)

    def test_agents_extraction_exports(self) -> None:
        from hiperhealth.agents.extraction.medical_reports import (
            MedicalReportFileExtractor,
        )
        from hiperhealth.agents.extraction.wearable import (
            WearableDataFileExtractor,
        )

        assert MedicalReportFileExtractor is not None
        assert WearableDataFileExtractor is not None

    def test_privacy_exports(self) -> None:
        from hiperhealth.privacy import (
            Deidentifier,
            deidentify_patient_record,
        )

        assert Deidentifier is not None
        assert callable(deidentify_patient_record)

    def test_privacy_deidentifier_module_exports(self) -> None:
        from hiperhealth.privacy.deidentifier import (
            Deidentifier,
            PrivacySkill,
            deidentify_patient_record,
        )

        assert Deidentifier is not None
        assert PrivacySkill is not None
        assert callable(deidentify_patient_record)
