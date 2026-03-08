"""Tests for schema exports and model validation."""

from __future__ import annotations

from datetime import datetime, timezone

import hiperhealth.schema as schema_pkg
import hiperhealth.schema.fhirx as fhir_mod
import pytest

from hiperhealth.schema.clinical_outputs import LLMDiagnosis
from hiperhealth.schema.human_evaluations import (
    AIOutput,
    DeIdentifiedDatasetDescriptor,
    Evaluation,
)
from pydantic import ValidationError


def test_schema_package_exports_llmdiagnosis():
    """Top-level schema package should re-export LLMDiagnosis."""
    assert schema_pkg.__all__ == ['LLMDiagnosis']
    assert schema_pkg.LLMDiagnosis is LLMDiagnosis


def test_llmdiagnosis_from_llm_parses_fenced_json():
    """from_llm should accept markdown fenced JSON responses."""
    raw = """
    ```json
    {"summary":"Short summary","options":["A","B"]}
    ```
    """
    parsed = LLMDiagnosis.from_llm(raw)
    assert parsed.summary == 'Short summary'
    assert parsed.options == ['A', 'B']


def test_llmdiagnosis_from_llm_invalid_payload_raises():
    """Missing required keys should raise pydantic ValidationError."""
    with pytest.raises(ValidationError):
        LLMDiagnosis.from_llm('{"summary":"only summary"}')


def test_fhirx_models_expose_language_field():
    """FHIR wrapper models should expose the shared language field."""
    classes = [
        fhir_mod.Patient,
        fhir_mod.Encounter,
        fhir_mod.Observation,
        fhir_mod.Condition,
        fhir_mod.Procedure,
        fhir_mod.ClinicalImpression,
        fhir_mod.Annotation,
    ]
    for cls in classes:
        assert 'language' in cls.model_fields
        assert issubclass(cls, fhir_mod.BaseLanguage)


def test_base_language_requires_language_value():
    """Base language schema should require explicit language value."""
    with pytest.raises(ValidationError):
        fhir_mod.BaseLanguage()


def test_human_evaluation_models_validate_expected_shapes():
    """Domain schemas should validate and keep typed data."""
    now = datetime.now(timezone.utc)

    ai_out = AIOutput(
        language='en',
        id='ai-1',
        encounter_id='enc-1',
        type='diagnosis',
        content='Differential suggestions',
        model_version='v1',
        timestamp=now,
    )
    assert ai_out.language == 'en'
    assert ai_out.type == 'diagnosis'

    evaluation = Evaluation(
        language='en',
        id='ev-1',
        aioutput_id='ai-1',
        output_type='diagnosis',
        ratings={
            'accuracy': 4,
            'relevance': 5,
            'usefulness': 4,
            'coherence': 5,
        },
        safety='safe',
        comments='Looks reasonable.',
        timestamp=now,
    )
    assert evaluation.ratings['accuracy'] == 4
    assert evaluation.safety == 'safe'

    descriptor = DeIdentifiedDatasetDescriptor(
        language='en',
        dataset_id='ds-1',
        generation_date=now,
        version='1.0.0',
        records=123,
        license='CC-BY-4.0',
        url='https://example.org/dataset',
    )
    assert descriptor.records == 123
