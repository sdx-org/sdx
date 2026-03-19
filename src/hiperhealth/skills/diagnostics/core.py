"""
title: Diagnostic-related LLM utilities and DiagnosticsSkill.
"""

from __future__ import annotations

import json

from typing import Any

from hiperhealth.agents.client import chat, chat_structured
from hiperhealth.llm import LLMSettings, StructuredLLM
from hiperhealth.pipeline.context import PipelineContext
from hiperhealth.pipeline.session import Inquiry
from hiperhealth.pipeline.skill import BaseSkill, SkillMetadata
from hiperhealth.pipeline.stages import Stage
from hiperhealth.schema.clinical_outputs import LLMDiagnosis, LLMInquiryList

_SUPPORTED_OUTPUT_LANGUAGES = {
    'en': 'English',
    'pt': 'Portuguese',
    'es': 'Spanish',
    'fr': 'French',
    'it': 'Italian',
}

_DIAG_PROMPT = (
    'You are an experienced physician assistant. '
    "Return a JSON object with keys 'summary' (two sentences) and "
    "'options' (array of differential diagnoses) given the patient data."
)

_EXAM_PROMPT = (
    'You are an experienced physician assistant. '
    "Given the selected diagnoses, return JSON with keys 'summary' and "
    "'options' (max 10 exam/procedure names)."
)

_REQ_PROMPT_TEMPLATE = (
    'You are an experienced physician assistant. '
    'Given the patient data below, identify what additional clinical '
    'information is missing or incomplete that would be important for '
    'the "{stage}" phase of care. '
    'Consider standard medical history elements: chief complaint, '
    'history of present illness, past medical history, medications, '
    'allergies, family history, social history, review of systems, '
    'and vital signs. '
    'Only request information that is NOT already present in the data. '
    'For each item, assign priority: "required" (essential for safety), '
    '"supplementary" (improves accuracy), or "deferred" (can wait '
    'until after initial assessment). '
    'Use input_type "select" with choices when there is a finite set '
    'of valid answers.'
)


def _language_name(language: str) -> str:
    return _SUPPORTED_OUTPUT_LANGUAGES.get(language, 'English')


def _natural_language_instruction(language: str) -> str:
    return (
        'Write all natural-language string values in '
        f'{_language_name(language)}. '
        'Keep JSON keys exactly as requested.'
    )


def _requirements_language_instruction(language: str) -> str:
    return (
        'Write `label`, `description`, and any `choices` values in '
        f'{_language_name(language)}. '
        'Keep `field` as stable English snake_case. '
        "Keep `priority` exactly one of 'required', 'supplementary', or "
        "'deferred'. "
        'Keep `input_type` in English. '
        'Keep JSON keys exactly as requested.'
    )


def _diagnosis_prompt(language: str) -> str:
    return f'{_DIAG_PROMPT}\n\n{_natural_language_instruction(language)}'


def _exam_prompt(language: str) -> str:
    return f'{_EXAM_PROMPT}\n\n{_natural_language_instruction(language)}'


def _requirements_prompt(stage: str, language: str) -> str:
    stage_label = stage.value if hasattr(stage, 'value') else stage
    base_prompt = _REQ_PROMPT_TEMPLATE.format(stage=stage_label)
    return f'{base_prompt}\n\n{_requirements_language_instruction(language)}'


def differential(
    patient: dict[str, Any],
    language: str = 'en',
    session_id: str | None = None,
    llm: StructuredLLM | None = None,
    llm_settings: LLMSettings | None = None,
) -> LLMDiagnosis:
    """
    title: Return summary + list of differential diagnoses.
    parameters:
      patient:
        type: dict[str, Any]
        description: Value for patient.
      language:
        type: str
        description: Value for language.
      session_id:
        type: str | None
        description: Value for session_id.
      llm:
        type: StructuredLLM | None
        description: Value for llm.
      llm_settings:
        type: LLMSettings | None
        description: Value for llm_settings.
    returns:
      type: LLMDiagnosis
      description: Return value.
    """
    prompt = _diagnosis_prompt(language)
    chat_kwargs: dict[str, Any] = {'session_id': session_id}
    if llm is not None:
        chat_kwargs['llm'] = llm
    if llm_settings is not None:
        chat_kwargs['llm_settings'] = llm_settings
    return chat(
        prompt,
        json.dumps(patient, ensure_ascii=False),
        **chat_kwargs,
    )


def exams(
    selected_dx: list[str],
    language: str = 'en',
    session_id: str | None = None,
    llm: StructuredLLM | None = None,
    llm_settings: LLMSettings | None = None,
) -> LLMDiagnosis:
    """
    title: Return summary + list of suggested examinations.
    parameters:
      selected_dx:
        type: list[str]
        description: Value for selected_dx.
      language:
        type: str
        description: Value for language.
      session_id:
        type: str | None
        description: Value for session_id.
      llm:
        type: StructuredLLM | None
        description: Value for llm.
      llm_settings:
        type: LLMSettings | None
        description: Value for llm_settings.
    returns:
      type: LLMDiagnosis
      description: Return value.
    """
    prompt = _exam_prompt(language)
    chat_kwargs: dict[str, Any] = {'session_id': session_id}
    if llm is not None:
        chat_kwargs['llm'] = llm
    if llm_settings is not None:
        chat_kwargs['llm_settings'] = llm_settings
    return chat(
        prompt,
        json.dumps(selected_dx, ensure_ascii=False),
        **chat_kwargs,
    )


class DiagnosticsSkill(BaseSkill):
    """
    title: Core differential diagnosis and exam suggestion skill.
    """

    def __init__(self) -> None:
        super().__init__(
            SkillMetadata(
                name='hiperhealth.diagnostics',
                version='0.4.0',
                stages=(Stage.DIAGNOSIS, Stage.EXAM),
                description=(
                    'Core differential diagnosis and exam suggestion.'
                ),
            )
        )

    def check_requirements(
        self, stage: str, ctx: PipelineContext
    ) -> list[Inquiry]:
        """
        title: Use the LLM to identify missing clinical information.
        summary: |-
          Sends the current patient data to the LLM and asks what
          additional information would improve the given stage.
          Fields already present in ctx.patient are filtered out.
        parameters:
          stage:
            type: str
          ctx:
            type: PipelineContext
        returns:
          type: list[Inquiry]
        """
        run_kwargs = ctx.extras.get('_run_kwargs', {})
        llm = run_kwargs.get('llm')
        llm_settings = run_kwargs.get('llm_settings')

        system_prompt = _requirements_prompt(stage, ctx.language)

        extra = ctx.extras.get('prompt_fragments', {}).get(
            f'{stage}_requirements', ''
        )
        if extra:
            system_prompt = f'{system_prompt}\n\n{extra}'

        result = chat_structured(
            system_prompt,
            json.dumps(ctx.patient, ensure_ascii=False),
            LLMInquiryList,
            session_id=ctx.session_id,
            llm=llm,
            llm_settings=llm_settings,
        )

        existing_fields = set(ctx.patient.keys())
        return [
            Inquiry(
                skill_name=self.metadata.name,
                stage=stage,
                field=item.field,
                label=item.label,
                description=item.description,
                priority=item.priority,
                input_type=item.input_type,
                choices=item.choices,
            )
            for item in result.inquiries
            if item.field not in existing_fields
        ]

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        """
        title: Run differential diagnosis or exam suggestions.
        parameters:
          stage:
            type: str
          ctx:
            type: PipelineContext
        returns:
          type: PipelineContext
        """
        run_kwargs = ctx.extras.get('_run_kwargs', {})
        llm = run_kwargs.get('llm')
        llm_settings = run_kwargs.get('llm_settings')

        if stage == Stage.DIAGNOSIS:
            prompt = _diagnosis_prompt(ctx.language)
            extra = ctx.extras.get('prompt_fragments', {}).get('diagnosis', '')
            if extra:
                prompt = f'{prompt}\n\n{extra}'

            result = chat(
                prompt,
                json.dumps(ctx.patient, ensure_ascii=False),
                session_id=ctx.session_id,
                llm=llm,
                llm_settings=llm_settings,
            )
            ctx.results[Stage.DIAGNOSIS] = result

        elif stage == Stage.EXAM:
            diagnosis = ctx.results.get(Stage.DIAGNOSIS)
            if not diagnosis:
                return ctx
            options = (
                diagnosis.options
                if hasattr(diagnosis, 'options')
                else diagnosis.get('options', [])
            )
            selected = (
                options
                if isinstance(options, list)
                else list(options.keys())
                if isinstance(options, dict)
                else []
            )

            prompt = _exam_prompt(ctx.language)
            extra = ctx.extras.get('prompt_fragments', {}).get('exam', '')
            if extra:
                prompt = f'{prompt}\n\n{extra}'

            result = chat(
                prompt,
                json.dumps(selected, ensure_ascii=False),
                session_id=ctx.session_id,
                llm=llm,
                llm_settings=llm_settings,
            )
            ctx.results[Stage.EXAM] = result

        return ctx


__all__ = ['DiagnosticsSkill', 'differential', 'exams']
