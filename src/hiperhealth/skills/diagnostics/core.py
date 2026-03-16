"""
title: Diagnostic-related LLM utilities and DiagnosticsSkill.
"""

from __future__ import annotations

import json

from typing import Any

from hiperhealth.agents.client import chat
from hiperhealth.llm import LLMSettings, StructuredLLM
from hiperhealth.pipeline.context import PipelineContext
from hiperhealth.pipeline.skill import BaseSkill, SkillMetadata
from hiperhealth.pipeline.stages import Stage
from hiperhealth.schema.clinical_outputs import LLMDiagnosis

_DIAG_PROMPTS = {
    'en': (
        'You are an experienced physician assistant. '
        "Return a JSON object with keys 'summary' (two sentences) and "
        "'options' (array of differential diagnoses) given the patient data."
    ),
    'pt': (
        'Você é um assistente médico experiente. '
        "Retorne um objeto JSON com as chaves 'summary' (duas frases) e "
        "'options' (lista de diagnósticos diferenciais) com base nos dados do "
        'paciente.'
    ),
    'es': (
        'Eres un asistente médico experimentado. '
        "Devuelve un objeto JSON con las claves 'summary' (dos frases) y "
        "'options' (lista de diagnósticos diferenciales) a partir de los "
        'datos del paciente.'
    ),
    'fr': (
        'Vous êtes un assistant médical expérimenté. '
        "Retournez un objet JSON avec les clés 'summary' (deux phrases) et "
        "'options' (liste des diagnostics différentiels) à partir des données "
        'du patient.'
    ),
    'it': (
        'Sei un assistente medico esperto. '
        "Restituisci un oggetto JSON con le chiavi 'summary' (due frasi) e "
        "'options' (elenco delle diagnosi differenziali) in base ai dati del "
        'paziente.'
    ),
}

_EXAM_PROMPTS = {
    'en': (
        'You are an experienced physician assistant. '
        "Given the selected diagnoses, return JSON with keys 'summary' and "
        "'options' (max 10 exam/procedure names)."
    ),
    'pt': (
        'Você é um assistente médico experiente. '
        'Com base nos diagnósticos selecionados, retorne um JSON com as '
        "chaves 'summary' e 'options' (no máximo 10 nomes de "
        'exames/procedimentos).'
    ),
    'es': (
        'Eres un asistente médico experimentado. '
        'Dado los diagnósticos seleccionados, devuelve un JSON con las claves '
        "'summary' y 'options' (máx. 10 nombres de "
        'exámenes/procedimientos).'
    ),
    'fr': (
        'Vous êtes un assistant médical expérimenté. '
        'À partir des diagnostics sélectionnés, retournez un JSON avec les '
        "clés 'summary' et 'options' (maximum 10 noms d'examens/"
        'procédures).'
    ),
    'it': (
        'Sei un assistente medico esperto. '
        'Dati i diagnosi selezionati, restituisci un JSON con le chiavi '
        "'summary' e 'options' (massimo 10 nomi di esami/procedure)."
    ),
}


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
    prompt = _DIAG_PROMPTS.get(language, _DIAG_PROMPTS['en'])
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
    prompt = _EXAM_PROMPTS.get(language, _EXAM_PROMPTS['en'])
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
            prompt = _DIAG_PROMPTS.get(ctx.language, _DIAG_PROMPTS['en'])
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
            selected = (
                diagnosis.options
                if isinstance(diagnosis.options, list)
                else list(diagnosis.options.keys())
            )

            prompt = _EXAM_PROMPTS.get(ctx.language, _EXAM_PROMPTS['en'])
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
