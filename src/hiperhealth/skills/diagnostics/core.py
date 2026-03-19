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


_REQ_PROMPT_TEMPLATE = {
    'en': (
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
    ),
    'pt': (
        'Você é um assistente médico experiente. '
        'Dados os dados do paciente abaixo, identifique quais informações '
        'clínicas adicionais estão faltando ou incompletas que seriam '
        'importantes para a fase de "{stage}" do atendimento. '
        'Considere elementos padrão do histórico médico: queixa principal, '
        'história da doença atual, antecedentes pessoais, medicamentos, '
        'alergias, história familiar, história social, revisão de sistemas '
        'e sinais vitais. '
        'Solicite apenas informações que NÃO estejam presentes nos dados. '
        'Para cada item, atribua prioridade: "required" (essencial para '
        'segurança), "supplementary" (melhora a precisão) ou "deferred" '
        '(pode esperar até após a avaliação inicial). '
        'Use input_type "select" com choices quando houver um conjunto '
        'finito de respostas válidas.'
    ),
    'es': (
        'Eres un asistente médico experimentado. '
        'Dados los datos del paciente a continuación, identifica qué '
        'información clínica adicional falta o está incompleta que sería '
        'importante para la fase de "{stage}" de la atención. '
        'Considera elementos estándar del historial médico: motivo de '
        'consulta, historia de la enfermedad actual, antecedentes '
        'personales, medicamentos, alergias, historia familiar, historia '
        'social, revisión por sistemas y signos vitales. '
        'Solo solicita información que NO esté presente en los datos. '
        'Para cada elemento, asigna prioridad: "required" (esencial para '
        'la seguridad), "supplementary" (mejora la precisión) o "deferred" '
        '(puede esperar hasta después de la evaluación inicial). '
        'Usa input_type "select" con choices cuando haya un conjunto '
        'finito de respuestas válidas.'
    ),
    'fr': (
        'Vous êtes un assistant médical expérimenté. '
        'À partir des données du patient ci-dessous, identifiez quelles '
        'informations cliniques supplémentaires manquent ou sont '
        'incomplètes et seraient importantes pour la phase de "{stage}" '
        'des soins. '
        'Considérez les éléments standard du dossier médical : motif de '
        'consultation, histoire de la maladie actuelle, antécédents '
        'médicaux, médicaments, allergies, histoire familiale, histoire '
        'sociale, revue des systèmes et signes vitaux. '
        'Ne demandez que les informations qui ne sont PAS déjà présentes. '
        'Pour chaque élément, attribuez une priorité : "required" '
        '(essentiel pour la sécurité), "supplementary" (améliore la '
        'précision) ou "deferred" (peut attendre après l\'évaluation '
        'initiale). '
        'Utilisez input_type "select" avec choices lorsqu\'il y a un '
        'ensemble fini de réponses valides.'
    ),
    'it': (
        'Sei un assistente medico esperto. '
        'Dati i dati del paziente qui sotto, identifica quali informazioni '
        'cliniche aggiuntive mancano o sono incomplete e sarebbero '
        'importanti per la fase di "{stage}" delle cure. '
        'Considera gli elementi standard della cartella clinica: motivo '
        'della visita, storia della malattia attuale, anamnesi patologica, '
        'farmaci, allergie, storia familiare, storia sociale, revisione '
        'dei sistemi e segni vitali. '
        'Richiedi solo informazioni che NON sono già presenti nei dati. '
        'Per ogni elemento, assegna una priorità: "required" (essenziale '
        'per la sicurezza), "supplementary" (migliora la precisione) o '
        '"deferred" (può aspettare fino a dopo la valutazione iniziale). '
        'Usa input_type "select" con choices quando c\'è un insieme '
        'finito di risposte valide.'
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

        template = _REQ_PROMPT_TEMPLATE.get(
            ctx.language, _REQ_PROMPT_TEMPLATE['en']
        )
        stage_label = stage.value if hasattr(stage, 'value') else stage
        system_prompt = template.format(stage=stage_label)

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
