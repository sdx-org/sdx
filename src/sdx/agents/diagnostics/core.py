"""Diagnostic-related LLM utilities."""

from __future__ import annotations

import json

from typing import Any, Dict, List

from sdx.agents.client import chat
from sdx.schema.clinical_outputs import LLMDiagnosis

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
    patient: Dict[str, Any], language: str = 'en'
) -> LLMDiagnosis:
    """Return summary + list of differential diagnoses."""
    prompt = _DIAG_PROMPTS.get(language, _DIAG_PROMPTS['en'])
    return chat(prompt, json.dumps(patient, ensure_ascii=False))


def exams(selected_dx: List[str], language: str = 'en') -> LLMDiagnosis:
    """Return summary + list of suggested examinations."""
    prompt = _EXAM_PROMPTS.get(language, _EXAM_PROMPTS['en'])
    return chat(prompt, json.dumps(selected_dx, ensure_ascii=False))


__all__ = ['differential', 'exams']
