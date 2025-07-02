"""CLI for physician-guided consultations."""

from __future__ import annotations

import json
import os

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, cast

import questionary
import typer

from dotenv import load_dotenv
from openai import OpenAI
from rich import print

# ────────────────────────── environment ──────────────────────────

load_dotenv(Path(__file__).parents[3] / '.envs' / '.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
MODEL_NAME = 'o4-mini-2025-04-16'

# ───────────────────────────── paths ─────────────────────────────

RECORDS_DIR = Path.home() / 'config' / '.sdx' / 'records'
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── utilities ───────────────────────────


def call_openai(system_msg: str, user_msg: str) -> dict[str, Any]:
    """Query the chat model.

    Parameters
    ----------
    system_msg
        Instruction in the system role.
    user_msg
        Prompt content in the user role.

    Returns
    -------
    dict
        Parsed JSON returned by the model.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    rsp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': user_msg},
        ],
    )
    result = json.loads(rsp.choices[0].message.content or '{}')
    return cast(dict[str, Any], result)


def save_record(payload: dict[str, Any]) -> Path:
    """Persist the consultation data."""
    path = RECORDS_DIR / f'{payload["meta"]["timestamp"]}.json'
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def multiselect(title: str, items: List[str]) -> List[str]:
    """Checkbox UI."""
    return questionary.checkbox(title, choices=items).ask() or []


app = typer.Typer(add_completion=False)

# ───────────────────────────── CLI ──────────────────────────────


@app.command('consult')
def consult() -> None:
    """Interactive consultation workflow."""
    meta = {'timestamp': datetime.utcnow().isoformat(timespec='seconds')}
    patient: Dict[str, Any] = {}

    print('[bold cyan]\nPatient demographics[/bold cyan]')
    patient['age'] = typer.prompt('Age (years)', type=int)
    patient['gender'] = typer.prompt('Gender (M/F/Other)')
    patient['weight_kg'] = typer.prompt('Weight (kg)', type=float)
    patient['height_cm'] = typer.prompt('Height (cm)', type=float)

    print('[bold cyan]\nLifestyle details[/bold cyan]')
    patient['diet'] = typer.prompt('Diet (e.g., balanced, keto)')
    patient['sleep_hours'] = typer.prompt('Sleep per night (h)', type=float)
    patient['physical_activity'] = typer.prompt('Physical exercise')
    patient['mental_exercises'] = typer.prompt('Mental activities')

    print('[bold cyan]\nCurrent symptoms[/bold cyan]')
    patient['symptoms'] = typer.prompt('Main symptoms (comma-separated)')

    print('[bold cyan]\nMental health[/bold cyan]')
    patient['mental_health'] = typer.prompt('Mental health concerns')

    print('[bold cyan]\nPrevious exams/tests[/bold cyan]')
    patient['previous_tests'] = typer.prompt("Summary or 'none'")

    # ── differential diagnosis ──
    sys_diag = (
        'You are an experienced physician assistant. '
        "Return a JSON object with keys 'summary' (two sentences) and "
        "'options' (array of differential diagnoses) given the patient data."
    )
    diag_json = call_openai(sys_diag, json.dumps(patient, ensure_ascii=False))
    print(f'\n[bold magenta]AI summary:[/bold magenta] {diag_json["summary"]}')
    chosen_diag = multiselect(
        'Select diagnoses to investigate', diag_json['options']
    )

    # ── exam suggestions ──
    sys_exam = (
        'You are an experienced physician assistant. '
        "Given the selected diagnoses, return JSON with keys 'summary' "
        "and 'options' (max 10 exam/procedure names)."
    )
    exam_json = call_openai(
        sys_exam, json.dumps(chosen_diag, ensure_ascii=False)
    )
    print(f'\n[bold magenta]AI summary:[/bold magenta] {exam_json["summary"]}')
    chosen_exams = multiselect('Select exams to request', exam_json['options'])

    record = {
        'meta': meta,
        'patient': patient,
        'ai': {
            'diagnosis_options': diag_json['options'],
            'selected_diagnoses': chosen_diag,
            'exam_options': exam_json['options'],
            'selected_exams': chosen_exams,
        },
    }
    path = save_record(record)
    print(f'\n[green]Record saved to {path}[/green]')


if __name__ == '__main__':
    app()
