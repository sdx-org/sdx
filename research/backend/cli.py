"""CLI for physician-guided consultations."""

from __future__ import annotations

import json

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import questionary
import typer

from hiperhealth.agents.diagnostics import core as diag
from rich import print

RECORDS_DIR = Path.home() / 'config' / '.hiperhealth' / 'records'
RECORDS_DIR.mkdir(parents=True, exist_ok=True)


def save_record(payload: dict[str, Any]) -> Path:
    """Save the record as JSON."""
    path = RECORDS_DIR / f'{payload["meta"]["timestamp"]}.json'
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def multiselect(title: str, items: list[str]) -> list[str]:
    """Provide checkbox field."""
    return questionary.checkbox(title, choices=items).ask() or []


app = typer.Typer(add_completion=False)


@app.command('consult')
def consult() -> None:
    """Interactive consultation workflow."""
    meta = {'timestamp': datetime.now(UTC).isoformat(timespec='seconds')}
    patient: dict[str, Any] = {}

    # ── inputs ──────────────────────────────────────────────────────────
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

    # ── LLM calls via agents ────────────────────────────────────────────
    diag_json = diag.differential(patient)
    print(f'\n[bold magenta]AI summary:[/bold magenta] {diag_json["summary"]}')
    chosen_diag = multiselect(
        'Select diagnoses to investigate', diag_json['options']
    )

    exam_json = diag.exams(chosen_diag)
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


if __name__ == '__main__':  # pragma: no cover
    app()
