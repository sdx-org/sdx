"""
title: Typed state objects for the optional notebook interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hiperhealth.pipeline import Stage

StageStatus = Literal[
    'not_started',
    'needs_information',
    'waiting_deferred',
    'ready',
    'complete',
    'failed',
]

STANDARD_STAGES: tuple[str, ...] = (
    Stage.SCREENING.value,
    Stage.INTAKE.value,
    Stage.DIAGNOSIS.value,
    Stage.EXAM.value,
    Stage.TREATMENT.value,
    Stage.PRESCRIPTION.value,
)


@dataclass(frozen=True)
class WorkflowStep:
    """
    title: A guided notebook workflow step.
    attributes:
      identifier:
        type: str
      title:
        type: str
      description:
        type: str
      stage:
        type: str | None
    """

    identifier: str
    title: str
    description: str
    stage: str | None = None


WORKFLOW_STEPS: tuple[WorkflowStep, ...] = (
    WorkflowStep(
        identifier='session',
        title='Create or load session',
        description=(
            'Create a parquet-backed session and default runner. The session '
            'file is the single source of truth for the notebook workflow.'
        ),
    ),
    WorkflowStep(
        identifier='screening',
        title='Stage 1 — Screening',
        description=(
            'Collect initial clinical fields and run privacy checks before '
            'downstream clinical stages.'
        ),
        stage=Stage.SCREENING.value,
    ),
    WorkflowStep(
        identifier='intake',
        title='Stage 2 — Intake',
        description=(
            'Provide previous labs, reports, wearable data, or extracted '
            'structured data for the encounter.'
        ),
        stage=Stage.INTAKE.value,
    ),
    WorkflowStep(
        identifier='diagnosis_first_pass',
        title='Stage 3 — Diagnosis, first pass',
        description=(
            'Check requirements, answer available inquiries, defer future lab '
            'data, and run a preliminary diagnosis.'
        ),
        stage=Stage.DIAGNOSIS.value,
    ),
    WorkflowStep(
        identifier='exam',
        title='Stage 4 — Exam',
        description=(
            'Use the preliminary diagnosis to suggest labs and procedures.'
        ),
        stage=Stage.EXAM.value,
    ),
    WorkflowStep(
        identifier='lab_results',
        title='Multi-visit update — Lab results arrive',
        description=(
            'Reload the same session later and add deferred lab results such '
            'as stool analysis or food sensitivity panels.'
        ),
    ),
    WorkflowStep(
        identifier='diagnosis_enriched',
        title='Stage 3 — Enriched diagnosis rerun',
        description=(
            'Re-check diagnosis requirements and rerun diagnosis with all '
            'accumulated clinical and laboratory data.'
        ),
        stage=Stage.DIAGNOSIS.value,
    ),
    WorkflowStep(
        identifier='treatment',
        title='Stage 5 — Treatment',
        description=(
            'Collect treatment preferences and generate a treatment plan.'
        ),
        stage=Stage.TREATMENT.value,
    ),
    WorkflowStep(
        identifier='prescription',
        title='Stage 6 — Prescription',
        description=(
            'Generate medication or supplement recommendations from the '
            'validated treatment context.'
        ),
        stage=Stage.PRESCRIPTION.value,
    ),
    WorkflowStep(
        identifier='inspect_session',
        title='Inspect session',
        description=(
            'Review session path, completed stages, pending inquiries, '
            'results, and the parquet event log.'
        ),
    ),
)


@dataclass(frozen=True)
class NotebookSessionSummary:
    """
    title: Lightweight summary of the active notebook session.
    attributes:
      data_dir:
        type: Path
      sessions:
        type: tuple[Path, Ellipsis]
      session_path:
        type: Path | None
      language:
        type: str
      stages_completed:
        type: tuple[str, Ellipsis]
      pending_inquiries:
        type: int
      event_count:
        type: int
    """

    data_dir: Path
    sessions: tuple[Path, ...]
    session_path: Path | None
    language: str
    stages_completed: tuple[str, ...]
    pending_inquiries: int
    event_count: int
