"""
title: Parquet-backed session file for pipeline interactions.
summary: |-
  Every interaction between System X and hiperhealth is recorded as
  an event row.  Current state (clinical data, results, pending
  inquiries) is derived by replaying events.  System X owns the file
  lifecycle (storage, deletion, retention).
"""

from __future__ import annotations

import json

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from pydantic import BaseModel

from hiperhealth.pipeline.context import PipelineContext

# ── Inquiry model ──────────────────────────────────────────────────


class Inquiry(BaseModel):
    """
    title: A single piece of information a skill needs to proceed.
    attributes:
      skill_name:
        type: str
      stage:
        type: str
      field:
        type: str
      label:
        type: str
      description:
        type: str
      priority:
        type: Literal[required, supplementary, deferred]
      input_type:
        type: str
      choices:
        type: list[str] | None
    """

    skill_name: str
    stage: str
    field: str
    label: str
    description: str = ''
    priority: Literal['required', 'supplementary', 'deferred'] = (
        'supplementary'
    )
    input_type: str = 'text'
    choices: list[str] | None = None


# ── Parquet schema ─────────────────────────────────────────────────

SESSION_SCHEMA = pa.schema(
    [
        pa.field('event_id', pa.uint32(), nullable=False),
        pa.field('timestamp', pa.timestamp('us', tz='UTC'), nullable=False),
        pa.field('event_type', pa.string(), nullable=False),
        pa.field('stage', pa.string(), nullable=True),
        pa.field('skill_name', pa.string(), nullable=True),
        pa.field('data', pa.string(), nullable=False),
    ]
)


# ── Session class ──────────────────────────────────────────────────


class Session:
    """
    title: Parquet-backed session that records every interaction.
    summary: |-
      System X creates or loads a session, provides clinical data,
      and uses the runner to assess / execute stages.  The parquet
      file is the single source of truth.
    attributes:
      path:
        type: Path
      _language:
        type: str
      _events:
        type: list[dict[str, Any]]
    """

    def __init__(self, path: Path, language: str = 'en') -> None:
        self.path: Path = path
        self._language: str = language
        self._events: list[dict[str, Any]] = []

    # ── Factory methods ────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        path: str | Path,
        language: str = 'en',
    ) -> Session:
        """
        title: Create a new session file.
        parameters:
          path:
            type: str | Path
          language:
            type: str
        returns:
          type: Session
        """
        path = Path(path)
        if path.exists():
            msg = f'Session file already exists: {path}'
            raise FileExistsError(msg)
        session = cls(path, language=language)
        session._save()
        return session

    @classmethod
    def load(cls, path: str | Path) -> Session:
        """
        title: Load an existing session from a parquet file.
        parameters:
          path:
            type: str | Path
        returns:
          type: Session
        """
        path = Path(path)
        if not path.exists():
            msg = f'Session file not found: {path}'
            raise FileNotFoundError(msg)
        session = cls(path)
        session._load()
        return session

    # ── Derived state ──────────────────────────────────────────────

    @property
    def language(self) -> str:
        """
        title: Return the session language.
        returns:
          type: str
        """
        return self._language

    @property
    def clinical_data(self) -> dict[str, Any]:
        """
        title: Reconstruct clinical data by replaying events.
        returns:
          type: dict[str, Any]
        """
        data: dict[str, Any] = {}
        for event in self._events:
            if event['event_type'] in (
                'clinical_data_set',
                'answers_provided',
            ):
                payload = json.loads(event['data'])
                data.update(payload.get('fields', {}))
        return data

    @property
    def results(self) -> dict[str, Any]:
        """
        title: Reconstruct stage results from completed events.
        returns:
          type: dict[str, Any]
        """
        results: dict[str, Any] = {}
        for event in self._events:
            if event['event_type'] == 'stage_completed':
                payload = json.loads(event['data'])
                results[event['stage']] = payload.get('results', {})
        return results

    @property
    def pending_inquiries(self) -> list[Inquiry]:
        """
        title: Inquiries not yet answered.
        returns:
          type: list[Inquiry]
        """
        answered = set(self.clinical_data.keys())
        pending: list[Inquiry] = []
        for event in self._events:
            if event['event_type'] == 'inquiries_raised':
                payload = json.loads(event['data'])
                for inq in payload.get('inquiries', []):
                    if inq['field'] not in answered:
                        pending.append(Inquiry.model_validate(inq))
        return pending

    @property
    def stages_completed(self) -> list[str]:
        """
        title: Which stages have been executed.
        returns:
          type: list[str]
        """
        return [
            e['stage']
            for e in self._events
            if e['event_type'] == 'stage_completed'
        ]

    @property
    def events(self) -> list[dict[str, Any]]:
        """
        title: Return a copy of all events.
        returns:
          type: list[dict[str, Any]]
        """
        return list(self._events)

    # ── System X actions ───────────────────────────────────────────

    def set_clinical_data(self, fields: dict[str, Any]) -> None:
        """
        title: Provide clinical information (no PII).
        parameters:
          fields:
            type: dict[str, Any]
        """
        self._append_event(
            'clinical_data_set',
            data={'fields': fields},
        )

    def provide_answers(self, answers: dict[str, Any]) -> None:
        """
        title: Provide answers to inquiries.
        parameters:
          answers:
            type: dict[str, Any]
        """
        self._append_event(
            'answers_provided',
            data={'fields': answers},
        )

    # ── Context bridge ─────────────────────────────────────────────

    def to_context(self) -> PipelineContext:
        """
        title: Build a PipelineContext from current session state.
        returns:
          type: PipelineContext
        """
        return PipelineContext(
            patient=self.clinical_data,
            language=self._language,
            session_id=self.path.stem,
            results=self.results,
            extras={},
        )

    def update_from_context(
        self,
        stage: str,
        ctx: PipelineContext,
    ) -> None:
        """
        title: Capture results after a stage runs.
        parameters:
          stage:
            type: str
          ctx:
            type: PipelineContext
        """
        stage_result = ctx.results.get(stage)
        result_data: Any
        if stage_result is not None:
            if hasattr(stage_result, 'model_dump'):
                result_data = stage_result.model_dump()
            else:
                result_data = stage_result
        else:
            result_data = {}
        self._append_event(
            'stage_completed',
            stage=stage,
            data={'results': result_data},
        )

    # ── Event recording ────────────────────────────────────────────

    def record_event(
        self,
        event_type: str,
        stage: str | None = None,
        skill_name: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        title: Record an arbitrary event (used by the runner).
        parameters:
          event_type:
            type: str
          stage:
            type: str | None
          skill_name:
            type: str | None
          data:
            type: dict[str, Any] | None
        """
        self._append_event(
            event_type,
            stage=stage,
            skill_name=skill_name,
            data=data,
        )

    # ── Internal I/O ───────────────────────────────────────────────

    def _append_event(
        self,
        event_type: str,
        stage: str | None = None,
        skill_name: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        event: dict[str, Any] = {
            'event_id': len(self._events),
            'timestamp': datetime.now(timezone.utc),
            'event_type': event_type,
            'stage': stage,
            'skill_name': skill_name,
            'data': json.dumps(
                data if data is not None else {},
                ensure_ascii=False,
                default=str,
            ),
        }
        self._events.append(event)
        self._save()

    def _load(self) -> None:
        table = pq.read_table(self.path, schema=SESSION_SCHEMA)
        rows = table.to_pylist()
        self._events = rows
        # Recover language from first clinical_data_set if present
        for event in rows:
            if event['event_type'] == 'clinical_data_set':
                payload = json.loads(event['data'])
                lang = payload.get('fields', {}).get('language')
                if lang:
                    self._language = lang
                break

    def _save(self) -> None:
        if not self._events:
            table = SESSION_SCHEMA.empty_table()
        else:
            table = pa.Table.from_pylist(self._events, schema=SESSION_SCHEMA)
        pq.write_table(table, self.path)


__all__ = ['SESSION_SCHEMA', 'Inquiry', 'Session']
