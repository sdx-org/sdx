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
from hiperhealth.pipeline.stages import Stage

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
        """
        title: Initialize an in-memory session wrapper for a parquet file.
        parameters:
          path:
            type: Path
          language:
            type: str
        """
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
    def skill_ui_data(self) -> dict[str, dict[str, dict[str, Any]]]:
        """
        title: Reconstruct skill-specific UI data by replaying events.
        returns:
          type: dict[str, dict[str, dict[str, Any]]]
        """
        data: dict[str, dict[str, dict[str, Any]]] = {}
        for event in self._events:
            if event['event_type'] != 'skill_ui_data_provided':
                continue
            payload = json.loads(event['data'])
            skill_id = payload.get('skill_id') or event.get('skill_name')
            view_id = payload.get('view_id')
            values = payload.get('values', {})
            if not isinstance(skill_id, str) or not isinstance(view_id, str):
                continue
            if not isinstance(values, dict):
                continue
            data.setdefault(skill_id, {})[view_id] = values
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

    def provide_skill_ui_data(
        self,
        skill_id: str,
        view_id: str,
        values: dict[str, Any],
        *,
        stage: str | None = None,
    ) -> None:
        """
        title: Persist skill-specific UI values for later pipeline hooks.
        parameters:
          skill_id:
            type: str
          view_id:
            type: str
          values:
            type: dict[str, Any]
          stage:
            type: str | None
        """
        self._append_event(
            'skill_ui_data_provided',
            stage=stage,
            skill_name=skill_id,
            data={
                'skill_id': skill_id,
                'view_id': view_id,
                'values': values,
            },
        )

    def summary(self) -> dict[str, Any]:
        """
        title: Return a plain-dict snapshot of the current session state.
        summary: |-
          Provides a single, JSON-serializable overview of the session
          without performing any additional I/O.  Useful for logging,
          notebook display, and integration-layer diagnostics.

          Returned keys:

          - session_id: filename stem of the underlying parquet file.
          - language: session language code.
          - clinical_data_fields: number of patient data fields collected.
          - stages_completed: ordered list of stages that have already run.
          - stages_pending: standard Stage enum members not yet completed.
          - pending_inquiries: total count of unanswered inquiry items.
          - required_inquiries: count of unanswered 'required'-priority items.
          - total_events: raw number of events in the event log.
        returns:
          type: dict[str, Any]
        """
        completed = self.stages_completed
        completed_set = set(completed)
        pending_inquiries = self.pending_inquiries
        return {
            'session_id': self.path.stem,
            'language': self._language,
            'clinical_data_fields': len(self.clinical_data),
            'stages_completed': list(completed),
            'stages_pending': [
                s.value for s in Stage if s.value not in completed_set
            ],
            'pending_inquiries': len(pending_inquiries),
            'required_inquiries': sum(
                1 for i in pending_inquiries if i.priority == 'required'
            ),
            'total_events': len(self._events),
        }

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
            extras={'skill_ui': self.skill_ui_data},
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
        """
        title: Append a new event and persist the session to disk.
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
        """
        title: Load session events from the parquet file.
        """
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
        """
        title: Write the current event log to the parquet file.
        """
        if not self._events:
            table = SESSION_SCHEMA.empty_table()
        else:
            table = pa.Table.from_pylist(self._events, schema=SESSION_SCHEMA)
        pq.write_table(table, self.path)


__all__ = ['SESSION_SCHEMA', 'Inquiry', 'Session']
