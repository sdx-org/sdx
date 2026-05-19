"""
title: Notebook controller for session-backed HiperHealth workflows.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hiperhealth.notebook._app_manifest import (
    SkillAppViewRecord,
    collect_skill_app_views,
)
from hiperhealth.notebook._state import (
    STANDARD_STAGES,
    NotebookSessionSummary,
    StageStatus,
)
from hiperhealth.pipeline import (
    Inquiry,
    Session,
    Stage,
    StageRunner,
    create_default_runner,
)

if TYPE_CHECKING:
    from hiperhealth.llm import LLMSettings, StructuredLLM

StageInput = str | Stage


class NotebookController:
    """
    title: Coordinate notebook UI actions with sessions and the pipeline.
    summary: |-
      The controller intentionally avoids importing notebook-only packages so
      it can be tested and imported in a base hiperhealth installation.
    attributes:
      data_dir:
        type: Path
      sessions_dir:
        type: Path
      runner:
        type: StageRunner
      _language:
        type: str
      session:
        type: Session | None
    """

    data_dir: Path
    sessions_dir: Path
    runner: StageRunner
    _language: str
    session: Session | None

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        session_path: str | Path | None = None,
        language: str = 'en',
        runner: StageRunner | None = None,
    ) -> None:
        """
        title: Initialize the notebook controller.
        parameters:
          data_dir:
            type: str | Path | None
          session_path:
            type: str | Path | None
          language:
            type: str
          runner:
            type: StageRunner | None
        """
        self.data_dir: Path = self._default_data_dir(data_dir)
        self.sessions_dir: Path = self.data_dir / 'sessions'
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.runner: StageRunner = runner or create_default_runner()
        self._language: str = language
        self.session: Session | None = None

        if session_path is not None:
            self.load_or_create_session(session_path)

    @staticmethod
    def _default_data_dir(data_dir: str | Path | None) -> Path:
        """
        title: Resolve the notebook data directory.
        parameters:
          data_dir:
            type: str | Path | None
        returns:
          type: Path
        """
        if data_dir is None:
            return Path.home() / '.hiperhealth' / 'notebook'
        return Path(data_dir).expanduser()

    @staticmethod
    def _stage_value(stage: StageInput) -> str:
        """
        title: Normalize a stage input to its string value.
        parameters:
          stage:
            type: StageInput
        returns:
          type: str
        """
        if isinstance(stage, Stage):
            return stage.value
        return stage

    def resolve_session_path(self, path: str | Path) -> Path:
        """
        title: Resolve a session path relative to the notebook sessions folder.
        parameters:
          path:
            type: str | Path
        returns:
          type: Path
        """
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.sessions_dir / candidate
        if candidate.suffix == '':
            candidate = candidate.with_suffix('.parquet')
        return candidate

    def list_sessions(self) -> tuple[Path, ...]:
        """
        title: Return known notebook session files.
        returns:
          type: tuple[Path, Ellipsis]
        """
        if not self.sessions_dir.exists():
            return ()
        return tuple(sorted(self.sessions_dir.glob('*.parquet')))

    def create_session(
        self,
        name: str | Path | None = None,
        *,
        language: str | None = None,
    ) -> Session:
        """
        title: Create and activate a new parquet-backed session.
        parameters:
          name:
            type: str | Path | None
          language:
            type: str | None
        returns:
          type: Session
        """
        session_name = name or self._new_session_name()
        path = self.resolve_session_path(session_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        session = Session.create(path, language=language or self._language)
        self.session = session
        return session

    def load_session(self, path: str | Path) -> Session:
        """
        title: Load and activate an existing session.
        parameters:
          path:
            type: str | Path
        returns:
          type: Session
        """
        session = Session.load(self.resolve_session_path(path))
        self._language = session.language
        self.session = session
        return session

    def load_or_create_session(self, path: str | Path) -> Session:
        """
        title: Load a session if present, otherwise create it.
        parameters:
          path:
            type: str | Path
        returns:
          type: Session
        """
        resolved = self.resolve_session_path(path)
        if resolved.exists():
            return self.load_session(resolved)
        return self.create_session(resolved)

    def current_session(self) -> Session:
        """
        title: Return the active session or raise an actionable error.
        returns:
          type: Session
        """
        if self.session is None:
            msg = (
                'No active HiperHealth notebook session. Create or load a '
                'session before editing data or running stages.'
            )
            raise RuntimeError(msg)
        return self.session

    def set_clinical_data(self, fields: Mapping[str, Any]) -> None:
        """
        title: Add clinical data to the active session.
        parameters:
          fields:
            type: Mapping[str, Any]
        """
        self.current_session().set_clinical_data(dict(fields))

    def provide_answers(self, answers: Mapping[str, Any]) -> None:
        """
        title: Add inquiry answers to the active session.
        parameters:
          answers:
            type: Mapping[str, Any]
        """
        self.current_session().provide_answers(dict(answers))

    def provide_skill_ui_data(
        self,
        skill_id: str,
        view_id: str,
        values: Mapping[str, Any],
        *,
        stage: StageInput | None = None,
    ) -> None:
        """
        title: Add skill-specific UI values to the active session.
        parameters:
          skill_id:
            type: str
          view_id:
            type: str
          values:
            type: Mapping[str, Any]
          stage:
            type: StageInput | None
        """
        stage_value = self._stage_value(stage) if stage is not None else None
        self.current_session().provide_skill_ui_data(
            skill_id=skill_id,
            view_id=view_id,
            values=dict(values),
            stage=stage_value,
        )

    def check_requirements(
        self,
        stage: StageInput,
        *,
        disabled_skills: Collection[str] | None = None,
        llm: StructuredLLM | None = None,
        llm_settings: LLMSettings | None = None,
    ) -> list[Inquiry]:
        """
        title: Check stage requirements for the active session.
        parameters:
          stage:
            type: StageInput
          disabled_skills:
            type: Collection[str] | None
          llm:
            type: StructuredLLM | None
          llm_settings:
            type: LLMSettings | None
        returns:
          type: list[Inquiry]
        """
        return self.runner.check_requirements(
            self._stage_value(stage),
            self.current_session(),
            disabled_skills=disabled_skills,
            **self._runtime_kwargs(llm=llm, llm_settings=llm_settings),
        )

    def run_stage(
        self,
        stage: StageInput,
        *,
        disabled_skills: Collection[str] | None = None,
        llm: StructuredLLM | None = None,
        llm_settings: LLMSettings | None = None,
    ) -> Session:
        """
        title: Run one stage against the active session.
        parameters:
          stage:
            type: StageInput
          disabled_skills:
            type: Collection[str] | None
          llm:
            type: StructuredLLM | None
          llm_settings:
            type: LLMSettings | None
        returns:
          type: Session
        """
        return self.runner.run_session(
            self._stage_value(stage),
            self.current_session(),
            disabled_skills=disabled_skills,
            **self._runtime_kwargs(llm=llm, llm_settings=llm_settings),
        )

    def pending_inquiries(
        self,
        stage: StageInput | None = None,
        *,
        unique: bool = True,
    ) -> list[Inquiry]:
        """
        title: Return pending inquiries, optionally scoped to one stage.
        parameters:
          stage:
            type: StageInput | None
          unique:
            type: bool
        returns:
          type: list[Inquiry]
        """
        inquiries = self.current_session().pending_inquiries
        stage_value = self._stage_value(stage) if stage is not None else None
        if stage_value is not None:
            inquiries = [inq for inq in inquiries if inq.stage == stage_value]
        if not unique:
            return inquiries

        deduplicated: list[Inquiry] = []
        seen: set[tuple[str, str, str]] = set()
        for inquiry in inquiries:
            key = (inquiry.stage, inquiry.skill_name, inquiry.field)
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(inquiry)
        return deduplicated

    def stage_run_count(self, stage: StageInput) -> int:
        """
        title: Count completed runs for a stage in the active session.
        parameters:
          stage:
            type: StageInput
        returns:
          type: int
        """
        stage_value = self._stage_value(stage)
        session = self.current_session()
        return sum(
            1
            for event in session.events
            if event.get('event_type') == 'stage_completed'
            and event.get('stage') == stage_value
        )

    def stage_status(self, stage: StageInput) -> StageStatus:
        """
        title: Return a coarse status for a workflow stage.
        parameters:
          stage:
            type: StageInput
        returns:
          type: StageStatus
        """
        if self.session is None:
            return 'not_started'

        inquiries = self.pending_inquiries(stage)
        actionable = [
            inquiry
            for inquiry in inquiries
            if inquiry.priority in ('required', 'supplementary')
        ]
        if actionable:
            return 'needs_information'
        if inquiries:
            return 'waiting_deferred'
        if self.stage_run_count(stage) > 0:
            return 'complete'
        return 'ready'

    def result_for_stage(self, stage: StageInput) -> Any:
        """
        title: Return the current result payload for one stage.
        parameters:
          stage:
            type: StageInput
        returns:
          type: Any
        """
        stage_value = self._stage_value(stage)
        for key, value in self.current_session().results.items():
            if key == stage_value:
                return value
        return None

    @property
    def clinical_data(self) -> dict[str, Any]:
        """
        title: Return accumulated clinical data for the active session.
        returns:
          type: dict[str, Any]
        """
        return self.current_session().clinical_data

    @property
    def results(self) -> dict[str, Any]:
        """
        title: Return accumulated stage results for the active session.
        returns:
          type: dict[str, Any]
        """
        return self.current_session().results

    @property
    def events(self) -> list[dict[str, Any]]:
        """
        title: Return the active session event log.
        returns:
          type: list[dict[str, Any]]
        """
        return self.current_session().events

    def active_skill_names(self) -> tuple[str, ...]:
        """
        title: Return active runner skill names.
        returns:
          type: tuple[str, Ellipsis]
        """
        return tuple(skill.metadata.name for skill in self.runner.skills)

    def skill_app_views(
        self,
        *,
        stage: StageInput | None = None,
        phase: str | None = None,
    ) -> list[SkillAppViewRecord]:
        """
        title: Return active skill app views for optional rendering.
        parameters:
          stage:
            type: StageInput | None
          phase:
            type: str | None
        returns:
          type: list[SkillAppViewRecord]
        """
        stage_value = self._stage_value(stage) if stage is not None else None
        return collect_skill_app_views(
            self.runner,
            stage=stage_value,
            phase=phase,
        )

    def summary(self) -> NotebookSessionSummary:
        """
        title: Return a summary of notebook session state.
        returns:
          type: NotebookSessionSummary
        """
        if self.session is None:
            return NotebookSessionSummary(
                data_dir=self.data_dir,
                sessions=self.list_sessions(),
                session_path=None,
                language=self._language,
                stages_completed=(),
                pending_inquiries=0,
                event_count=0,
            )

        session = self.current_session()
        return NotebookSessionSummary(
            data_dir=self.data_dir,
            sessions=self.list_sessions(),
            session_path=session.path,
            language=session.language,
            stages_completed=tuple(session.stages_completed),
            pending_inquiries=len(session.pending_inquiries),
            event_count=len(session.events),
        )

    def workflow_statuses(self) -> dict[str, StageStatus]:
        """
        title: Return statuses for the standard stages.
        returns:
          type: dict[str, StageStatus]
        """
        return {stage: self.stage_status(stage) for stage in STANDARD_STAGES}

    @staticmethod
    def _new_session_name() -> str:
        """
        title: Return a timestamped default session filename.
        returns:
          type: str
        """
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        return f'hiperhealth-{stamp}.parquet'

    @staticmethod
    def _runtime_kwargs(
        *,
        llm: StructuredLLM | None,
        llm_settings: LLMSettings | None,
    ) -> dict[str, Any]:
        """
        title: Build optional runtime keyword arguments for pipeline calls.
        parameters:
          llm:
            type: StructuredLLM | None
          llm_settings:
            type: LLMSettings | None
        returns:
          type: dict[str, Any]
        """
        kwargs: dict[str, Any] = {}
        if llm is not None:
            kwargs['llm'] = llm
        if llm_settings is not None:
            kwargs['llm_settings'] = llm_settings
        return kwargs
