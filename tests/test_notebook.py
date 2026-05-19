"""
title: Tests for optional notebook UI support.
"""

from __future__ import annotations

import builtins
import importlib
import sys

from collections.abc import Iterator
from pathlib import Path
from textwrap import dedent

import pytest

from hiperhealth.notebook import ui
from hiperhealth.notebook._controller import NotebookController
from hiperhealth.pipeline import (
    BaseSkill,
    Inquiry,
    PipelineContext,
    SkillMetadata,
    SkillRegistry,
    Stage,
    StageRunner,
)


def _create_skill_app_channel_repo(path: Path) -> Path:
    """
    title: Create a channel repo fixture with one app-enabled skill.
    parameters:
      path:
        type: Path
    returns:
      type: Path
    """
    skill_dir = path / 'skills' / 'appskill'
    skill_dir.mkdir(parents=True)
    (path / 'skills-channel.yaml').write_text(
        dedent(
            """
            api_version: 1
            channel:
              name: traditional-medicine
              display_name: Traditional Medicine
            skills:
              - name: appskill
            """
        ),
        encoding='utf-8',
    )
    (skill_dir / 'skill.py').write_text(
        dedent(
            """
            from hiperhealth.pipeline import BaseSkill, SkillMetadata


            class AppSkill(BaseSkill):
                def __init__(self):
                    super().__init__(
                        SkillMetadata(
                            name='appskill',
                            stages=('diagnosis',),
                        )
                    )
            """
        ),
        encoding='utf-8',
    )
    (skill_dir / 'skill.yaml').write_text(
        dedent(
            """
            name: appskill
            version: 1.0.0
            entry_point: "skill:AppSkill"
            stages:
              - diagnosis
            app:
              api_version: 1
              title: App Skill
              views:
                - id: diagnosis_options
                  title: Diagnosis options
                  stage: diagnosis
                  phase: pre_run
                  data_schema:
                    type: object
                    properties:
                      focus:
                        type: string
                        title: Focus
                  ui_schema:
                    type: VerticalLayout
                    elements:
                      - type: Control
                        scope: "#/properties/focus"
                        options:
                          x-hiperhealth-binding:
                            target: session.skill_ui_data
                  actions:
                    - id: save
                      label: Save
                      type: session.provide_skill_ui_data
            """
        ),
        encoding='utf-8',
    )
    return path


class _NotebookTestSkill(BaseSkill):
    """
    title: Small deterministic skill for notebook controller tests.
    """

    def __init__(self) -> None:
        """
        title: Initialize the deterministic notebook test skill.
        """
        super().__init__(
            SkillMetadata(
                name='notebook_test',
                stages=(Stage.DIAGNOSIS.value,),
            )
        )

    def check_requirements(
        self,
        stage: str,
        ctx: PipelineContext,
    ) -> list[Inquiry]:
        """
        title: Ask for a chief complaint when it is missing.
        parameters:
          stage:
            type: str
          ctx:
            type: PipelineContext
        returns:
          type: list[Inquiry]
        """
        if 'chief_complaint' in ctx.patient:
            return []
        return [
            Inquiry(
                skill_name=self.metadata.name,
                stage=stage,
                field='chief_complaint',
                label='Chief complaint',
                description='Synthetic test inquiry.',
                priority='required',
                input_type='textarea',
            )
        ]

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        """
        title: Write a deterministic result.
        parameters:
          stage:
            type: str
          ctx:
            type: PipelineContext
        returns:
          type: PipelineContext
        """
        ctx.results[stage] = {
            'summary': 'Synthetic diagnosis result',
            'fields': sorted(ctx.patient),
        }
        return ctx


@pytest.fixture
def clean_notebook_modules() -> Iterator[None]:
    """
    title: Remove optional notebook widget modules from import cache.
    returns:
      type: Iterator[None]
    """
    removed = {
        name: module
        for name, module in sys.modules.items()
        if name == 'hiperhealth.notebook._widgets'
        or name.startswith('hiperhealth.notebook._widgets.')
        or name == 'ipywidgets'
        or name.startswith('ipywidgets.')
    }
    for name in removed:
        sys.modules.pop(name, None)
    yield
    for name in list(sys.modules):
        if name == 'hiperhealth.notebook._widgets':
            sys.modules.pop(name, None)
    sys.modules.update(removed)


def test_ui_import_is_lazy(clean_notebook_modules: None) -> None:
    """
    title: Importing the notebook UI gateway should not import widgets.
    parameters:
      clean_notebook_modules:
        type: None
    """
    sys.modules.pop('hiperhealth.notebook.ui', None)

    module = importlib.import_module('hiperhealth.notebook.ui')

    assert module is not None
    assert 'hiperhealth.notebook._widgets' not in sys.modules
    assert 'ipywidgets' not in sys.modules


def test_show_reports_missing_notebook_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_notebook_modules: None,
) -> None:
    """
    title: ui.show should raise an actionable install message when needed.
    parameters:
      tmp_path:
        type: Path
      monkeypatch:
        type: pytest.MonkeyPatch
      clean_notebook_modules:
        type: None
    """
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        """
        title: Simulate a missing ipywidgets dependency during import.
        parameters:
          name:
            type: str
          globals:
            type: dict[str, object] | None
          locals:
            type: dict[str, object] | None
          fromlist:
            type: tuple[str, Ellipsis]
          level:
            type: int
        returns:
          type: object
        """
        if name == 'ipywidgets':
            raise ModuleNotFoundError(
                "No module named 'ipywidgets'",
                name='ipywidgets',
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    with pytest.raises(ui.NotebookDependencyError, match='hiperhealth'):
        ui.show(data_dir=tmp_path)


def test_controller_runs_session_workflow(tmp_path: Path) -> None:
    """
    title: NotebookController should support the assess-answer-run cycle.
    parameters:
      tmp_path:
        type: Path
    """
    runner = StageRunner(skills=[_NotebookTestSkill()])
    controller = NotebookController(data_dir=tmp_path, runner=runner)

    session = controller.create_session('case-study')
    assert session.path == tmp_path / 'sessions' / 'case-study.parquet'

    controller.set_clinical_data({'symptoms': 'Synthetic bloating symptom'})
    inquiries = controller.check_requirements(Stage.DIAGNOSIS)

    assert len(inquiries) == 1
    assert inquiries[0].field == 'chief_complaint'
    assert controller.stage_status(Stage.DIAGNOSIS) == 'needs_information'

    controller.provide_answers(
        {'chief_complaint': 'Synthetic abdominal discomfort'}
    )
    controller.run_stage(Stage.DIAGNOSIS)

    assert controller.stage_run_count(Stage.DIAGNOSIS) == 1
    assert controller.stage_status(Stage.DIAGNOSIS) == 'complete'
    assert controller.result_for_stage(Stage.DIAGNOSIS)['summary'] == (
        'Synthetic diagnosis result'
    )
    assert controller.summary().event_count >= 5


def test_controller_loads_existing_session(tmp_path: Path) -> None:
    """
    title: NotebookController should load sessions relative to data_dir.
    parameters:
      tmp_path:
        type: Path
    """
    runner = StageRunner(skills=[_NotebookTestSkill()])
    first = NotebookController(data_dir=tmp_path, runner=runner)
    first.create_session('load-me')
    first.set_clinical_data({'symptoms': 'Synthetic fatigue'})

    second = NotebookController(data_dir=tmp_path, runner=runner)
    loaded = second.load_session('load-me')

    assert loaded.path == tmp_path / 'sessions' / 'load-me.parquet'
    assert second.clinical_data['symptoms'] == 'Synthetic fatigue'


def test_controller_discovers_skill_app_views(tmp_path: Path) -> None:
    """
    title: NotebookController should expose app views for active skills.
    parameters:
      tmp_path:
        type: Path
    """
    repo = _create_skill_app_channel_repo(tmp_path / 'channel')
    registry = SkillRegistry(
        registry_dir=tmp_path / 'registry' / 'artifacts' / 'skills'
    )
    registry.add_channel(str(repo), local_name='tm')
    registry.install_skill('tm.appskill')
    runner = StageRunner(registry=registry)
    runner.register('tm.appskill')
    controller = NotebookController(
        data_dir=tmp_path / 'notebook',
        runner=runner,
    )

    views = controller.skill_app_views(
        stage=Stage.DIAGNOSIS,
        phase='pre_run',
    )

    assert len(views) == 1
    assert views[0].skill_id == 'tm.appskill'
    assert views[0].view.id == 'diagnosis_options'

    controller.create_session('skill-app')
    controller.provide_skill_ui_data(
        skill_id='tm.appskill',
        view_id='diagnosis_options',
        values={'focus': 'gastrointestinal'},
        stage=Stage.DIAGNOSIS,
    )

    assert controller.current_session().skill_ui_data == {
        'tm.appskill': {'diagnosis_options': {'focus': 'gastrointestinal'}}
    }
