"""
title: Tests for real notebook widget rendering.
"""

# ruff: noqa: E402

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from textwrap import dedent
from typing import TypeVar

import pytest

widgets = pytest.importorskip('ipywidgets')

from hiperhealth.notebook import ui
from hiperhealth.notebook._app_manifest import SkillAppViewRecord
from hiperhealth.notebook._controller import NotebookController
from hiperhealth.notebook._ipywidgets_forms import SkillAppForm
from hiperhealth.notebook._widgets import NotebookUI
from hiperhealth.pipeline import (
    BaseSkill,
    Inquiry,
    PipelineContext,
    SkillMetadata,
    SkillRegistry,
    SkillUIView,
    Stage,
    StageRunner,
)

TWidget = TypeVar('TWidget', bound=widgets.Widget)


class _WidgetSkill(BaseSkill):
    """
    title: Deterministic skill used by real notebook widget tests.
    """

    def __init__(self) -> None:
        """
        title: Initialize the deterministic widget test skill.
        """
        super().__init__(
            SkillMetadata(
                name='widget.skill',
                stages=(Stage.DIAGNOSIS.value,),
            )
        )

    def check_requirements(
        self,
        stage: str,
        ctx: PipelineContext,
    ) -> list[Inquiry]:
        """
        title: Request a chief complaint when it is missing.
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
                priority='required',
                input_type='textarea',
            ),
            Inquiry(
                skill_name=self.metadata.name,
                stage=stage,
                field='severity',
                label='Severity',
                priority='supplementary',
                input_type='choice',
                choices=['mild', 'moderate', 'severe'],
            ),
        ]

    def execute(self, stage: str, ctx: PipelineContext) -> PipelineContext:
        """
        title: Write a deterministic stage result.
        parameters:
          stage:
            type: str
          ctx:
            type: PipelineContext
        returns:
          type: PipelineContext
        """
        ctx.results[stage] = {
            'summary': 'Widget diagnosis complete',
            'fields': sorted(ctx.patient),
        }
        return ctx


def _walk(widget: widgets.Widget) -> Iterator[widgets.Widget]:
    """
    title: Yield a widget and all recursive children.
    parameters:
      widget:
        type: widgets.Widget
    returns:
      type: Iterator[widgets.Widget]
    """
    yield widget
    for child in getattr(widget, 'children', ()):  # pragma: no branch
        if isinstance(child, widgets.Widget):
            yield from _walk(child)


def _find_widget(
    root: widgets.Widget,
    widget_type: type[TWidget],
    *,
    description: str | None = None,
) -> TWidget:
    """
    title: Return the first widget matching a type and optional description.
    parameters:
      root:
        type: widgets.Widget
      widget_type:
        type: type[TWidget]
      description:
        type: str | None
    returns:
      type: TWidget
    """
    for widget in _walk(root):
        if not isinstance(widget, widget_type):
            continue
        if description is not None and widget.description != description:
            continue
        return widget
    msg = f'Widget not found: {widget_type.__name__} {description!r}'
    raise AssertionError(msg)


def _find_button(root: widgets.Widget, description: str) -> widgets.Button:
    """
    title: Return a button by description.
    parameters:
      root:
        type: widgets.Widget
      description:
        type: str
    returns:
      type: widgets.Button
    """
    return _find_widget(root, widgets.Button, description=description)


def _create_skill_app_channel_repo(path: Path) -> Path:
    """
    title: Create a local channel repository with an app-enabled skill.
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
              name: dynamic-apps
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
              title: Dynamic App Skill
              views:
                - id: options
                  title: Dynamic options
                  stage: diagnosis
                  phase: pre_run
                  data_schema:
                    type: object
                    required:
                      - focus
                    properties:
                      focus:
                        type: string
                        title: Focus
                      include_notes:
                        type: boolean
                        title: Include notes
                  ui_schema:
                    type: VerticalLayout
                    elements:
                      - type: Control
                        scope: "#/properties/focus"
                        options:
                          x-hiperhealth-binding:
                            target: session.skill_ui_data
                      - type: Control
                        scope: "#/properties/include_notes"
                        options:
                          x-hiperhealth-binding:
                            target: context.extras.skill_ui
                  actions:
                    - id: save
                      label: Save app data
                      type: session.provide_skill_ui_data
                    - id: run
                      label: Run diagnosis
                      type: runner.run_session
            """
        ),
        encoding='utf-8',
    )
    return path


def _app_controller(tmp_path: Path) -> NotebookController:
    """
    title: Return a controller with an installed app-enabled skill.
    parameters:
      tmp_path:
        type: Path
    returns:
      type: NotebookController
    """
    repo = _create_skill_app_channel_repo(tmp_path / 'channel')
    registry = SkillRegistry(
        registry_dir=tmp_path / 'registry' / 'artifacts' / 'skills'
    )
    registry.add_channel(str(repo), local_name='apps')
    registry.install_skill('apps.appskill')
    runner = StageRunner(registry=registry)
    runner.register('apps.appskill')
    return NotebookController(data_dir=tmp_path / 'notebook', runner=runner)


def _app_record() -> SkillAppViewRecord:
    """
    title: Build a standalone app view record for action dispatch tests.
    returns:
      type: SkillAppViewRecord
    """
    view = SkillUIView(
        id='options',
        title='Options',
        stage='diagnosis',
        phase='pre_run',
        data_schema={
            'type': 'object',
            'properties': {
                'focus': {'type': 'string', 'title': 'Focus'},
            },
        },
        ui_schema={
            'type': 'VerticalLayout',
            'elements': [
                {
                    'type': 'Control',
                    'scope': '#/properties/focus',
                    'options': {
                        'x-hiperhealth-binding': {
                            'target': 'session.skill_ui_data',
                        },
                    },
                }
            ],
        },
    )
    return SkillAppViewRecord(
        skill_id='widget.skill',
        skill_title='Widget Skill',
        view=view,
    )


def test_skill_app_form_renders_and_collects_values() -> None:
    """
    title: SkillAppForm should render real widgets and collect typed values.
    """
    form = SkillAppForm(
        data_schema={
            'type': 'object',
            'required': ['name'],
            'properties': {
                'name': {
                    'type': 'string',
                    'title': 'Name',
                    'description': 'Synthetic name',
                },
                'notes': {'type': 'string', 'title': 'Notes'},
                'enabled': {'type': 'boolean', 'title': 'Enabled'},
                'count': {'type': 'integer', 'title': 'Count'},
                'ratio': {'type': 'number', 'title': 'Ratio'},
                'focus': {
                    'type': 'string',
                    'title': 'Focus',
                    'enum': ['broad', 'gut'],
                },
                'priority': {
                    'type': 'string',
                    'title': 'Priority',
                    'enum': ['low', 'high'],
                },
                'items': {'type': 'array', 'title': 'Items'},
                'metadata': {'type': 'object', 'title': 'Metadata'},
            },
        },
        ui_schema={
            'type': 'Categorization',
            'elements': [
                {
                    'type': 'Category',
                    'label': 'Main',
                    'elements': [
                        {
                            'type': 'Group',
                            'label': 'Basics',
                            'elements': [
                                {
                                    'type': 'HorizontalLayout',
                                    'options': {'width': '80%'},
                                    'elements': [
                                        {
                                            'type': 'Control',
                                            'scope': '#/properties/name',
                                            'options': {
                                                'x-hiperhealth-binding': {
                                                    'target': (
                                                        'session.answers'
                                                    ),
                                                    'field': 'chief_complaint',
                                                }
                                            },
                                        },
                                        {
                                            'type': 'Control',
                                            'scope': '#/properties/enabled',
                                        },
                                    ],
                                },
                                {
                                    'type': 'Control',
                                    'scope': '#/properties/notes',
                                    'options': {
                                        'multi': True,
                                        'height': '80px',
                                    },
                                },
                                {
                                    'type': 'Control',
                                    'scope': '#/properties/count',
                                },
                                {
                                    'type': 'Control',
                                    'scope': '#/properties/ratio',
                                },
                                {
                                    'type': 'Control',
                                    'scope': '#/properties/focus',
                                },
                                {
                                    'type': 'Control',
                                    'scope': '#/properties/priority',
                                    'options': {'format': 'radio'},
                                },
                                {
                                    'type': 'Control',
                                    'scope': '#/properties/items',
                                },
                                {
                                    'type': 'Control',
                                    'scope': '#/properties/metadata',
                                },
                            ],
                        }
                    ],
                }
            ],
        },
        initial_values={
            'name': 'Bloating',
            'enabled': True,
            'count': 2,
            'ratio': 1.5,
            'focus': 'gut',
            'priority': 'high',
            'items': ['a'],
            'metadata': {'source': 'synthetic'},
        },
    )

    assert isinstance(form.widget, widgets.Tab)
    form._controls['notes'].value = 'Some notes'
    form._controls['items'].value = '["x", "y"]'
    form._controls['metadata'].value = '{"source": "manual"}'

    values = form.values()
    grouped = form.binding_values()

    assert values['enabled'] is True
    assert values['items'] == ['x', 'y']
    assert values['metadata'] == {'source': 'manual'}
    assert grouped['session.answers']['chief_complaint'] == 'Bloating'


def test_skill_app_form_validation_paths() -> None:
    """
    title: SkillAppForm should validate required fields and JSON editors.
    """
    form = SkillAppForm(
        data_schema={
            'type': 'object',
            'required': ['required_text'],
            'properties': {
                'required_text': {'type': 'string'},
                'payload': {'type': 'object'},
            },
        },
        ui_schema={
            'type': 'VerticalLayout',
            'elements': [
                {'type': 'Control', 'scope': '#/properties/required_text'},
                {'type': 'Control', 'scope': '#/properties/payload'},
            ],
        },
    )

    with pytest.raises(ValueError, match='Missing required'):
        form.values()

    form._controls['required_text'].value = 'present'
    form._controls['payload'].value = '{invalid'
    with pytest.raises(ValueError, match='valid JSON'):
        form.values()

    unsupported = SkillAppForm(
        data_schema={'type': 'object'},
        ui_schema={'type': 'Unsupported'},
    )
    assert isinstance(unsupported.widget, widgets.HTML)


def test_notebook_ui_real_page_interactions(tmp_path: Path) -> None:
    """
    title: NotebookUI should support real session/page button interactions.
    parameters:
      tmp_path:
        type: Path
    """
    controller = NotebookController(
        data_dir=tmp_path,
        runner=StageRunner(skills=[_WidgetSkill()]),
    )
    notebook = NotebookUI(controller=controller)

    name = _find_widget(notebook.widget, widgets.Text, description='Name')
    name.value = 'case-study'
    _find_button(notebook.widget, 'Create session').click()
    assert controller.session is not None

    _find_button(notebook.widget, 'Load selected').click()
    assert controller.current_session().path.name == 'case-study.parquet'

    notebook._nav.value = 'patient'
    _find_button(notebook.widget, 'Load example screening fields').click()
    editor = _find_widget(
        notebook.widget,
        widgets.Textarea,
        description='Clinical JSON',
    )
    assert 'symptoms' in editor.value
    _find_button(notebook.widget, 'Save clinical data').click()
    assert 'symptoms' in controller.clinical_data

    notebook._nav.value = 'workflow'
    _find_button(notebook.widget, 'Check requirements').click()
    assert controller.current_session().pending_inquiries

    notebook._nav.value = 'inquiries'
    stage_filter = _find_widget(
        notebook.widget,
        widgets.Dropdown,
        description='Stage',
    )
    stage_filter.value = 'diagnosis'
    answer = _find_widget(
        notebook.widget,
        widgets.Textarea,
        description='Answer',
    )
    answer.value = 'Synthetic abdominal discomfort'
    _find_button(notebook.widget, 'Save answers').click()
    assert 'chief_complaint' in controller.clinical_data

    notebook._nav.value = 'workflow'
    _find_button(notebook.widget, 'Run stage').click()
    assert controller.result_for_stage(Stage.DIAGNOSIS)['summary'] == (
        'Widget diagnosis complete'
    )

    notebook._nav.value = 'results'
    assert _find_widget(notebook.widget, widgets.Accordion) is not None

    notebook._nav.value = 'events'
    _find_button(notebook.widget, 'Show selected event data').click()
    assert _find_widget(notebook.widget, widgets.Textarea).value

    notebook._nav.value = 'settings'
    assert 'widget.skill' in notebook._content.children[1].value


def test_notebook_ui_empty_and_error_states(tmp_path: Path) -> None:
    """
    title: NotebookUI should render empty states and validation errors.
    parameters:
      tmp_path:
        type: Path
    """
    controller = NotebookController(data_dir=tmp_path, runner=StageRunner())
    notebook = NotebookUI(controller=controller)

    assert notebook._safe_clinical_data() == {}
    assert notebook._safe_stage_run_count('diagnosis') == 0
    assert notebook._session_options() == (('No sessions found', ''),)

    notebook._nav.value = 'patient'
    editor = _find_widget(
        notebook.widget,
        widgets.Textarea,
        description='Clinical JSON',
    )
    editor.value = '[]'
    _find_button(notebook.widget, 'Save clinical data').click()
    assert 'must be a JSON object' in notebook._alert.value

    notebook._nav.value = 'results'
    assert 'No active' in notebook._content.children[0].value

    notebook._nav.value = 'events'
    assert 'No active' in notebook._content.children[0].value

    controller.create_session('empty')
    notebook._nav.value = 'results'
    assert 'No stage results' in notebook._content.children[0].value
    notebook._nav.value = 'events'
    assert 'No session events' in notebook._content.children[0].value

    assert NotebookUI._parse_json_object('{"a": 1}', 'Payload') == {'a': 1}
    with pytest.raises(ValueError, match='valid JSON'):
        NotebookUI._parse_json_object('{bad', 'Payload')
    assert NotebookUI._button_style('secondary') == ''
    assert NotebookUI._button_style('danger') == 'danger'
    assert NotebookUI._skill_ui_values(
        {'context.extras.skill_ui': {'focus': 'gut'}},
        {'fallback': True},
    ) == {'focus': 'gut'}
    assert '<table' in NotebookUI._event_table(controller.events)


def test_notebook_ui_skill_app_page_and_actions(tmp_path: Path) -> None:
    """
    title: NotebookUI should render app manifests and dispatch real actions.
    parameters:
      tmp_path:
        type: Path
    """
    controller = _app_controller(tmp_path)
    controller.create_session('app-session')
    notebook = NotebookUI(controller=controller)

    notebook._nav.value = 'skill_apps'
    focus = _find_widget(notebook.widget, widgets.Text, description='Focus')
    focus.value = 'gastrointestinal'
    include_notes = _find_widget(
        notebook.widget,
        widgets.Checkbox,
        description='Include notes',
    )
    include_notes.value = True
    _find_button(notebook.widget, 'Save app data').click()

    assert controller.current_session().skill_ui_data['apps.appskill'][
        'options'
    ] == {'focus': 'gastrointestinal'}

    _find_button(notebook.widget, 'Run diagnosis').click()
    assert controller.stage_run_count(Stage.DIAGNOSIS) == 1

    phase_filter = _find_widget(
        notebook.widget,
        widgets.Dropdown,
        description='Phase',
    )
    phase_filter.value = 'settings'
    empty_message = notebook._content.children[3].children[0].value
    assert 'No active skill app views' in empty_message


def test_skill_app_action_dispatch_paths(tmp_path: Path) -> None:
    """
    title: Skill app dispatcher should execute every allowlisted action.
    parameters:
      tmp_path:
        type: Path
    """
    controller = NotebookController(
        data_dir=tmp_path,
        runner=StageRunner(skills=[_WidgetSkill()]),
    )
    controller.create_session('dispatch')
    notebook = NotebookUI(controller=controller)
    record = _app_record()

    def form_for(
        target: str,
        value: str = 'value',
    ) -> SkillAppForm:
        """
        title: Build a one-control form bound to a target.
        parameters:
          target:
            type: str
          value:
            type: str
        returns:
          type: SkillAppForm
        """
        form = SkillAppForm(
            data_schema={
                'type': 'object',
                'properties': {
                    'focus': {'type': 'string', 'title': 'Focus'},
                },
            },
            ui_schema={
                'type': 'VerticalLayout',
                'elements': [
                    {
                        'type': 'Control',
                        'scope': '#/properties/focus',
                        'options': {
                            'x-hiperhealth-binding': {'target': target},
                        },
                    }
                ],
            },
        )
        form._controls['focus'].value = value
        return form

    notebook._dispatch_skill_app_action(
        record,
        form_for('session.clinical_data', 'clinical'),
        'session.set_clinical_data',
    )
    assert controller.clinical_data['focus'] == 'clinical'

    notebook._dispatch_skill_app_action(
        record,
        form_for('session.answers', 'answer'),
        'session.provide_answers',
    )
    assert controller.clinical_data['focus'] == 'answer'

    notebook._dispatch_skill_app_action(
        record,
        form_for('session.skill_ui_data', 'ui-data'),
        'session.provide_skill_ui_data',
    )
    assert controller.current_session().skill_ui_data['widget.skill'][
        'options'
    ] == {'focus': 'ui-data'}

    notebook._dispatch_skill_app_action(
        record,
        form_for('session.skill_ui_data', 'precheck'),
        'runner.check_requirements',
    )
    assert controller.current_session().pending_inquiries

    notebook._dispatch_skill_app_action(
        record,
        form_for('session.skill_ui_data', 'run'),
        'runner.run_session',
    )
    assert controller.stage_run_count(Stage.DIAGNOSIS) == 1

    with pytest.raises(ValueError, match='Unsupported'):
        notebook._dispatch_skill_app_action(
            record,
            form_for('session.skill_ui_data'),
            'unsupported.action',
        )

    empty_form = SkillAppForm(
        data_schema={'type': 'object', 'properties': {}},
        ui_schema={'type': 'VerticalLayout', 'elements': []},
    )
    notebook._save_skill_ui_before_runner_action(record, {}, {})
    callback = notebook._skill_app_action_handler(
        record,
        empty_form,
        'session.provide_skill_ui_data',
        widgets.Output(),
    )
    callback(None)


def test_ui_show_with_real_widgets(tmp_path: Path) -> None:
    """
    title: ui.show should instantiate the real NotebookUI when widgets exist.
    parameters:
      tmp_path:
        type: Path
    """
    app = ui.show(data_dir=tmp_path, runner=StageRunner())

    assert isinstance(app, NotebookUI)
