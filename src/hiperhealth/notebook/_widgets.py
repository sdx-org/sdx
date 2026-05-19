"""
title: ipywidgets implementation for the optional notebook interface.
"""

from __future__ import annotations

import json

from collections.abc import Collection
from html import escape
from typing import TYPE_CHECKING, Any, Callable, cast

import ipywidgets as widgets

from IPython.display import display as ipython_display
from pydantic import ValidationError

from hiperhealth.llm import LLMSettings
from hiperhealth.notebook._app_manifest import SkillAppViewRecord
from hiperhealth.notebook._controller import NotebookController
from hiperhealth.notebook._ipywidgets_forms import SkillAppForm
from hiperhealth.notebook._state import STANDARD_STAGES, WORKFLOW_STEPS
from hiperhealth.pipeline import Inquiry

if TYPE_CHECKING:
    from hiperhealth.llm import StructuredLLM

_NAV_OPTIONS: tuple[tuple[str, str], ...] = (
    ('🏠 Home', 'home'),
    ('🧾 Patient data', 'patient'),
    ('🧭 Workflow', 'workflow'),
    ('❓ Inquiries', 'inquiries'),
    ('🧩 Skill apps', 'skill_apps'),
    ('📋 Results', 'results'),
    ('🕘 Events', 'events'),
    ('⚙️ Skills & settings', 'settings'),
)

_STAGE_LABELS: tuple[tuple[str, str], ...] = (
    ('Screening', 'screening'),
    ('Intake', 'intake'),
    ('Diagnosis', 'diagnosis'),
    ('Exam', 'exam'),
    ('Treatment', 'treatment'),
    ('Prescription', 'prescription'),
)

_ALERT_COLORS = {
    'info': ('#e7f3ff', '#084298'),
    'success': ('#e8f7ee', '#0f5132'),
    'warning': ('#fff3cd', '#664d03'),
    'danger': ('#f8d7da', '#842029'),
}


class NotebookUI:
    """
    title: Interactive notebook widget for HiperHealth sessions.
    attributes:
      controller:
        type: NotebookController
      _llm:
        type: StructuredLLM | None
      _llm_settings:
        type: LLMSettings | None
      _provider:
        type: widgets.Text
      _model:
        type: widgets.Text
      _api_key:
        type: widgets.Password
      _persist_raw:
        type: widgets.Checkbox
      _nav:
        type: widgets.Select
      _alert:
        type: widgets.HTML
      _content:
        type: widgets.VBox
      widget:
        type: widgets.VBox
        description: Root ipywidgets container.
    """

    controller: NotebookController
    _llm: StructuredLLM | None
    _llm_settings: LLMSettings | None
    _provider: widgets.Text
    _model: widgets.Text
    _api_key: widgets.Password
    _persist_raw: widgets.Checkbox
    _nav: widgets.Select
    _alert: widgets.HTML
    _content: widgets.VBox
    widget: widgets.VBox

    def __init__(
        self,
        *,
        controller: NotebookController,
        llm: StructuredLLM | None = None,
        llm_settings: LLMSettings | None = None,
    ) -> None:
        """
        title: Build the notebook widget shell.
        parameters:
          controller:
            type: NotebookController
          llm:
            type: StructuredLLM | None
          llm_settings:
            type: LLMSettings | None
        """
        self.controller: NotebookController = controller
        self._llm: StructuredLLM | None = llm
        self._llm_settings: LLMSettings | None = llm_settings
        self._provider: widgets.Text = widgets.Text(
            description='Provider',
            placeholder='openai, ollama, groq, ...',
            value=llm_settings.provider if llm_settings else '',
        )
        self._model: widgets.Text = widgets.Text(
            description='Model',
            placeholder='o4-mini, llama3.2:3b, ...',
            value=llm_settings.model if llm_settings else '',
        )
        self._api_key: widgets.Password = widgets.Password(
            description='API key',
            placeholder='Not persisted by the notebook UI',
            value=llm_settings.api_key if llm_settings else '',
        )
        self._persist_raw: widgets.Checkbox = widgets.Checkbox(
            description='Persist raw provider payloads',
            value=llm_settings.persist_raw if llm_settings else False,
        )
        self._nav: widgets.Select = widgets.Select(
            options=_NAV_OPTIONS,
            value='home',
            rows=len(_NAV_OPTIONS),
            layout=widgets.Layout(width='220px'),
        )
        self._alert: widgets.HTML = widgets.HTML()
        self._content: widgets.VBox = widgets.VBox(
            layout=widgets.Layout(width='100%')
        )
        self._nav.observe(self._on_nav_change, names='value')
        self.widget: widgets.VBox = widgets.VBox(
            [
                self._header(),
                widgets.HBox(
                    [
                        self._nav,
                        widgets.VBox(
                            [self._alert, self._content],
                            layout=widgets.Layout(width='100%'),
                        ),
                    ],
                    layout=widgets.Layout(width='100%'),
                ),
            ],
            layout=widgets.Layout(width='100%'),
        )
        self._render_current()

    def display(self) -> None:
        """
        title: Display the widget in the current notebook output cell.
        """
        display_widget = cast(Callable[[object], None], ipython_display)
        display_widget(self.widget)

    def _header(self) -> widgets.HTML:
        """
        title: Build the notebook UI header.
        returns:
          type: widgets.HTML
          description: Header widget.
        """
        return widgets.HTML(
            value=(
                '<div style="padding:16px 18px;border-radius:12px;'
                'background:linear-gradient(135deg,#0f766e,#1d4ed8);'
                'color:white;margin-bottom:12px">'
                '<h2 style="margin:0">HiperHealth Notebook UI</h2>'
                '<p style="margin:6px 0 0 0">Guided clinical pipeline '
                'workflow for session-backed notebooks.</p></div>'
            )
        )

    def _on_nav_change(self, change: dict[str, object]) -> None:
        """
        title: Re-render the selected navigation page.
        parameters:
          change:
            type: dict[str, object]
        """
        if change.get('name') == 'value':
            self._render_current()

    def _render_current(self) -> None:
        """
        title: Render the currently selected navigation page.
        """
        page = self._nav.value
        renderers = {
            'home': self._render_home,
            'patient': self._render_patient_data,
            'workflow': self._render_workflow,
            'inquiries': self._render_inquiries,
            'skill_apps': self._render_skill_apps,
            'results': self._render_results,
            'events': self._render_events,
            'settings': self._render_settings,
        }
        renderers[page]()

    def _render_home(self) -> None:
        """
        title: Render session creation and loading controls.
        """
        summary = self.controller.summary()
        create_name = widgets.Text(
            description='Name',
            placeholder='case-study.parquet',
            layout=widgets.Layout(width='420px'),
        )
        create_button = widgets.Button(
            description='Create session',
            button_style='success',
            icon='plus',
        )
        load_select = widgets.Dropdown(
            options=self._session_options(),
            description='Session',
            layout=widgets.Layout(width='520px'),
        )
        load_button = widgets.Button(
            description='Load selected',
            button_style='primary',
            icon='folder-open',
        )
        refresh_button = widgets.Button(description='Refresh', icon='refresh')

        def create_session(_: object) -> None:
            """
            title: Create a session from the home page button.
            parameters:
              _:
                type: object
            """
            try:
                session = self.controller.create_session(
                    create_name.value.strip() or None
                )
            except (FileExistsError, OSError, RuntimeError) as exc:
                self._set_alert(str(exc), 'danger')
                return
            self._set_alert(f'Created session: {session.path}', 'success')
            self._render_home()

        def load_session(_: object) -> None:
            """
            title: Load the selected session from the home page button.
            parameters:
              _:
                type: object
            """
            if not load_select.value:
                self._set_alert('No session selected.', 'warning')
                return
            try:
                session = self.controller.load_session(str(load_select.value))
            except (FileNotFoundError, OSError, RuntimeError) as exc:
                self._set_alert(str(exc), 'danger')
                return
            self._set_alert(f'Loaded session: {session.path}', 'success')
            self._render_home()

        create_button.on_click(create_session)
        load_button.on_click(load_session)
        refresh_button.on_click(lambda _: self._render_home())

        self._content.children = (
            self._privacy_banner(),
            self._summary_card(),
            widgets.HTML('<h3>Create or load a parquet session</h3>'),
            widgets.HBox([create_name, create_button]),
            widgets.HBox([load_select, load_button, refresh_button]),
            widgets.HTML(
                '<p><strong>Data directory:</strong> '
                f'{escape(str(summary.data_dir))}</p>'
            ),
            self._workflow_reference(),
        )

    def _render_patient_data(self) -> None:
        """
        title: Render clinical data editing controls.
        """
        data = self._safe_clinical_data()
        editor = widgets.Textarea(
            value=self._format_json(data),
            description='Clinical JSON',
            layout=widgets.Layout(width='100%', height='320px'),
        )
        save_button = widgets.Button(
            description='Save clinical data',
            button_style='success',
            icon='save',
        )
        example_button = widgets.Button(
            description='Load example screening fields',
            icon='flask',
        )

        def save(_: object) -> None:
            """
            title: Save clinical JSON from the patient data editor.
            parameters:
              _:
                type: object
            """
            try:
                fields = self._parse_json_object(editor.value, 'Clinical data')
                self.controller.set_clinical_data(fields)
            except (ValueError, RuntimeError, ValidationError) as exc:
                self._set_alert(str(exc), 'danger')
                return
            self._set_alert('Clinical data saved to the session.', 'success')
            self._render_patient_data()

        def load_example(_: object) -> None:
            """
            title: Load synthetic screening fields into the editor.
            parameters:
              _:
                type: object
            """
            current = self._safe_clinical_data()
            current.update(self._example_screening_fields())
            editor.value = self._format_json(current)
            self._set_alert(
                'Synthetic example fields loaded into the editor. Review and '
                'save them before running screening.',
                'info',
            )

        save_button.on_click(save)
        example_button.on_click(load_example)
        self._content.children = (
            widgets.HTML('<h3>Patient data</h3>'),
            widgets.HTML(
                '<p>Enter de-identified clinical fields only. The active '
                'session parquet file may contain sensitive clinical data.</p>'
            ),
            widgets.HBox([save_button, example_button]),
            editor,
        )

    def _render_workflow(self) -> None:
        """
        title: Render stage workflow controls.
        """
        stage_select = widgets.Dropdown(
            options=_STAGE_LABELS,
            value='diagnosis',
            description='Stage',
        )
        disabled = self._disabled_skills_select()
        check_button = widgets.Button(
            description='Check requirements',
            button_style='info',
            icon='question-circle',
        )
        run_button = widgets.Button(
            description='Run stage',
            button_style='primary',
            icon='play',
        )
        output = widgets.Output(
            layout=widgets.Layout(
                border='1px solid #ddd',
                padding='8px',
                min_height='72px',
            )
        )

        def check_requirements(_: object) -> None:
            """
            title: Check requirements for the selected workflow stage.
            parameters:
              _:
                type: object
            """
            try:
                inquiries = self.controller.check_requirements(
                    str(stage_select.value),
                    disabled_skills=self._selected_disabled(disabled),
                    llm=self._llm,
                    llm_settings=self._current_llm_settings(),
                )
            except Exception as exc:
                self._render_output_error(output, exc)
                return
            with output:
                output.clear_output()
                print(f'Requirements returned: {len(inquiries)}')
                for inquiry in inquiries:
                    print(
                        f'[{inquiry.priority}] {inquiry.field}: '
                        f'{inquiry.label}'
                    )
            self._set_alert('Requirement check completed.', 'success')

        def run_stage(_: object) -> None:
            """
            title: Run the selected workflow stage.
            parameters:
              _:
                type: object
            """
            try:
                session = self.controller.run_stage(
                    str(stage_select.value),
                    disabled_skills=self._selected_disabled(disabled),
                    llm=self._llm,
                    llm_settings=self._current_llm_settings(),
                )
            except Exception as exc:
                self._render_output_error(output, exc)
                return
            with output:
                output.clear_output()
                print(f'Ran stage: {stage_select.value}')
                print(f'Session: {session.path}')
                print(f'Events: {len(session.events)}')
            self._set_alert(
                f'Completed stage: {stage_select.value}',
                'success',
            )
            self._render_workflow()

        check_button.on_click(check_requirements)
        run_button.on_click(run_stage)

        self._content.children = (
            widgets.HTML('<h3>Guided workflow</h3>'),
            self._workflow_status_table(),
            self._workflow_reference(),
            widgets.HTML('<h4>Run a stage</h4>'),
            widgets.VBox(
                [
                    stage_select,
                    disabled,
                    widgets.HBox([check_button, run_button]),
                    output,
                ]
            ),
        )

    def _render_inquiries(self) -> None:
        """
        title: Render pending inquiries and answer controls.
        """
        try:
            inquiries = self.controller.pending_inquiries()
        except RuntimeError as exc:
            self._content.children = (self._empty_state(str(exc)),)
            return

        stage_filter = widgets.Dropdown(
            options=(('All stages', ''), *_STAGE_LABELS),
            value='',
            description='Stage',
        )
        form_box = widgets.VBox()
        controls: dict[str, widgets.Widget] = {}

        def render_form(stage_value: str = '') -> None:
            """
            title: Render pending inquiry controls for a stage filter.
            parameters:
              stage_value:
                type: str
            """
            filtered = [
                inquiry
                for inquiry in inquiries
                if not stage_value or inquiry.stage == stage_value
            ]
            controls.clear()
            if not filtered:
                form_box.children = (
                    self._empty_state('No pending inquiries.'),
                )
                return
            form_box.children = tuple(
                self._inquiry_control(inquiry, controls)
                for inquiry in filtered
            )

        def on_filter(change: dict[str, object]) -> None:
            """
            title: Re-render inquiry controls when the stage filter changes.
            parameters:
              change:
                type: dict[str, object]
            """
            if change.get('name') == 'value':
                render_form(str(stage_filter.value))

        def save_answers(_: object) -> None:
            """
            title: Save inquiry answers entered in the current form.
            parameters:
              _:
                type: object
            """
            answers = self._collect_answers(controls)
            if not answers:
                self._set_alert('No inquiry answers entered.', 'warning')
                return
            try:
                self.controller.provide_answers(answers)
            except RuntimeError as exc:
                self._set_alert(str(exc), 'danger')
                return
            self._set_alert(
                f'Saved {len(answers)} inquiry answer(s).',
                'success',
            )
            self._render_inquiries()

        stage_filter.observe(on_filter, names='value')
        save_button = widgets.Button(
            description='Save answers',
            button_style='success',
            icon='save',
        )
        save_button.on_click(save_answers)
        render_form()

        self._content.children = (
            widgets.HTML('<h3>Inquiries</h3>'),
            widgets.HTML(
                '<p>Answer required and supplementary fields now. Deferred '
                'fields can be answered after future visits or lab '
                'results.</p>'
            ),
            widgets.HBox([stage_filter, save_button]),
            form_box,
        )

    def _render_skill_apps(self) -> None:
        """
        title: Render declarative app views contributed by active skills.
        """
        stage_filter = widgets.Dropdown(
            options=(('All stages', ''), *_STAGE_LABELS),
            value='',
            description='Stage',
        )
        phase_filter = widgets.Dropdown(
            options=(
                ('All phases', ''),
                ('Setup', 'setup'),
                ('Requirements', 'requirements'),
                ('Pre-run', 'pre_run'),
                ('Run', 'run'),
                ('Post-run', 'post_run'),
                ('Results', 'results'),
                ('Settings', 'settings'),
            ),
            value='',
            description='Phase',
        )
        views_box = widgets.VBox()

        def render_views() -> None:
            """
            title: Render skill app views matching the current filters.
            """
            stage = str(stage_filter.value) or None
            phase = str(phase_filter.value) or None
            records = self.controller.skill_app_views(
                stage=stage,
                phase=phase,
            )
            if not records:
                views_box.children = (
                    self._empty_state(
                        'No active skill app views match this filter.'
                    ),
                )
                return
            views_box.children = tuple(
                self._render_skill_app_record(record) for record in records
            )

        def on_filter(change: dict[str, object]) -> None:
            """
            title: Re-render skill app views when a filter changes.
            parameters:
              change:
                type: dict[str, object]
            """
            if change.get('name') == 'value':
                render_views()

        stage_filter.observe(on_filter, names='value')
        phase_filter.observe(on_filter, names='value')
        render_views()
        self._content.children = (
            widgets.HTML('<h3>Skill apps</h3>'),
            widgets.HTML(
                '<p>Active skills can declare optional JSON Schema / JSON '
                'Forms-style views in <code>skill.yaml</code>. Values are '
                'stored only through allowlisted HiperHealth actions.</p>'
            ),
            widgets.HBox([stage_filter, phase_filter]),
            views_box,
        )

    def _render_results(self) -> None:
        """
        title: Render accumulated stage results.
        """
        try:
            results = self.controller.results
        except RuntimeError as exc:
            self._content.children = (self._empty_state(str(exc)),)
            return
        if not results:
            self._content.children = (
                self._empty_state('No stage results yet.'),
            )
            return

        children: list[widgets.Widget] = []
        titles: list[str] = []
        for stage in STANDARD_STAGES:
            result = results.get(stage)
            if result is None:
                continue
            children.append(
                widgets.Textarea(
                    value=self._format_json(result),
                    disabled=True,
                    layout=widgets.Layout(width='100%', height='260px'),
                )
            )
            titles.append(stage)
        accordion = widgets.Accordion(children=children)
        for index, title in enumerate(titles):
            accordion.set_title(index, title)

        self._content.children = (
            widgets.HTML('<h3>Results</h3>'),
            widgets.HTML(
                '<p>Results are validated pipeline outputs persisted in the '
                'session event log. Treat exports as sensitive clinical '
                'data.</p>'
            ),
            accordion,
        )

    def _render_events(self) -> None:
        """
        title: Render session events and selected event details.
        """
        try:
            events = self.controller.events
        except RuntimeError as exc:
            self._content.children = (self._empty_state(str(exc)),)
            return
        if not events:
            self._content.children = (self._empty_state('No session events.'),)
            return

        recent = events[-100:]
        details = widgets.Textarea(
            value='',
            disabled=True,
            layout=widgets.Layout(width='100%', height='220px'),
        )
        event_select = widgets.Dropdown(
            options=[
                (
                    f'#{event["event_id"]} {event["event_type"]}',
                    int(event['event_id']),
                )
                for event in recent
            ],
            description='Event',
            layout=widgets.Layout(width='420px'),
        )
        show_details = widgets.Button(
            description='Show selected event data',
            icon='eye',
        )

        def show_event(_: object) -> None:
            """
            title: Show raw data for the selected session event.
            parameters:
              _:
                type: object
            """
            selected = int(event_select.value)
            for event in events:
                if int(event['event_id']) == selected:
                    details.value = self._format_json(event)
                    return

        show_details.on_click(show_event)
        self._content.children = (
            widgets.HTML('<h3>Events and audit</h3>'),
            widgets.HTML(
                '<p>The table omits raw event data by default. Use the detail '
                'viewer intentionally because event payloads may contain '
                'sensitive clinical data.</p>'
            ),
            widgets.HTML(self._event_table(recent)),
            widgets.HBox([event_select, show_details]),
            details,
        )

    def _render_settings(self) -> None:
        """
        title: Render skills and runtime settings.
        """
        skill_rows = ''.join(
            '<tr><td>{name}</td><td>{version}</td><td>{stages}</td></tr>'.format(
                name=escape(skill.metadata.name),
                version=escape(skill.metadata.version),
                stages=escape(', '.join(skill.metadata.stages)),
            )
            for skill in self.controller.runner.skills
        )
        if not skill_rows:
            skill_rows = '<tr><td colspan="3">No active skills.</td></tr>'

        self._content.children = (
            widgets.HTML('<h3>Skills and settings</h3>'),
            widgets.HTML(
                '<table style="width:100%;border-collapse:collapse">'
                '<thead><tr><th align="left">Skill</th>'
                '<th align="left">Version</th>'
                '<th align="left">Stages</th></tr></thead>'
                f'<tbody>{skill_rows}</tbody></table>'
            ),
            widgets.HTML('<h4>LLM runtime settings</h4>'),
            widgets.HTML(
                '<p>Settings entered here are used only for stage runs from '
                'this widget instance. API keys are not written to the '
                'notebook data directory by the UI.</p>'
            ),
            widgets.VBox(
                [
                    self._provider,
                    self._model,
                    self._api_key,
                    self._persist_raw,
                ]
            ),
            widgets.HTML(
                '<h4>Custom skills</h4><p>Channel skills installed through '
                'the existing registry can be used by passing a prepared '
                '<code>StageRunner</code> to '
                '<code>ui.show(runner=...)</code>. '
                'Dedicated install controls can be added after the base '
                'workflow stabilizes.</p>'
            ),
        )

    def _render_skill_app_record(
        self,
        record: SkillAppViewRecord,
    ) -> widgets.Widget:
        """
        title: Render one skill app view card.
        parameters:
          record:
            type: SkillAppViewRecord
        returns:
          type: widgets.Widget
          description: Skill app view card widget.
        """
        form = SkillAppForm(
            data_schema=record.view.data_schema,
            ui_schema=record.view.ui_schema,
            initial_values=self._skill_app_initial_values(record),
        )
        output = widgets.Output(
            layout=widgets.Layout(
                border='1px solid #ddd',
                padding='8px',
                min_height='48px',
            )
        )
        actions = record.view.actions or []
        buttons: list[widgets.Button] = []
        for action in actions:
            button = widgets.Button(
                description=action.label,
                button_style=self._button_style(action.style),
            )
            button.on_click(
                self._skill_app_action_handler(
                    record,
                    form,
                    action.type,
                    output,
                )
            )
            buttons.append(button)
        if not buttons:
            button = widgets.Button(
                description='Save skill UI data',
                button_style='primary',
            )
            button.on_click(
                self._skill_app_action_handler(
                    record,
                    form,
                    'session.provide_skill_ui_data',
                    output,
                )
            )
            buttons.append(button)

        return widgets.VBox(
            [
                widgets.HTML(
                    '<div style="border-top:1px solid #d0d7de;'
                    'padding-top:12px;margin-top:12px">'
                    f'<h4>{escape(record.view.title)}</h4>'
                    f'<p><strong>Skill:</strong> '
                    f'{escape(record.skill_id)} · '
                    f'<strong>Stage:</strong> '
                    f'{escape(record.view.stage)} · '
                    f'<strong>Phase:</strong> '
                    f'{escape(record.view.phase)}</p></div>'
                ),
                form.widget,
                widgets.HBox(buttons),
                output,
            ]
        )

    def _skill_app_action_handler(
        self,
        record: SkillAppViewRecord,
        form: SkillAppForm,
        action_type: str,
        output: widgets.Output,
    ) -> Callable[[object], None]:
        """
        title: Build an action callback for a skill app form.
        parameters:
          record:
            type: SkillAppViewRecord
          form:
            type: SkillAppForm
          action_type:
            type: str
          output:
            type: widgets.Output
            description: Output widget.
        returns:
          type: Callable[[object], None]
        """

        def callback(_: object) -> None:
            """
            title: Execute the configured skill app action callback.
            parameters:
              _:
                type: object
            """
            try:
                self._dispatch_skill_app_action(record, form, action_type)
            except Exception as exc:
                self._render_output_error(output, exc)
                return
            with output:
                output.clear_output()
                print(f'Completed action: {action_type}')
            self._set_alert(f'Completed skill app action: {action_type}')

        return callback

    def _dispatch_skill_app_action(
        self,
        record: SkillAppViewRecord,
        form: SkillAppForm,
        action_type: str,
    ) -> None:
        """
        title: Execute an allowlisted skill app action.
        parameters:
          record:
            type: SkillAppViewRecord
          form:
            type: SkillAppForm
          action_type:
            type: str
        """
        grouped = form.binding_values()
        values = form.values()
        if action_type == 'session.set_clinical_data':
            self.controller.set_clinical_data(
                grouped.get('session.clinical_data', values)
            )
            return
        if action_type == 'session.provide_answers':
            self.controller.provide_answers(
                grouped.get('session.answers', values)
            )
            return
        if action_type == 'session.provide_skill_ui_data':
            self.controller.provide_skill_ui_data(
                record.skill_id,
                record.view.id,
                self._skill_ui_values(grouped, values),
                stage=record.view.stage,
            )
            return
        if action_type == 'runner.check_requirements':
            self._save_skill_ui_before_runner_action(record, grouped, values)
            self.controller.check_requirements(
                record.view.stage,
                llm=self._llm,
                llm_settings=self._current_llm_settings(),
            )
            return
        if action_type == 'runner.run_session':
            self._save_skill_ui_before_runner_action(record, grouped, values)
            self.controller.run_stage(
                record.view.stage,
                llm=self._llm,
                llm_settings=self._current_llm_settings(),
            )
            return
        msg = f'Unsupported skill app action: {action_type}.'
        raise ValueError(msg)

    def _save_skill_ui_before_runner_action(
        self,
        record: SkillAppViewRecord,
        grouped: dict[str, dict[str, Any]],
        values: dict[str, Any],
    ) -> None:
        """
        title: Persist skill UI values before runner-backed actions.
        parameters:
          record:
            type: SkillAppViewRecord
          grouped:
            type: dict[str, dict[str, Any]]
          values:
            type: dict[str, Any]
        """
        skill_values = self._skill_ui_values(grouped, values)
        if not skill_values:
            return
        self.controller.provide_skill_ui_data(
            record.skill_id,
            record.view.id,
            skill_values,
            stage=record.view.stage,
        )

    @staticmethod
    def _skill_ui_values(
        grouped: dict[str, dict[str, Any]],
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """
        title: Return values bound to skill UI targets.
        parameters:
          grouped:
            type: dict[str, dict[str, Any]]
          values:
            type: dict[str, Any]
        returns:
          type: dict[str, Any]
        """
        return (
            grouped.get('session.skill_ui_data')
            or grouped.get('context.extras.skill_ui')
            or values
        )

    def _skill_app_initial_values(
        self,
        record: SkillAppViewRecord,
    ) -> dict[str, Any]:
        """
        title: Return persisted values for a skill app view.
        parameters:
          record:
            type: SkillAppViewRecord
        returns:
          type: dict[str, Any]
        """
        if self.controller.session is None:
            return {}
        skill_data = self.controller.session.skill_ui_data
        view_data = skill_data.get(record.skill_id, {}).get(record.view.id, {})
        return dict(view_data)

    @staticmethod
    def _button_style(style: str) -> str:
        """
        title: Map declarative action styles to ipywidgets button styles.
        parameters:
          style:
            type: str
        returns:
          type: str
        """
        if style == 'secondary':
            return ''
        return style

    def _summary_card(self) -> widgets.HTML:
        """
        title: Build a compact active-session summary card.
        returns:
          type: widgets.HTML
          description: Summary widget.
        """
        summary = self.controller.summary()
        session_path = (
            str(summary.session_path) if summary.session_path else '—'
        )
        completed = ', '.join(summary.stages_completed) or '—'
        active = escape(session_path)
        language = escape(summary.language)
        completed_label = escape(completed)
        return widgets.HTML(
            value=(
                '<div style="border:1px solid #d0d7de;border-radius:10px;'
                'padding:12px;margin:8px 0">'
                '<h3 style="margin-top:0">Session summary</h3>'
                f'<p><strong>Active session:</strong> {active}</p>'
                f'<p><strong>Language:</strong> {language}</p>'
                f'<p><strong>Completed stages:</strong> {completed_label}</p>'
                f'<p><strong>Pending inquiries:</strong> '
                f'{summary.pending_inquiries}</p>'
                f'<p><strong>Events:</strong> {summary.event_count}</p>'
                '</div>'
            )
        )

    def _privacy_banner(self) -> widgets.HTML:
        """
        title: Build the privacy reminder banner.
        returns:
          type: widgets.HTML
          description: Privacy banner widget.
        """
        return widgets.HTML(
            value=(
                '<div style="background:#fff3cd;color:#664d03;padding:12px;'
                'border-radius:10px;margin-bottom:10px">'
                '<strong>Privacy reminder:</strong> use synthetic or '
                'de-identified data in examples. Session parquet files may '
                'contain sensitive clinical artifacts; protect them like '
                'clinical records.</div>'
            )
        )

    def _workflow_reference(self) -> widgets.HTML:
        """
        title: Build the docs/example.qmd guided workflow overview.
        returns:
          type: widgets.HTML
          description: Workflow reference widget.
        """
        items = ''.join(
            '<li><strong>{title}</strong><br><span>{description}</span></li>'.format(
                title=escape(step.title),
                description=escape(step.description),
            )
            for step in WORKFLOW_STEPS
        )
        return widgets.HTML(
            value=(
                '<details open><summary><strong>Guided workflow based on '
                'docs/example.qmd</strong></summary>'
                f'<ol>{items}</ol></details>'
            )
        )

    def _workflow_status_table(self) -> widgets.HTML:
        """
        title: Build a stage status table.
        returns:
          type: widgets.HTML
          description: Stage status widget.
        """
        try:
            statuses = self.controller.workflow_statuses()
        except RuntimeError:
            statuses = {stage: 'not_started' for stage in STANDARD_STAGES}
        rows = ''.join(
            '<tr><td>{stage}</td><td>{status}</td><td>{runs}</td></tr>'.format(
                stage=escape(stage),
                status=escape(status.replace('_', ' ')),
                runs=self._safe_stage_run_count(stage),
            )
            for stage, status in statuses.items()
        )
        return widgets.HTML(
            value=(
                '<table style="width:100%;border-collapse:collapse;'
                'margin-bottom:12px">'
                '<thead><tr><th align="left">Stage</th>'
                '<th align="left">Status</th>'
                '<th align="left">Runs</th></tr></thead>'
                f'<tbody>{rows}</tbody></table>'
            )
        )

    def _inquiry_control(
        self,
        inquiry: Inquiry,
        controls: dict[str, widgets.Widget],
    ) -> widgets.Widget:
        """
        title: Build one inquiry answer control.
        parameters:
          inquiry:
            type: Inquiry
          controls:
            type: dict[str, widgets.Widget]
            description: Mutable field-to-widget map.
        returns:
          type: widgets.Widget
          description: Inquiry widget.
        """
        if inquiry.choices:
            control = widgets.Dropdown(
                options=('', *inquiry.choices),
                description='Answer',
                layout=widgets.Layout(width='100%'),
            )
        elif inquiry.input_type in {'textarea', 'multiline', 'json'}:
            control = widgets.Textarea(
                description='Answer',
                layout=widgets.Layout(width='100%', height='90px'),
            )
        else:
            control = widgets.Text(
                description='Answer',
                layout=widgets.Layout(width='100%'),
            )
        controls[inquiry.field] = control
        priority = escape(inquiry.priority)
        description = escape(inquiry.description or '')
        return widgets.VBox(
            [
                widgets.HTML(
                    '<div style="border-top:1px solid #ddd;padding-top:8px">'
                    f'<strong>{escape(inquiry.label)}</strong> '
                    f'<code>{escape(inquiry.field)}</code> '
                    f'<span>({priority}, {escape(inquiry.stage)})</span><br>'
                    f'<small>{description}</small></div>'
                ),
                control,
            ]
        )

    def _collect_answers(
        self,
        controls: dict[str, widgets.Widget],
    ) -> dict[str, Any]:
        """
        title: Collect non-empty inquiry answers from form controls.
        parameters:
          controls:
            type: dict[str, widgets.Widget]
            description: Field-to-widget map.
        returns:
          type: dict[str, Any]
        """
        answers: dict[str, Any] = {}
        for field, control in controls.items():
            value = getattr(control, 'value', '')
            if value in ('', None):
                continue
            answers[field] = value
        return answers

    def _disabled_skills_select(self) -> widgets.SelectMultiple:
        """
        title: Build the temporary disabled-skills selector.
        returns:
          type: widgets.SelectMultiple
          description: SelectMultiple widget.
        """
        options = self.controller.active_skill_names()
        return widgets.SelectMultiple(
            options=options,
            description='Disable',
            rows=max(3, min(6, len(options) or 3)),
            layout=widgets.Layout(width='420px'),
        )

    @staticmethod
    def _selected_disabled(widget: widgets.SelectMultiple) -> Collection[str]:
        """
        title: Return selected disabled skills.
        parameters:
          widget:
            type: widgets.SelectMultiple
            description: SelectMultiple widget.
        returns:
          type: Collection[str]
        """
        return tuple(str(value) for value in widget.value)

    def _current_llm_settings(self) -> LLMSettings | None:
        """
        title: Build optional LLM settings from runtime controls.
        returns:
          type: LLMSettings | None
        """
        provider = self._provider.value.strip()
        model = self._model.value.strip()
        api_key = self._api_key.value.strip()
        if not provider and not model and not api_key:
            return self._llm_settings
        base = self._llm_settings or LLMSettings()
        return base.with_overrides(
            provider=provider or None,
            model=model or None,
            api_key=api_key if api_key else None,
            persist_raw=bool(self._persist_raw.value),
        )

    def _session_options(self) -> tuple[tuple[str, str], ...]:
        """
        title: Return dropdown options for known sessions.
        returns:
          type: tuple[tuple[str, str], Ellipsis]
        """
        sessions = self.controller.list_sessions()
        if not sessions:
            return (('No sessions found', ''),)
        return tuple((path.name, str(path)) for path in sessions)

    def _safe_clinical_data(self) -> dict[str, Any]:
        """
        title: Return clinical data or an empty mapping if no session exists.
        returns:
          type: dict[str, Any]
        """
        try:
            return self.controller.clinical_data
        except RuntimeError:
            return {}

    def _safe_stage_run_count(self, stage: str) -> int:
        """
        title: Return a stage run count without requiring an active session.
        parameters:
          stage:
            type: str
        returns:
          type: int
        """
        try:
            return self.controller.stage_run_count(stage)
        except RuntimeError:
            return 0

    @staticmethod
    def _event_table(events: list[dict[str, Any]]) -> str:
        """
        title: Build an HTML table for event metadata.
        parameters:
          events:
            type: list[dict[str, Any]]
        returns:
          type: str
        """
        rows = ''.join(
            '<tr><td>{event_id}</td><td>{timestamp}</td><td>{event_type}</td>'
            '<td>{stage}</td><td>{skill}</td></tr>'.format(
                event_id=escape(str(event.get('event_id', ''))),
                timestamp=escape(str(event.get('timestamp', ''))),
                event_type=escape(str(event.get('event_type', ''))),
                stage=escape(str(event.get('stage') or '')),
                skill=escape(str(event.get('skill_name') or '')),
            )
            for event in events
        )
        return (
            '<table style="width:100%;border-collapse:collapse">'
            '<thead><tr><th align="left">ID</th>'
            '<th align="left">Timestamp</th>'
            '<th align="left">Type</th>'
            '<th align="left">Stage</th>'
            '<th align="left">Skill</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>'
        )

    @staticmethod
    def _example_screening_fields() -> dict[str, Any]:
        """
        title: Return synthetic screening fields from the example workflow.
        returns:
          type: dict[str, Any]
        """
        return {
            'symptoms': (
                'Patient reports chronic bloating, abdominal discomfort after '
                'meals, fatigue, brain fog, and intermittent diarrhea for the '
                'past 6 months. Symptoms worsen with gluten and dairy intake.'
            ),
            'age': 38,
            'biological_sex': 'female',
            'medical_history': (
                'Irritable bowel syndrome diagnosed 3 years ago'
            ),
            'medications': 'Omeprazole 20mg daily',
            'allergies': 'None known',
        }

    @staticmethod
    def _empty_state(message: str) -> widgets.HTML:
        """
        title: Build an empty-state message.
        parameters:
          message:
            type: str
        returns:
          type: widgets.HTML
          description: Empty-state widget.
        """
        return widgets.HTML(
            value=(
                '<div style="padding:16px;border:1px dashed #d0d7de;'
                'border-radius:10px;color:#57606a">'
                f'{escape(message)}</div>'
            )
        )

    def _set_alert(self, message: str, kind: str = 'info') -> None:
        """
        title: Show a status alert.
        parameters:
          message:
            type: str
          kind:
            type: str
        """
        background, color = _ALERT_COLORS.get(kind, _ALERT_COLORS['info'])
        self._alert.value = (
            '<div style="padding:10px;border-radius:8px;margin-bottom:8px;'
            f'background:{background};color:{color}">{escape(message)}</div>'
        )

    @staticmethod
    def _render_output_error(output: widgets.Output, exc: Exception) -> None:
        """
        title: Render an actionable error in a widget output area.
        parameters:
          output:
            type: widgets.Output
            description: Output widget.
          exc:
            type: Exception
        """
        with output:
            output.clear_output()
            print(f'Error: {exc}')

    @staticmethod
    def _parse_json_object(text: str, label: str) -> dict[str, Any]:
        """
        title: Parse a JSON object from user-provided text.
        parameters:
          text:
            type: str
          label:
            type: str
        returns:
          type: dict[str, Any]
        """
        try:
            value = json.loads(text or '{}')
        except json.JSONDecodeError as exc:
            msg = f'{label} must be valid JSON: {exc.msg}'
            raise ValueError(msg) from exc
        if not isinstance(value, dict):
            msg = f'{label} must be a JSON object.'
            raise ValueError(msg)
        return value

    @staticmethod
    def _format_json(value: Any) -> str:
        """
        title: Format a value as indented JSON.
        parameters:
          value:
            type: Any
        returns:
          type: str
        """
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
