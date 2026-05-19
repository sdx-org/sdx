"""
title: Render JSON Forms-style skill app schemas with ipywidgets.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from html import escape
from typing import Any

import ipywidgets as widgets

_DEFAULT_BINDING_TARGET = 'session.skill_ui_data'


@dataclass(frozen=True)
class ControlBinding:
    """
    title: Binding metadata for one rendered control.
    attributes:
      key:
        type: str
      target:
        type: str
      field:
        type: str
    """

    key: str
    target: str
    field: str


class SkillAppForm:
    """
    title: Rendered skill app form with value collection helpers.
    attributes:
      _data_schema:
        type: dict[str, Any]
      _ui_schema:
        type: dict[str, Any]
      _initial_values:
        type: dict[str, Any]
      _controls:
        type: dict[str, widgets.Widget]
      _bindings:
        type: dict[str, ControlBinding]
      widget:
        type: widgets.Widget
        description: Root ipywidgets widget.
    """

    _data_schema: dict[str, Any]
    _ui_schema: dict[str, Any]
    _initial_values: dict[str, Any]
    _controls: dict[str, widgets.Widget]
    _bindings: dict[str, ControlBinding]
    widget: widgets.Widget

    def __init__(
        self,
        *,
        data_schema: dict[str, Any],
        ui_schema: dict[str, Any],
        initial_values: dict[str, Any] | None = None,
    ) -> None:
        """
        title: Create a form from data and UI schemas.
        parameters:
          data_schema:
            type: dict[str, Any]
          ui_schema:
            type: dict[str, Any]
          initial_values:
            type: dict[str, Any] | None
        """
        self._data_schema: dict[str, Any] = data_schema
        self._ui_schema: dict[str, Any] = ui_schema
        self._initial_values: dict[str, Any] = initial_values or {}
        self._controls: dict[str, widgets.Widget] = {}
        self._bindings: dict[str, ControlBinding] = {}
        self.widget: widgets.Widget = self._render_element(ui_schema)

    def values(self) -> dict[str, Any]:
        """
        title: Return all non-empty form values keyed by schema property.
        returns:
          type: dict[str, Any]
        """
        values: dict[str, Any] = {}
        for key, control in self._controls.items():
            value = self._control_value(key, control)
            if value in ('', None, [], {}):
                continue
            values[key] = value
        self._validate_required(values)
        return values

    def binding_values(self) -> dict[str, dict[str, Any]]:
        """
        title: Return values grouped by HiperHealth binding target.
        returns:
          type: dict[str, dict[str, Any]]
        """
        values = self.values()
        grouped: dict[str, dict[str, Any]] = {}
        for key, value in values.items():
            binding = self._bindings.get(
                key,
                ControlBinding(
                    key=key,
                    target=_DEFAULT_BINDING_TARGET,
                    field=key,
                ),
            )
            grouped.setdefault(binding.target, {})[binding.field] = value
        return grouped

    def _render_element(self, element: dict[str, Any]) -> widgets.Widget:
        """
        title: Render one UI schema element recursively.
        parameters:
          element:
            type: dict[str, Any]
        returns:
          type: widgets.Widget
          description: Widget.
        """
        element_type = element.get('type')
        if element_type == 'Control':
            return self._render_control(element)
        if element_type == 'HorizontalLayout':
            return widgets.HBox(
                self._render_children(element),
                layout=self._layout_from_options(element.get('options')),
            )
        if element_type in {'VerticalLayout', 'Group', 'Category'}:
            children = self._render_children(element)
            label = element.get('label')
            if isinstance(label, str) and label:
                children.insert(0, widgets.HTML(f'<h4>{escape(label)}</h4>'))
            return widgets.VBox(
                children,
                layout=self._layout_from_options(element.get('options')),
            )
        if element_type == 'Categorization':
            return self._render_categorization(element)
        return widgets.HTML(
            f'Unsupported UI element: {escape(str(element_type))}'
        )

    def _render_children(
        self,
        element: dict[str, Any],
    ) -> list[widgets.Widget]:
        """
        title: Render child elements.
        parameters:
          element:
            type: dict[str, Any]
        returns:
          type: list[widgets.Widget]
        """
        children = element.get('elements', [])
        if not isinstance(children, list):
            return []
        return [
            self._render_element(child)
            for child in children
            if isinstance(child, dict)
        ]

    def _render_categorization(
        self,
        element: dict[str, Any],
    ) -> widgets.Widget:
        """
        title: Render category children as tabs.
        parameters:
          element:
            type: dict[str, Any]
        returns:
          type: widgets.Widget
          description: Tab widget.
        """
        children = self._render_children(element)
        tab = widgets.Tab(children=children)
        raw_children = element.get('elements', [])
        if isinstance(raw_children, list):
            for index, child in enumerate(raw_children):
                if not isinstance(child, dict):
                    continue
                label = child.get('label') or child.get('title') or 'Tab'
                tab.set_title(index, str(label))
        return tab

    def _render_control(self, element: dict[str, Any]) -> widgets.Widget:
        """
        title: Render one bound control.
        parameters:
          element:
            type: dict[str, Any]
        returns:
          type: widgets.Widget
          description: Widget.
        """
        scope = str(element.get('scope', ''))
        key = _scope_to_key(scope)
        schema = self._property_schema(key)
        options = _as_dict(element.get('options'))
        label = _label_for_control(key, schema, element)
        readonly = bool(schema.get('readOnly') or options.get('readonly'))
        value = self._initial_values.get(key, schema.get('default'))
        control = self._control_for_schema(
            key=key,
            label=label,
            schema=schema,
            options=options,
            value=value,
            readonly=readonly,
        )
        self._controls[key] = control
        self._bindings[key] = self._binding_for_control(key, options)
        description = schema.get('description')
        if isinstance(description, str) and description:
            description_widget = widgets.HTML(
                f'<small>{escape(description)}</small>'
            )
            return widgets.VBox([control, description_widget])
        return control

    def _control_for_schema(
        self,
        *,
        key: str,
        label: str,
        schema: dict[str, Any],
        options: dict[str, Any],
        value: Any,
        readonly: bool,
    ) -> widgets.Widget:
        """
        title: Build an ipywidgets control for a JSON schema property.
        parameters:
          key:
            type: str
          label:
            type: str
          schema:
            type: dict[str, Any]
          options:
            type: dict[str, Any]
          value:
            type: Any
          readonly:
            type: bool
        returns:
          type: widgets.Widget
          description: Widget.
        """
        layout = self._layout_from_options(options)
        enum_values = schema.get('enum')
        if isinstance(enum_values, list):
            selected = value if value in enum_values else None
            if options.get('format') == 'radio':
                radio_value = (
                    selected
                    if selected is not None
                    else enum_values[0]
                    if enum_values
                    else None
                )
                return widgets.RadioButtons(
                    description=label,
                    options=enum_values,
                    value=radio_value,
                    disabled=readonly,
                    layout=layout,
                )
            return widgets.Dropdown(
                description=label,
                options=[('', None), *[(item, item) for item in enum_values]],
                value=selected,
                disabled=readonly,
                layout=layout,
            )

        schema_type = schema.get('type', 'string')
        if schema_type == 'boolean':
            return widgets.Checkbox(
                description=label,
                value=bool(value) if value is not None else False,
                disabled=readonly,
                layout=layout,
            )
        if schema_type == 'integer':
            return widgets.IntText(
                description=label,
                value=int(value) if value is not None else 0,
                disabled=readonly,
                layout=layout,
            )
        if schema_type == 'number':
            return widgets.FloatText(
                description=label,
                value=float(value) if value is not None else 0.0,
                disabled=readonly,
                layout=layout,
            )
        if schema_type in {'array', 'object'}:
            return widgets.Textarea(
                description=label,
                value='' if value is None else _format_json(value),
                disabled=readonly,
                layout=layout,
            )
        if options.get('multi') or options.get('format') == 'textarea':
            return widgets.Textarea(
                description=label,
                value='' if value is None else str(value),
                disabled=readonly,
                layout=layout,
            )
        return widgets.Text(
            description=label,
            value='' if value is None else str(value),
            disabled=readonly,
            layout=layout,
        )

    def _control_value(self, key: str, control: widgets.Widget) -> Any:
        """
        title: Convert one widget value into a JSON-compatible value.
        parameters:
          key:
            type: str
          control:
            type: widgets.Widget
            description: Widget.
        returns:
          type: Any
        """
        value = getattr(control, 'value', None)
        schema_type = self._property_schema(key).get('type', 'string')
        if schema_type in {'array', 'object'} and isinstance(value, str):
            if value.strip() == '':
                return None
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:
                msg = f'{key} must be valid JSON: {exc.msg}'
                raise ValueError(msg) from exc
        return value

    def _validate_required(self, values: dict[str, Any]) -> None:
        """
        title: Ensure required fields have values.
        parameters:
          values:
            type: dict[str, Any]
        """
        required = self._data_schema.get('required', [])
        if not isinstance(required, list):
            return
        missing = [field for field in required if field not in values]
        if missing:
            joined = ', '.join(str(field) for field in missing)
            msg = f'Missing required skill UI field(s): {joined}.'
            raise ValueError(msg)

    def _property_schema(self, key: str) -> dict[str, Any]:
        """
        title: Return the JSON schema fragment for one property.
        parameters:
          key:
            type: str
        returns:
          type: dict[str, Any]
        """
        properties = self._data_schema.get('properties', {})
        if not isinstance(properties, dict):
            return {}
        value = properties.get(key, {})
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _binding_for_control(
        key: str,
        options: dict[str, Any],
    ) -> ControlBinding:
        """
        title: Resolve HiperHealth binding metadata for one control.
        parameters:
          key:
            type: str
          options:
            type: dict[str, Any]
        returns:
          type: ControlBinding
        """
        binding = options.get('x-hiperhealth-binding')
        if not isinstance(binding, dict):
            return ControlBinding(
                key=key,
                target=_DEFAULT_BINDING_TARGET,
                field=key,
            )
        target = binding.get('target', _DEFAULT_BINDING_TARGET)
        field = binding.get('field', key)
        return ControlBinding(key=key, target=str(target), field=str(field))

    @staticmethod
    def _layout_from_options(options: Any) -> widgets.Layout:
        """
        title: Build widget layout hints from UI schema options.
        parameters:
          options:
            type: Any
        returns:
          type: widgets.Layout
          description: Widget layout.
        """
        values = _as_dict(options)
        width = str(values.get('width', '100%'))
        height = values.get('height')
        kwargs: dict[str, str] = {'width': width}
        if height is not None:
            kwargs['height'] = str(height)
        return widgets.Layout(**kwargs)


def _scope_to_key(scope: str) -> str:
    """
    title: Convert a JSON Forms scope to a top-level property key.
    parameters:
      scope:
        type: str
    returns:
      type: str
    """
    prefix = '#/properties/'
    if not scope.startswith(prefix):
        return scope.rsplit('/', maxsplit=1)[-1]
    return scope[len(prefix) :].split('/')[0]


def _label_for_control(
    key: str,
    schema: dict[str, Any],
    element: dict[str, Any],
) -> str:
    """
    title: Resolve a human-readable control label.
    parameters:
      key:
        type: str
      schema:
        type: dict[str, Any]
      element:
        type: dict[str, Any]
    returns:
      type: str
    """
    label = element.get('label')
    if label is False:
        return ''
    if isinstance(label, str):
        return label
    title = schema.get('title')
    if isinstance(title, str):
        return title
    return key.replace('_', ' ').title()


def _as_dict(value: Any) -> dict[str, Any]:
    """
    title: Return a dictionary or an empty fallback.
    parameters:
      value:
        type: Any
    returns:
      type: dict[str, Any]
    """
    return value if isinstance(value, dict) else {}


def _format_json(value: Any) -> str:
    """
    title: Format a value as JSON for an editor control.
    parameters:
      value:
        type: Any
    returns:
      type: str
    """
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)
