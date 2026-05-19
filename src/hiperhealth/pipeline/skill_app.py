"""
title: Declarative app UI schemas for pipeline skills.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SkillUIPhase = Literal[
    'setup',
    'requirements',
    'pre_run',
    'run',
    'post_run',
    'results',
    'settings',
]

SkillUIActionType = Literal[
    'session.set_clinical_data',
    'session.provide_answers',
    'session.provide_skill_ui_data',
    'runner.check_requirements',
    'runner.run_session',
]

SkillUIActionStyle = Literal['primary', 'secondary', 'success', 'danger']

_ALLOWED_UI_ELEMENT_TYPES = {
    'Control',
    'VerticalLayout',
    'HorizontalLayout',
    'Group',
    'Categorization',
    'Category',
}
_ALLOWED_BINDING_TARGETS = {
    'session.clinical_data',
    'session.answers',
    'session.skill_ui_data',
    'context.extras.skill_ui',
    'run_kwargs',
    'results.stage',
}


class SkillUIAction(BaseModel):
    """
    title: Declarative action button for a skill UI view.
    attributes:
      id:
        type: str
      label:
        type: str
      type:
        type: SkillUIActionType
      style:
        type: SkillUIActionStyle
    """

    model_config = ConfigDict(extra='forbid')

    id: str
    label: str
    type: SkillUIActionType
    style: SkillUIActionStyle = 'secondary'


class SkillUIView(BaseModel):
    """
    title: One stage/phase-specific skill UI view.
    attributes:
      id:
        type: str
      title:
        type: str
      stage:
        type: str
      phase:
        type: SkillUIPhase
      priority:
        type: int
      data_schema:
        type: dict[str, Any]
      ui_schema:
        type: dict[str, Any]
      actions:
        type: list[SkillUIAction]
      global_view:
        type: bool
    """

    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    id: str
    title: str
    stage: str
    phase: SkillUIPhase
    priority: int = 100
    data_schema: dict[str, Any]
    ui_schema: dict[str, Any]
    actions: list[SkillUIAction] = Field(default_factory=list)
    global_view: bool = Field(default=False, alias='global')

    @model_validator(mode='after')
    def _validate_view(self) -> SkillUIView:
        """
        title: Validate schemas and renderer-safe UI elements.
        returns:
          type: SkillUIView
        """
        if self.data_schema.get('type') != 'object':
            msg = 'Skill app data_schema must be a JSON object schema.'
            raise ValueError(msg)
        _validate_ui_element(self.ui_schema)
        return self


class SkillAppManifest(BaseModel):
    """
    title: Optional app UI declaration embedded in a skill manifest.
    attributes:
      api_version:
        type: int
      title:
        type: str
      description:
        type: str
      views:
        type: list[SkillUIView]
    """

    model_config = ConfigDict(extra='forbid')

    api_version: int = 1
    title: str = ''
    description: str = ''
    views: list[SkillUIView] = Field(default_factory=list)

    @model_validator(mode='after')
    def _validate_unique_view_ids(self) -> SkillAppManifest:
        """
        title: Ensure view identifiers are unique within a skill app.
        returns:
          type: SkillAppManifest
        """
        view_ids = [view.id for view in self.views]
        duplicates = sorted(
            {view_id for view_id in view_ids if view_ids.count(view_id) > 1}
        )
        if duplicates:
            joined = ', '.join(duplicates)
            msg = f'Duplicate skill app view ids: {joined}.'
            raise ValueError(msg)
        return self


def _validate_ui_element(element: dict[str, Any]) -> None:
    """
    title: Validate one JSON Forms-style UI schema element recursively.
    parameters:
      element:
        type: dict[str, Any]
    """
    element_type = element.get('type')
    if element_type not in _ALLOWED_UI_ELEMENT_TYPES:
        msg = f'Unsupported skill app UI element type: {element_type!r}.'
        raise ValueError(msg)

    options = element.get('options')
    if options is not None and not isinstance(options, dict):
        msg = 'Skill app UI element options must be an object when provided.'
        raise ValueError(msg)
    if isinstance(options, dict):
        _validate_options(options)

    if element_type == 'Control':
        scope = element.get('scope')
        if not isinstance(scope, str) or not scope.startswith('#/'):
            msg = 'Skill app Control elements require a JSON pointer scope.'
            raise ValueError(msg)
        return

    elements = element.get('elements')
    if not isinstance(elements, list) or not elements:
        msg = f'Skill app {element_type} elements must include children.'
        raise ValueError(msg)
    for child in elements:
        if not isinstance(child, dict):
            msg = 'Skill app UI schema children must be objects.'
            raise ValueError(msg)
        _validate_ui_element(child)


def _validate_options(options: dict[str, Any]) -> None:
    """
    title: Validate HiperHealth-specific renderer options.
    parameters:
      options:
        type: dict[str, Any]
    """
    binding = options.get('x-hiperhealth-binding')
    if binding is None:
        return
    if not isinstance(binding, dict):
        msg = 'x-hiperhealth-binding must be an object.'
        raise ValueError(msg)
    target = binding.get('target')
    if target not in _ALLOWED_BINDING_TARGETS:
        msg = f'Unsupported x-hiperhealth-binding target: {target!r}.'
        raise ValueError(msg)
    field = binding.get('field')
    if field is not None and not isinstance(field, str):
        msg = 'x-hiperhealth-binding.field must be a string when provided.'
        raise ValueError(msg)


__all__ = [
    'SkillAppManifest',
    'SkillUIAction',
    'SkillUIActionStyle',
    'SkillUIActionType',
    'SkillUIPhase',
    'SkillUIView',
]
