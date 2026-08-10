"""
title: Unit tests for pipeline data models.
"""

from __future__ import annotations

import pytest

from hiperhealth.pipeline.models import (
    AgentStep,
    AgentStepResult,
    ExecutionStep,
    PromptFragment,
    SkillResult,
)
from pydantic import ValidationError


def test_prompt_fragment_defaults_and_validation() -> None:
    """
    title: Test PromptFragment initialization and defaults.
    """
    # Valid initialization with defaults
    fragment = PromptFragment(
        stage='diagnosis',
        skill_name='test_skill',
        title='Test Title',
        content='Test content',
    )
    assert fragment.priority == 100
    assert fragment.include_in_final_prompt is True

    # Validation error on missing required fields
    with pytest.raises(ValidationError):
        PromptFragment(
            stage='diagnosis', skill_name='test_skill', title='Test Title'
        )


def test_prompt_fragment_serialization_roundtrip() -> None:
    """
    title: Test PromptFragment JSON serialization round trip.
    """
    fragment = PromptFragment(
        stage='exam',
        skill_name='exam_skill',
        title='Findings',
        content='Nothing unusual',
        priority=50,
        include_in_final_prompt=False,
    )

    json_data = fragment.model_dump_json()
    restored = PromptFragment.model_validate_json(json_data)

    assert restored == fragment


def test_skill_result_defaults_and_validation() -> None:
    """
    title: Test SkillResult initialization and defaults.
    """
    result = SkillResult(
        stage='diagnosis',
        skill_name='test_skill',
        status='succeeded',
    )
    assert result.summary == ''
    assert result.data == {}
    assert result.prompt_fragment is None

    # Invalid status
    with pytest.raises(ValidationError):
        SkillResult(
            stage='diagnosis',
            skill_name='test_skill',
            status='invalid_status',  # type: ignore
        )


def test_skill_result_serialization_roundtrip() -> None:
    """
    title: Test SkillResult JSON serialization round trip.
    """
    fragment = PromptFragment(
        stage='diagnosis',
        skill_name='test_skill',
        title='Fragment',
        content='Content',
    )
    result = SkillResult(
        stage='diagnosis',
        skill_name='test_skill',
        status='failed',
        summary='Skill failed',
        data={'error_code': 123},
        prompt_fragment=[fragment],
    )

    json_data = result.model_dump_json()
    restored = SkillResult.model_validate_json(json_data)

    assert restored == result


def test_execution_step_defaults_and_validation() -> None:
    """
    title: Test ExecutionStep initialization and defaults.
    """
    step = ExecutionStep(
        run_id='run-123',
        stage='intake',
        skill_name='intake_skill',
        hook='execute',
        input_hash='abc',
        status='started',
    )
    assert step.attempt == 1
    assert step.error_data is None

    # Invalid status
    with pytest.raises(ValidationError):
        ExecutionStep(
            run_id='run-123',
            stage='intake',
            skill_name='intake_skill',
            hook='execute',
            input_hash='abc',
            status='pending',  # type: ignore
        )


def test_execution_step_serialization_roundtrip() -> None:
    """
    title: Test ExecutionStep JSON serialization round trip.
    """
    step = ExecutionStep(
        run_id='run-456',
        stage='treatment',
        skill_name='treatment_skill',
        hook='post',
        attempt=2,
        input_hash='def',
        status='failed',
        error_data={'reason': 'timeout'},
    )

    json_data = step.model_dump_json()
    restored = ExecutionStep.model_validate_json(json_data)

    assert restored == step


def test_agent_step_defaults_and_validation() -> None:
    """
    title: Test AgentStep initialization and defaults.
    """
    step = AgentStep(
        name='planning',
        description='Plan the next moves',
    )
    assert step.name == 'planning'

    with pytest.raises(ValidationError):
        AgentStep(name='planning')  # missing description


def test_agent_step_serialization_roundtrip() -> None:
    """
    title: Test AgentStep JSON serialization round trip.
    """
    step = AgentStep(
        name='planning',
        description='Plan the next moves',
    )

    json_data = step.model_dump_json()
    restored = AgentStep.model_validate_json(json_data)

    assert restored == step


def test_agent_step_result_defaults_and_validation() -> None:
    """
    title: Test AgentStepResult initialization and defaults.
    """
    step_result = AgentStepResult(
        name='sub-task',
        status='skipped',
    )
    assert step_result.data == {}

    # Invalid status
    with pytest.raises(ValidationError):
        AgentStepResult(
            name='sub-task',
            status='done',  # type: ignore
        )


def test_agent_step_result_serialization_roundtrip() -> None:
    """
    title: Test AgentStepResult JSON serialization round trip.
    """
    step_result = AgentStepResult(
        name='analysis',
        status='succeeded',
        data={'score': 0.95},
    )

    json_data = step_result.model_dump_json()
    restored = AgentStepResult.model_validate_json(json_data)

    assert restored == step_result
