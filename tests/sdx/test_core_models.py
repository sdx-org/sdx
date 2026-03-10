"""Tests for SDX core types."""

from hiperhealth.sdx.core.types import (
    Context,
    FieldSpec,
    RequestSpec,
    StepResult,
)


def test_context_initialization():
    """Verify Context initializes with default values."""
    ctx = Context(session_id='test_session')
    assert ctx.session_id == 'test_session'
    assert ctx.patient == {}
    assert ctx.audit == []
    assert ctx.created_at is not None


def test_request_spec_initialization():
    """Verify RequestSpec correctly handles nested FieldSpecs."""
    field = FieldSpec(id='age', label='Age', kind='number', required=True)
    req = RequestSpec(message='Provide age', fields=[field])
    assert req.message == 'Provide age'
    assert len(req.fields) == 1
    assert req.fields[0].id == 'age'
    assert req.fields[0].kind == 'number'


def test_step_result_initialization():
    """Verify StepResult initializes with routing information."""
    result = StepResult(status='route', next_step='diagnosis')
    assert result.status == 'route'
    assert result.next_step == 'diagnosis'
    assert result.errors == []
