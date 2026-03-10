"""Tests for SDX abstract base classes to ensure coverage."""

from typing import Any, Optional

from hiperhealth.sdx.core.llm import LLMProvider
from hiperhealth.sdx.core.plugins import Plugin
from hiperhealth.sdx.core.steps import Step
from hiperhealth.sdx.core.types import (
    Context,
    LLMResponse,
    Prompt,
    RequestSpec,
    StepResult,
)


class MockLLM(LLMProvider):
    """Concrete LLM for testing."""

    def generate(
        self, _prompt: Prompt, _meta: Optional[dict[str, Any]] = None
    ) -> LLMResponse:
        """Return a mock LLM response."""
        return LLMResponse(raw='test output')


class MockPlugin(Plugin):
    """Concrete Plugin for testing hooks."""

    pass


class MockStep(Step):
    """Concrete Step for testing."""

    def start(self, _ctx: Context) -> StepResult:
        """Emit mock start result."""
        return StepResult(status='more')

    def consume(self, _ctx: Context, _payload: dict[str, Any]) -> StepResult:
        """Emit mock consume result."""
        return StepResult(status='done')


def test_abstract_llm_coverage():
    """Test LLMProvider interface coverage."""
    llm = MockLLM()
    prompt = Prompt(system='sys', user='user')
    res = llm.generate(prompt)
    assert res.raw == 'test output'


def test_abstract_plugin_coverage():
    """Test Plugin hook coverage."""
    plugin = MockPlugin()
    ctx = Context(session_id='test')
    spec = RequestSpec(message='msg')
    res = LLMResponse(raw='raw')
    prompt = Prompt(system='s', user='u')
    result = StepResult()

    # Call all hooks to ensure coverage
    plugin.before_step_start(ctx, 'step1')
    assert plugin.before_request_emit(ctx, spec, 'step1') == spec
    assert plugin.validate_input(ctx, {}, 'step1') == []
    plugin.before_llm(ctx, 'step1')
    assert plugin.modify_prompt(ctx, prompt, 'step1') == prompt
    assert plugin.after_llm(ctx, res, 'step1') == res
    assert plugin.after_step_finish(ctx, result) == result
    assert plugin.provide_steps() == []


def test_abstract_step_coverage():
    """Test Step interface coverage."""
    step = MockStep()
    ctx = Context(session_id='test')
    res = LLMResponse(raw='raw')

    assert step.start(ctx).status == 'more'
    assert step.consume(ctx, {}).status == 'done'
    assert step.build_prompt(ctx) is None
    step.apply_llm(ctx, res)  # Should do nothing but provide coverage
