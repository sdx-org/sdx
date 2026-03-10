"""Plugin architecture base classes and hooks."""

from __future__ import annotations

import abc

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from .types import Context, LLMResponse, Prompt, RequestSpec, StepResult

if TYPE_CHECKING:
    from .steps import Step


class Plugin(BaseModel, abc.ABC):
    """Plugin base with ordered hooks."""

    name: ClassVar[str] = 'plugin'
    priority: ClassVar[int] = 50

    def before_step_start(self, _ctx: Context, _step_id: str) -> None:
        """Run before a step starts."""
        return None

    def before_request_emit(
        self, _ctx: Context, spec: RequestSpec, _step_id: str
    ) -> RequestSpec:
        """Adjust outgoing request spec."""
        return spec

    def validate_input(
        self, _ctx: Context, _payload: dict[str, Any], _step_id: str
    ) -> list[str]:
        """Validate input; return error messages."""
        return []

    def before_llm(self, _ctx: Context, _step_id: str) -> None:
        """Run before any LLM call."""
        return None

    def modify_prompt(
        self, _ctx: Context, prompt: Prompt, _step_id: str
    ) -> Prompt:
        """Alter LLM prompt."""
        return prompt

    def after_llm(
        self, _ctx: Context, response: LLMResponse, _step_id: str
    ) -> LLMResponse:
        """Run after LLM call."""
        return response

    def after_step_finish(
        self, _ctx: Context, result: StepResult
    ) -> StepResult:
        """Run after a step emits a result."""
        return result

    def provide_steps(self) -> list[Step]:
        """Optionally provide additional steps."""
        return []
