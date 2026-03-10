"""Base class for pipeline steps."""

from __future__ import annotations

import abc

from typing import Any, ClassVar, Optional

from pydantic import BaseModel

from .types import Context, LLMResponse, Prompt, StepResult


class Step(BaseModel, abc.ABC):
    """A pipeline step."""

    id: ClassVar[str] = 'step'
    requires: list[str] = []

    @abc.abstractmethod
    def start(self, _ctx: Context) -> StepResult:
        """Emit the first request or route."""
        raise NotImplementedError

    @abc.abstractmethod
    def consume(self, _ctx: Context, _payload: dict[str, Any]) -> StepResult:
        """Validate and persist input, then route or request more."""
        raise NotImplementedError

    def build_prompt(self, _ctx: Context) -> Optional[Prompt]:
        """Return a prompt when this step needs model inference."""
        return None

    def apply_llm(self, _ctx: Context, _out: LLMResponse) -> None:
        """Update context based on LLM response."""
        return None
