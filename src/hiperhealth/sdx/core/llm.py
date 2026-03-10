"""LLM provider interface."""

from __future__ import annotations

import abc

from typing import Any, Optional

from pydantic import BaseModel

from .types import LLMResponse, Prompt


class LLMProvider(BaseModel, abc.ABC):
    """Abstract LLM provider."""

    @abc.abstractmethod
    def generate(
        self, _prompt: Prompt, _meta: Optional[dict[str, Any]] = None
    ) -> LLMResponse:
        """Return a completion for a prompt."""
        raise NotImplementedError
