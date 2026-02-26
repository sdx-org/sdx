from abc import ABC, abstractmethod
from typing import Any


class BasePipeline(ABC):
    """Abstract base for NLP pipelines used in the project.

    Implementations should be light-weight wrappers around heavy models
    and support lazy initialization via the registry's proxy.
    """

    def __init__(self, name: str):
        self.name = name
        self.initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Prepare heavy resources (models, tokenizers)."""

    @abstractmethod
    def process(self, text: str) -> Any:
        """Process input text and return structured output."""

    def shutdown(self) -> None:
        """Optional: release resources."""
        return None

    def health_check(self) -> bool:
        """Optional health check; return True when pipeline is usable."""
        return True
