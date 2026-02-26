"""Mock pipeline used for tests and CI.

Provides a tiny, dependency-free pipeline implementation registered as
"mock" so tests can exercise the registry without heavy NLP deps.
"""

from typing import Any

from ..base import BasePipeline
from ..registry import register_pipeline


class MockPipeline(BasePipeline):
    """A tiny example pipeline used for testing and CI.

    This avoids bringing heavy NLP dependencies into CI while demonstrating
    the pipeline interface and lazy-loading behavior.
    """

    def __init__(self) -> None:
        super().__init__('mock')
        self._inits = 0

    def initialize(self) -> None:
        """Initialize the pipeline (increment init counter)."""
        self._inits += 1
        self.initialized = True

    def process(self, text: str) -> Any:
        """Return whitespace-split tokens for the given text."""
        return [t for t in text.split() if t]

    def shutdown(self) -> None:
        """Shutdown / release state for the mock pipeline."""
        self.initialized = False

    def health_check(self) -> bool:
        """Return True (mock pipeline is always healthy)."""
        return True


@register_pipeline('mock')
def _mock_factory() -> MockPipeline:
    """Factory returning a new `MockPipeline` instance."""
    return MockPipeline()
