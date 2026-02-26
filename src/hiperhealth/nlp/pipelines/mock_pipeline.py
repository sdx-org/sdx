from typing import Any

from ..base import BasePipeline


class MockPipeline(BasePipeline):
    """A tiny example pipeline used for testing and CI.

    This avoids bringing heavy NLP dependencies into CI while demonstrating
    the pipeline interface and lazy-loading behavior.
    """

    def __init__(self) -> None:
        super().__init__("mock")
        self._inits = 0

    def initialize(self) -> None:
        self._inits += 1
        self.initialized = True

    def process(self, text: str) -> Any:
        # trivial processing: split on whitespace and return tokens
        return [t for t in text.split() if t]

    def shutdown(self) -> None:
        self.initialized = False

    def health_check(self) -> bool:
        return True


# Register the mock pipeline so it's available via the registry by default.
from ..registry import register_pipeline


@register_pipeline("mock")
def _mock_factory() -> MockPipeline:
    return MockPipeline()
