"""Public API for the `hiperhealth.nlp` package.

Exports the primary pipeline base and registry helpers used by callers.
"""

from .base import BasePipeline
from .registry import (
    LazyPipelineProxy,
    get_pipeline,
    list_pipelines,
    register_pipeline,
)

# Keep `__all__` sorted for consistency with import-order checks
__all__ = [
    'BasePipeline',
    'LazyPipelineProxy',
    'get_pipeline',
    'list_pipelines',
    'register_pipeline',
]
