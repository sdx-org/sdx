# Pluggable NLP Pipeline Registry

This package provides a small, dependency-free skeleton for registering and
lazily loading NLP pipelines.

## Quick start

Register a pipeline (example):

```py
from hiperhealth.nlp import register_pipeline, get_pipeline

@register_pipeline("example")
def make_example():
    from hiperhealth.nlp.pipelines.mock_pipeline import MockPipeline
    return MockPipeline()

p = get_pipeline("example")
print(p.process("hello world"))
```

## Notes

- `LazyPipelineProxy` defers heavy initialization until the first `process()`
  call.
- Adapters for heavy frameworks (spaCy, presidio, transformers) should be
  implemented in follow-up PRs and can be kept behind optional dependencies.
