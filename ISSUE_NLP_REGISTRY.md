**Title**: Pluggable NLP Pipeline Registry with Lazy Model Loading

**Motivation**
- Reduce memory and startup cost by making heavy NLP models lazy-loadable and pluggable.
- Provide a consistent plugin API so contributors can add new pipelines without touching core code.

**Design summary**
- Add a `BasePipeline` ABC and a central `registry` with a `@register_pipeline(name)` decorator.
- Provide a `LazyPipelineProxy` that defers heavy initialization until first `process()` call.
- Add a lightweight `mock_pipeline` adapter and unit tests demonstrating thread-safe lazy init.

**Deliverables (phase 1 — this issue)**
- `src/hiperhealth/nlp/base.py` — `BasePipeline` ABC.
- `src/hiperhealth/nlp/registry.py` — registry, decorator, `get_pipeline`, `LazyPipelineProxy`.
- `src/hiperhealth/nlp/pipelines/mock_pipeline.py` — example adapter with no heavy deps.
- `tests/test_nlp_registry.py` — unit tests for lazy init and thread-safety.
- Documentation snippet and usage example in `DEVELOPMENT.md` or `README`.

**How to test (quick)**
- `python -m pytest tests/test_nlp_registry.py -q`
- Example usage:
  ```py
  from hiperhealth.nlp import get_pipeline, register_pipeline
  p = get_pipeline('mock')
  print(p.process('hello world'))
  ```

**Acceptance criteria**
- Registry can register and list pipelines.
- `LazyPipelineProxy` initializes underlying pipeline exactly once on first `process()` call (thread-safe).
- Tests pass on CI (mock pipelines used to avoid heavy deps).

**Labels**: `enhancement`, `architecture`, `needs-review`

**Estimated effort**: 5–10 days (phase 1: skeleton + tests: 1–2 days)
