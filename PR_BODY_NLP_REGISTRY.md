**Summary** Introduce a pluggable NLP pipeline registry with lazy model loading.
This PR implements a small, dependency-free skeleton that provides:

- `BasePipeline` abstract base class
- `register_pipeline` decorator and `get_pipeline()` API
- `LazyPipelineProxy` for thread-safe lazy initialization
- example `mock_pipeline` adapter and unit tests

**Files added**

- `src/hiperhealth/nlp/base.py`
- `src/hiperhealth/nlp/registry.py`
- `src/hiperhealth/nlp/pipelines/mock_pipeline.py`
- `tests/test_nlp_registry.py`
- `ISSUE_NLP_REGISTRY.md` (issue draft)

**Why**

- Enables contributors to register new NLP pipelines without changing core code.
- Lazy loading reduces memory and startup cost when multiple heavy pipelines are
  registered.

**How to test (local)**

1. Activate venv
2. Run unit tests for the registry:
   ```bash
   python -m pytest tests/test_nlp_registry.py -q
   ```
3. Quick smoke (Python REPL):

   ````py
   from hiperhealth.nlp import register_pipeline, get_pipeline
   # use mock pipeline already registered by tests or register a new one
   p = get_pipeline('test_lazy')
   print(p.process('hello'))
   **Summary**
   Introduce a pluggable NLP pipeline registry with lazy model loading. This PR implements a small, dependency-free skeleton that provides:

   - `BasePipeline` abstract base class
   - `register_pipeline` decorator and `get_pipeline()` API
   - `LazyPipelineProxy` for thread-safe lazy initialization
   - example `mock_pipeline` adapter and unit tests

   **Files added**
   - `src/hiperhealth/nlp/base.py`
   - `src/hiperhealth/nlp/registry.py`
   - `src/hiperhealth/nlp/pipelines/mock_pipeline.py`
   - `tests/test_nlp_registry.py`
   - `ISSUE_NLP_REGISTRY.md` (issue draft)

   **Why**
   - Enables contributors to register new NLP pipelines without changing core code.
   - Lazy loading reduces memory and startup cost when multiple heavy pipelines are registered.

   **How to test (local)**
   1. Activate venv
   2. Run unit tests for the registry:
      ```bash
      python -m pytest tests/test_nlp_registry.py -q
   ````

   3. Quick smoke (Python REPL):
      ```py
      from hiperhealth.nlp import register_pipeline, get_pipeline
      # use mock pipeline already registered by tests or register a new one
      p = get_pipeline('test_lazy')
      print(p.process('hello'))
      ```

   **Roadmap & starter tasks**
   - Phase 1 (this PR): skeleton + tests (done)
   - Phase 2: add `spacy` adapter behind optional dependencies; include adapter
     tests that mock heavy models.
   - Phase 3: presidio/transformers adapters + metrics instrumentation.
   - Phase 4: benchmarking scripts and documentation for contributors.

   **Starter subtasks**
   - Add an adapter for `spaCy` that implements `BasePipeline` and registers as
     `spacy_basic`.
   - Add configuration support to select active pipelines via environment
     variable or config file.
   - Create a small CI job that runs registry unit tests and a mocked adapter
     test.

   **Checklist**
   - [ ] Unit tests for registry pass
   - [ ] Mock adapter present and registered
   - [ ] README / usage example included
   - [ ] Issue with roadmap and starter tasks created

   **Review notes**
   - This PR purposely avoids heavy runtime deps (spaCy/presidio) — adapters
     will be added in follow-up PRs to keep CI fast.
   - Key focus: API shape, thread-safety, and test coverage for lazy init.

   **Branch**: `ghs/nlp-pipeline-registry-lazy-load`
