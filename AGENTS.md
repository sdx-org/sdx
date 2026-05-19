# AI Skill: HiperHealth Contributor Guide

This file is the shared operating manual for AI contributors working in the
`hiperhealth` repository. Use it to keep implementation style, review standards,
privacy expectations, and delivery quality consistent across agents.

## Recommended AI Configuration

For non-trivial work in this repository, especially changes touching clinical
logic, privacy, LLM prompts, schemas, or release automation, use Codex with the
latest model available and set the reasoning level to `xhigh`.

## When To Use This Skill

Use this guidance for any change inside the HiperHealth library/SDK repository:

- pipeline, stage runner, session, or skill registry changes
- built-in skill updates: diagnostics, extraction, privacy, or future skills
- LLM provider, prompt, structured-output, or configuration changes
- Pydantic schema, FHIR, SQLAlchemy model, or generated model changes
- CLI behavior updates
- documentation, examples, or MkDocs/Quarto updates
- tests, typing, linting, release, and CI-related maintenance

## Core Objectives

1. Protect patient privacy and avoid committing real PHI, secrets, or API keys.
2. Preserve public API and serialized data compatibility unless the task
   explicitly changes them.
3. Keep clinical outputs structured, validated, and clearly scoped as software
   assistance rather than medical advice.
4. Keep source, schemas, generated models, docs, and tests aligned.
5. Keep quality gates green: tests, coverage, mypy, ruff, douki, bandit,
   vulture, and pre-commit.
6. Make minimal, targeted edits with clear intent.

## Project Snapshot

- Package: `hiperhealth`
- Purpose: core Python library/SDK for HiperHealth clinical AI workflows; this
  repository is not the web application.
- Runtime: Python `>=3.10,<4` with CI currently covering Python 3.10 to 3.13 on
  Ubuntu and macOS.
- Packaging: `setuptools` with `src` layout and `uv`/`pip` installation.
- CLI entry point: `hiperhealth = hiperhealth.cli:main`
- Main architecture:
  `clinical data -> PipelineContext/Session -> StageRunner -> Skills -> typed outputs`
- Core concepts:
  - independently executable stages: screening, intake, diagnosis, exam,
    treatment, prescription
  - composable skills with `pre()`, `execute()`, and `post()` hooks
  - parquet-backed sessions for resumable workflows
  - Pydantic schemas for clinical outputs and FHIR-facing data models
  - LiteLLM-backed structured generation for provider-agnostic LLM access
  - Presidio-backed privacy/de-identification utilities
  - PDF/image/wearable extraction helpers
- Docs stack: MkDocs Material, mkdocstrings, and Quarto for rendered examples.
- Release: semantic-release with conventional commit / PR title expectations.

## Repository Layout

- `src/hiperhealth/`: package implementation
- `src/hiperhealth/agents/`: lower-level agent-style workflows and extraction
  helpers
- `src/hiperhealth/pipeline/`: stage, context, session, runner, registry, and
  skill abstractions
- `src/hiperhealth/skills/`: built-in installable skills and skill metadata
- `src/hiperhealth/privacy/`: de-identification implementation
- `src/hiperhealth/schema/`: Pydantic schema definitions
- `src/hiperhealth/models/sqla/`: SQLAlchemy models, including generated FHIR
  mappings
- `tests/`: pytest coverage and test data
- `docs/`: user-facing documentation and rendered examples
- `scripts/`: install, build, publishing, docs navigation, and model generation
  helpers
- `conda/dev.yaml`: development environment definition
- `.makim.yaml`: task runner definitions
- `.github/workflows/`: CI, docs, and release workflows
- `pyproject.toml`: package metadata, dependencies, and tool configuration

## Architecture And Responsibilities

### Pipeline and Skills

- `PipelineContext` carries patient data, language, session identifiers,
  accumulated results, and intermediate metadata.
- `Stage` defines the supported clinical workflow stages.
- `StageRunner` runs all skills registered for a stage in registration order.
- Skills should declare immutable `SkillMetadata` and implement only the hooks
  they need.
- `check_requirements()` should return structured `Inquiry` objects rather than
  raising for missing optional or collectable data.
- Keep stage execution deterministic except for explicitly injected external
  dependencies such as LLM clients.
- Do not add hidden global state to skills or runners; prefer explicit context,
  settings, and dependency injection.

### Sessions

- Sessions persist interaction history to parquet and must remain usable from
  notebooks and data tools.
- Preserve event log semantics and backward compatibility where possible.
- Treat session files as potentially sensitive clinical artifacts. Do not add
  logging or debug output that dumps patient records by default.

### Skill Registry and Channels

- The CLI manages skill channels and installs external skills from local paths
  or Git sources.
- Canonical external skill ids use `<local_channel_name>.<skill_name>`.
- Keep registry operations explicit, testable, and safe for repeated runs.
- Avoid network access in unit tests; mock Git or filesystem operations when
  needed.

### LLM Integration

- Use `hiperhealth.llm.LLMSettings` and the `StructuredLLM` protocol for LLM
  workflows.
- Keep LiteLLM provider handling provider-agnostic; avoid hard-coding one hosted
  provider unless the task explicitly requires it.
- Validate model output locally with Pydantic. Do not trust raw LLM text as a
  typed clinical result.
- Keep prompts concise, auditable, and free of unnecessary PHI exposure.
- Tests should use fake or injected completion functions. Do not require real
  API keys or live provider calls for standard test runs.
- Update `docs/llm_configuration.md` when configuration variables, defaults, or
  provider behavior changes.

### Schemas, FHIR, and SQLAlchemy Models

- Pydantic schemas in `src/hiperhealth/schema/` are source-of-truth for typed
  medical output structures.
- SQLAlchemy FHIR models live under `src/hiperhealth/models/sqla/`.
- If a schema change requires generated FHIR/SQLAlchemy updates, run the model
  generator and commit the synchronized generated output.
- Prefer explicit field names, validation, and enums over loose dictionaries for
  public clinical outputs.

### Privacy and Extraction

- Privacy features must be conservative by default and test-covered with
  synthetic data.
- Extraction features may depend on system packages such as `tesseract` and
  `libmagic`; keep graceful failures and clear error messages for missing
  dependencies.
- Never commit real medical reports, patient records, images, IDs, addresses, or
  names. Use synthetic fixtures only.

## Clinical, Privacy, And Safety Rules

- This repository supports clinical workflows, but code and docs should not
  present generated outputs as a replacement for clinician judgment.
- Do not add examples that include real PHI or secrets.
- Do not log API keys, authorization headers, raw prompts with sensitive data,
  full patient records, or unredacted model outputs by default.
- Prefer de-identified or minimized data before sending content to external LLM
  providers.
- Keep language around diagnosis and treatment suggestions cautious and
  evidence-seeking.
- Make failure modes safe: validate inputs, handle malformed external files, and
  surface actionable errors.

## Code Style And Standards

### Design Principles

- Apply SOLID principles where they improve clarity, testability, and change
  safety.
- Prefer guard clauses and early returns to keep control flow flat.
- Avoid unnecessary comments; comment only non-obvious intent or decisions.
- Favor explicit dependency injection for LLM clients, file systems, registries,
  and side-effectful operations.
- Keep public APIs typed and stable.

### Formatting and Static Quality

- Python indentation: 4 spaces.
- Ruff line length: 79.
- Ruff format uses single quotes.
- Type checking is strict in `pyproject.toml` with `check_untyped_defs = true`.
- Keep `src/hiperhealth/py.typed` valid by maintaining type annotations for
  public symbols.
- Pre-commit runs ruff, ruff format, douki, mypy, bandit, and vulture.
- Avoid broad `Any`, blanket `# type: ignore`, or untyped public callables.

### Python Docstring Convention

- Python docstrings use Douki-style YAML content blocks, for example:

  ```python
  """
  title: Short summary in imperative or descriptive form.
  parameters:
    value:
      type: str
  returns:
    type: str
  """
  ```

- Keep docstrings present and consistent for new or updated public modules,
  classes, and functions.
- If `douki sync` rewrites docstrings, review the result before finalizing.

### YAML and Automation Configuration

- Be careful with shell blocks in YAML-backed automation files such as
  `.github/workflows/*.yaml` and `.makim.yaml`.
- Prefer simple commands or scripts in `scripts/` for complex automation.
- Do not commit secrets or machine-specific paths into configuration files.

## Tooling And Commands

Environment setup:

```bash
conda env create -f conda/dev.yaml
conda activate hiperhealthlib
./scripts/install-dev.sh
```

If using mamba:

```bash
mamba env create -f conda/dev.yaml
mamba activate hiperhealthlib
./scripts/install-dev.sh
```

High-value commands:

```bash
# unit tests with coverage threshold from .makim.yaml
makim tests.unit

# run one test file or path
makim tests.unit --path tests/test_pipeline.py
pytest tests/test_pipeline.py -q

# CI-like local checks: tests then linter stack
makim tests.ci

# lint/pre-commit stack
makim tests.linter
pre-commit run --all-files --verbose

# targeted static checks
ruff check src tests
ruff format src tests
mypy src/hiperhealth

# docs
makim docs.render
makim docs.build

# package build
makim package.build
```

Schema/model generation:

```bash
makim gen.fhir-models
```

## CI Contract

GitHub Actions currently run:

- branch freshness check for pull requests
- tests on Python 3.10, 3.11, 3.12, and 3.13 across Ubuntu and macOS
- unit tests through `makim tests.unit`
- semantic-release PR title check using conventional commits
- linter/pre-commit stack through `makim tests.linter`
- docs build through `makim docs.build`
- release dry-run on pull requests and real release on manual workflow dispatch

Do not rely on behavior that only works on one Python version or one operating
system unless the task explicitly narrows support.

## Documentation Contract

When behavior changes, update relevant docs in the same change:

- `README.md` for public quickstarts and high-level behavior
- `docs/usage.md` for workflow examples
- `docs/skills.md` for skill/channel behavior
- `docs/llm_configuration.md` for LLM settings and providers
- `docs/installation.md` for setup or system dependency changes
- `docs/informed_consent.md` for consent, privacy, or data-handling changes
- generated API docs/navigation when public APIs change

If examples call LLM-backed workflows, show safe configuration and avoid real
secrets or patient-identifying content.

## Testing Contract

- Add or update tests near changed behavior under `tests/`.
- Keep test data synthetic and privacy-safe.
- Use fake LLM clients or injected completion functions for deterministic tests.
- Avoid live network calls, live provider credentials, or environment-dependent
  behavior in standard tests.
- For pipeline changes, cover `PipelineContext`, `StageRunner`, requirements,
  and session behavior where relevant.
- For registry/CLI changes, test CLI argument handling and installed skill state
  transitions.
- For schema changes, test validation and serialization round trips.
- For extraction changes, include malformed input and unsupported file cases.
- For privacy changes, include positive and negative de-identification cases.
- Keep coverage at or above the threshold enforced by `makim tests.unit`.

## Change Playbooks

### Adding or Changing a Built-in Skill

1. Update or add the skill implementation under `src/hiperhealth/skills/`.
2. Define accurate `SkillMetadata` and stage coverage.
3. Add or update `skill.yaml` metadata if the skill is installable.
4. Add deterministic tests for hooks, requirements, and outputs.
5. Update docs and README examples if users should know about the skill.

### Changing Pipeline or Session Behavior

1. Keep context/session serialization compatible where practical.
2. Add migration or compatibility handling if stored session shape changes.
3. Cover runner order, required inquiries, and resumability in tests.
4. Update usage docs and examples.

### Changing LLM Behavior

1. Route settings through `LLMSettings` and `StructuredLLM`.
2. Preserve provider-agnostic LiteLLM behavior unless explicitly changing it.
3. Validate outputs with Pydantic models.
4. Add tests with fake completions for success and invalid-output paths.
5. Update `docs/llm_configuration.md` and README defaults/examples.

### Changing Schemas or Generated Models

1. Update Pydantic schema definitions first.
2. Regenerate SQLAlchemy/FHIR models when required: `makim gen.fhir-models`.
3. Add validation, serialization, and model-mapping tests.
4. Review generated diffs carefully and avoid unrelated churn.

### Changing CLI or Registry Behavior

1. Update `src/hiperhealth/cli.py` and/or registry implementation.
2. Keep JSON output stable unless the task requires a breaking change.
3. Add CLI tests for new flags, failure modes, and state transitions.
4. Update `README.md` or `docs/skills.md`.

## Contributor Workflow Expectations

1. Inspect local files before planning changes.
2. Make minimal focused edits.
3. Add or update tests for behavior changes.
4. Run targeted checks first, then broader checks when feasible.
5. Keep docs/examples in sync with code behavior.
6. Use conventional commit wording for PR titles because semantic-release checks
   it.
7. Report any checks that could not be run and why.

## PR Review Checklist For AI Agents

Before finalizing, verify:

- [ ] no outdated project-specific content remains in this guide or touched
      files
- [ ] no real PHI, secrets, API keys, or unredacted patient records were added
- [ ] public API and serialized data changes are intentional and documented
- [ ] LLM behavior is provider-agnostic or explicitly documented otherwise
- [ ] behavior changes are covered by tests
- [ ] docs/examples are updated for user-visible changes
- [ ] generated models are synchronized when schemas require it
- [ ] `ruff`, `mypy`, and relevant tests pass, or blockers are reported
- [ ] no unrelated refactors or formatting churn are included
- [ ] error messages are explicit and actionable

## Non-Goals / Avoid

- Do not copy guidance from unrelated projects into this repository.
- Do not invent unsupported clinical claims, model guarantees, or workflow
  capabilities in docs or examples.
- Do not commit real patient data or credentials.
- Do not make standard tests depend on live LLM providers, network access, or
  local-only tools unless those tests are explicitly skipped when unavailable.
- Do not bypass Pydantic validation for structured clinical outputs.
- Do not add provider-specific branches when `LLMSettings`/LiteLLM can express
  the behavior generically.
- Do not update generated files without the corresponding source change, and do
  not update source schemas without necessary generated-file synchronization.
- Do not leave code, docs, tests, and examples out of sync.
