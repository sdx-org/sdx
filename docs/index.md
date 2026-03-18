![LOGO](/images/logo.png)

# hiperhealth

[![Built with Material for MkDocs](https://img.shields.io/badge/Material_for_MkDocs-526CFE?style=for-the-badge&logo=MaterialForMkDocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)
![Conda](https://img.shields.io/badge/Virtual%20environment-conda-brightgreen?logo=anaconda)[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![vulture](https://img.shields.io/badge/Find%20unused%20code-vulture-blue)
![mypy](https://img.shields.io/badge/Static%20typing-mypy-blue)
![pytest](https://img.shields.io/badge/Testing-pytest-cyan?logo=pytest)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
![Makim](https://img.shields.io/badge/Automation%20task-Makim-blue)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI-blue?logo=githubactions)

`hiperhealth` is a Python library for clinical AI workflows. It provides a
**skill-based pipeline** for composable, stage-independent execution of clinical
tasks — from screening and intake through diagnosis, exams, treatment, and
prescription.

- License: BSD 3 Clause
- Documentation: https://hiperhealth.com
- [Informed Consent Template](./informed_consent.md)

## Features

- **Skill-based pipeline** — stages run independently, at different times, by
  different actors. Context serializes to JSON between invocations.
- **Built-in skills:**
  - **DiagnosticsSkill** — differential diagnosis and exam suggestions via LLM
  - **ExtractionSkill** — PDF/image medical report and CSV/JSON wearable data
    extraction
  - **PrivacySkill** — PII detection and de-identification
- **Extensible** — create custom skills as Python classes, install third-party
  skills via entry points
- Provider-configurable LLM backend through LiteLLM (8+ providers)
- Pydantic schemas and SQLAlchemy FHIR-oriented models

## Documentation guide

- Start with [Installation](./installation.md)
- Configure LLM backends in [LLM Configuration](./llm_configuration.md)
- See end-to-end examples in [Usage](./usage.md)
- Learn how to build custom skills in [Creating Skills](./skills.md)

## Current scope

This repository is the `hiperhealth` library/SDK, not the web application.

The pipeline operates through `PipelineContext`, a Pydantic model that carries
patient data, results, and audit entries across stages. Each stage produces
results accessible via `ctx.results[stage_name]`.

## Credits

This package was created with Cookiecutter and the
[osl-incubator/scicookie](https://github.com/osl-incubator/scicookie) project
template.
