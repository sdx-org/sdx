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

`hiperhealth` is a Python library for clinical AI workflows. It provides
diagnostics-oriented LLM helpers, local medical report text extraction, wearable
data normalization, de-identification utilities, and healthcare schemas/models.

- License: BSD 3 Clause
- Documentation: https://hiperhealth.com
- [Informed Consent Template](./informed_consent.md)

## Features

- Differential diagnosis suggestions from structured patient data
- Exam and procedure suggestions from selected diagnoses
- Provider-configurable LLM backend through LiteLLM
- Local PDF/image medical report text extraction
- CSV/JSON wearable data extraction and normalization
- PII detection and de-identification helpers
- Pydantic schemas and SQLAlchemy FHIR-oriented models

## Documentation guide

- Start with [Installation](./installation.md)
- Configure LLM backends in [LLM Configuration](./llm_configuration.md)
- See end-to-end examples in [Usage](./usage.md)

## Current scope

This repository is the `hiperhealth` library/SDK, not the web application.

The current diagnostics pipeline returns `LLMDiagnosis` objects with:

- `summary`: short clinical summary text
- `options`: diagnoses or exam/procedure suggestions

Medical report extraction currently returns extracted text and metadata rather
than FHIR resources.

## Credits

This package was created with Cookiecutter and the
[osl-incubator/scicookie](https://github.com/osl-incubator/scicookie) project
template.
