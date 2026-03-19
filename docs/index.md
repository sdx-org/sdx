---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

![HiPerHealth](images/logo.png){ width="180" }

# HiPerHealth

<p class="hero-tagline">
A Python library for clinical AI workflows.<br>
Composable, stage-independent pipelines for screening, diagnosis, treatment, and more.
</p>

<div class="hero-buttons">

[Get Started](installation.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/hiperhealth/hiperhealth){ .md-button }

</div>

</div>

---

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

### :material-pipe: Skill-Based Pipeline

Stages run independently, at different times, by different actors. Compose
clinical workflows from modular skills.

</div>

<div class="feature-card" markdown>

### :material-file-document-outline: Session Files

Parquet-backed event logs for persistent, resumable clinical workflows across
multiple patient visits.

</div>

<div class="feature-card" markdown>

### :material-clipboard-check-outline: Requirement Checking

Skills declare what information they need before execution, with three priority
levels: required, supplementary, and deferred.

</div>

<div class="feature-card" markdown>

### :material-stethoscope: Built-in Skills

**DiagnosticsSkill** for differential diagnosis, **ExtractionSkill** for
PDF/image reports, and **PrivacySkill** for PII de-identification.

</div>

<div class="feature-card" markdown>

### :material-puzzle-outline: Extensible

Create custom skills as Python classes, install third-party skills via entry
points or Git URLs.

</div>

<div class="feature-card" markdown>

### :material-flask-outline: Data Science Friendly

Sessions are standard parquet files, queryable with pandas, polars, or DuckDB.
Use from Jupyter notebooks.

</div>

</div>

---

## Quick Start

<div class="quick-start" markdown>

Install from PyPI:

```bash
pip install hiperhealth
```

Run a diagnosis:

```python
from hiperhealth.pipeline import PipelineContext, Stage, create_default_runner

runner = create_default_runner()

ctx = PipelineContext(
    patient={"symptoms": "chest pain, shortness of breath", "age": 45},
    language="en",
    session_id="visit-1",
)

ctx = runner.run(Stage.DIAGNOSIS, ctx)
print(ctx.results["diagnosis"].summary)
```

</div>

---

## Documentation Guide

| Section                                   | Description                                             |
| ----------------------------------------- | ------------------------------------------------------- |
| [Installation](installation.md)           | Install hiperhealth and system dependencies             |
| [LLM Configuration](llm_configuration.md) | Configure LLM backends (OpenAI, Ollama, Groq, and more) |
| [Usage](usage.md)                         | End-to-end examples: pipeline, sessions, extraction     |
| [Creating Skills](skills.md)              | Build and register custom pipeline skills               |
| [API Reference](api/index.md)             | Auto-generated Python API documentation                 |
| [Changelog](changelog.md)                 | Release notes and version history                       |
| [Contributing](contributing.md)           | Development setup and contributor guide                 |

---

<div class="badges" markdown>

[![PyPI](https://img.shields.io/pypi/v/hiperhealth)](https://pypi.org/project/hiperhealth/)
![Python](https://img.shields.io/pypi/pyversions/hiperhealth)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue)](https://github.com/hiperhealth/hiperhealth/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

</div>
