# hiperhealth

Core Python library for HiperHealth clinical AI workflows.

This repository is the **library/SDK** package (`hiperhealth`) and not the web
application.

- Software License: BSD 3-Clause
- Documentation: https://hiperhealth.com
- Source: https://github.com/hiperhealth/hiperhealth

## What this library provides

- **Skill-based pipeline** for composable clinical workflows:
  - Stages (screening, intake, diagnosis, exam, treatment, prescription) that
    can be executed independently at different times by different actors
  - Skills are composable plugins that affect one or more stages
  - **Session files** (parquet) for persistent, resumable interactions — assess
    requirements, collect patient answers, and execute stages asynchronously
  - **Requirement checking** — skills declare what information they need before
    execution, enabling structured clinical data gathering
- **Built-in skills:**
  - **DiagnosticsSkill** — LLM-powered differential diagnosis and exam
    suggestions
  - **ExtractionSkill** — medical reports (PDF/image) and wearable data
    (CSV/JSON) extraction
  - **PrivacySkill** — PII detection and de-identification
- **Data science friendly** — sessions are parquet files that can be loaded
  directly into pandas, polars, or DuckDB for analysis. Physicians can use
  hiperhealth interactively from Jupyter notebooks to study patient cases.
- Domain schemas and models:
  - Pydantic schemas
  - SQLAlchemy FHIR model definitions

## Installation

### Stable release

```bash
pip install hiperhealth
```

### From source (development)

```bash
git clone https://github.com/hiperhealth/hiperhealth.git
cd hiperhealth
./scripts/install-dev.sh
```

## System requirements

Some extraction features depend on system packages:

- `tesseract` (OCR for image-based reports)
- `libmagic` (MIME type detection)

They are included in the conda dev environment (`conda/dev.yaml`).

## Configuration

Diagnostics and exam suggestions use a LiteLLM-backed adapter, so the provider
can be changed through environment variables or `LLMSettings(...)` without
editing library code.

Recognized provider values for `HIPERHEALTH_DIAGNOSTICS_LLM_PROVIDER` are:

- `openai` (default)
- `ollama`
- `cohere`
- `fireworks`
- `gemini`
- `groq`
- `huggingface`
- `huggingface-inference`
- `together`

Compatibility alias:

- `ollama-openai` is accepted and normalized to `ollama`

Supported diagnostics configuration variables:

- `HIPERHEALTH_DIAGNOSTICS_LLM_PROVIDER`
- `HIPERHEALTH_DIAGNOSTICS_LLM_MODEL`
- `HIPERHEALTH_DIAGNOSTICS_LLM_API_KEY`
- `HIPERHEALTH_DIAGNOSTICS_LLM_BASE_URL`
- `HIPERHEALTH_DIAGNOSTICS_LLM_TEMPERATURE`
- `HIPERHEALTH_DIAGNOSTICS_LLM_MAX_TOKENS`
- `HIPERHEALTH_DIAGNOSTICS_LLM_API_PARAMS` (JSON object of extra LiteLLM kwargs)

Generic fallbacks are also supported through `HIPERHEALTH_LLM_*`. For OpenAI
compatibility, `OPENAI_API_KEY` and `OPENAI_MODEL` are still used as legacy
fallbacks.

Default models:

- `openai`: `o4-mini`
- `ollama`: `llama3.2:1b`

Example with OpenAI:

```bash
export HIPERHEALTH_DIAGNOSTICS_LLM_PROVIDER="openai"
export HIPERHEALTH_DIAGNOSTICS_LLM_API_KEY="your-key"
export HIPERHEALTH_DIAGNOSTICS_LLM_MODEL="o4-mini"
```

Example with local Ollama:

```bash
export HIPERHEALTH_DIAGNOSTICS_LLM_PROVIDER="ollama"
export HIPERHEALTH_DIAGNOSTICS_LLM_MODEL="llama3.2:3b"
export HIPERHEALTH_DIAGNOSTICS_LLM_BASE_URL="http://localhost:11434/v1"
```

If you use a fully-qualified LiteLLM model name such as `openai/o4-mini` or
`groq/llama-3.3-70b-versatile`, the model string is passed through as-is. In
that case, set `HIPERHEALTH_DIAGNOSTICS_LLM_API_KEY` explicitly unless your
chosen provider also matches one of the recognized provider names above.

More detail is available in
[docs/llm_configuration.md](docs/llm_configuration.md).

## Quickstart

### 1. Pipeline-based workflow (recommended)

The pipeline runs stages independently through composable skills:

```python
from hiperhealth.pipeline import PipelineContext, Stage, create_default_runner

# Create a runner with all built-in skills
runner = create_default_runner()

# Run screening (de-identifies PII)
ctx = PipelineContext(
    patient={"symptoms": "Patient John has chest pain", "age": 45},
    language="en",
    session_id="visit-1",
)
ctx = runner.run(Stage.SCREENING, ctx)

# Serialize context — different actor can resume later
saved = ctx.model_dump_json()

# ... hours later, restore and run diagnosis
ctx = PipelineContext.model_validate_json(saved)
ctx = runner.run(Stage.DIAGNOSIS, ctx)
print(ctx.results["diagnosis"].summary)
```

### 2. Session-based workflow (recommended for multi-visit scenarios)

Sessions use parquet files to persist the full interaction history. This
supports asynchronous workflows where data arrives over days (e.g., patient
answers, lab results):

```python
from hiperhealth.pipeline import Session, Stage, create_default_runner

runner = create_default_runner()

# Day 1: Patient reports symptoms
session = Session.create('/data/sessions/abc123.parquet')
session.set_clinical_data({
    'symptoms': 'chronic bloating, fatigue',
    'age': 34,
    'biological_sex': 'female',
})

# Check what information skills need before running diagnosis
inquiries = runner.check_requirements(Stage.DIAGNOSIS, session)
# -> [Inquiry(field='dietary_history', priority='required', ...)]
# Show required/supplementary inquiries to the patient

# Day 3: Patient responds
session = Session.load('/data/sessions/abc123.parquet')
session.provide_answers({'dietary_history': 'High carb, low fiber...'})

# Re-check requirements, then run
inquiries = runner.check_requirements(Stage.DIAGNOSIS, session)
required = [i for i in inquiries if i.priority == 'required']
if not required:
    runner.run_session(Stage.DIAGNOSIS, session, llm=my_llm)
```

### 3. Interactive analysis in Jupyter notebooks

Physicians can use hiperhealth directly from notebooks to study patient cases:

```python
from hiperhealth.pipeline import Session, Stage, create_default_runner
import polars as pl

runner = create_default_runner()
session = Session.load('/data/sessions/abc123.parquet')

# Inspect session state
session.clinical_data       # all patient data
session.results             # stage results
session.pending_inquiries   # unanswered questions
session.stages_completed    # which stages have run

# Analyze the event log directly
df = pl.read_parquet('/data/sessions/abc123.parquet')
df.filter(pl.col('event_type') == 'stage_completed')
```

### 4. Standalone diagnostic functions

```python
from hiperhealth.skills.diagnostics.core import differential, exams

patient = {"age": 45, "symptoms": "chest pain, shortness of breath"}

dx = differential(patient, language="en", session_id="demo-1")
print(dx.summary, dx.options)

ex = exams(["Acute coronary syndrome"], language="en", session_id="demo-1")
print(ex.summary, ex.options)
```

Supported output languages: `en`, `pt`, `es`, `fr`, `it`. Unknown values fall
back to English.

### 5. Data extraction

```python
from hiperhealth.skills.extraction.medical_reports import MedicalReportFileExtractor
from hiperhealth.skills.extraction.wearable import WearableDataFileExtractor

# Medical reports (PDF/image)
report_ext = MedicalReportFileExtractor()
report = report_ext.extract_report_data("path/to/report.pdf")

# Wearable data (CSV/JSON)
wearable_ext = WearableDataFileExtractor()
data = wearable_ext.extract_wearable_data("path/to/data.csv")
```

### 6. De-identification

```python
from hiperhealth.skills.privacy.deidentifier import Deidentifier, deidentify_patient_record

engine = Deidentifier()
record = {
    "symptoms": "Patient John Doe reports severe headache.",
    "mental_health": "Lives at 123 Main St",
}
clean = deidentify_patient_record(record, engine)
```

### 7. Installing and registering custom skills

Skills are installed from local paths or Git URLs into an internal registry,
then registered into a runner by name:

```python
from hiperhealth.pipeline import SkillRegistry, create_default_runner, Stage

# Install a skill (copies into ~/.hiperhealth/skills/)
registry = SkillRegistry()
registry.install('/path/to/ayurveda_skill/')
registry.install('https://github.com/my_org/tcm_skill')

# Register installed skills into a pipeline
runner = create_default_runner()
runner.register('ayurveda', index=0)       # insert at the beginning
runner.register('traditional-chinese-medicine')  # append at the end
```

Each skill project must include a `hiperhealth.yaml` metadata file. See
[Creating Skills](docs/skills.md) for a full guide on writing skill projects.

### 8. Custom skill class

```python
from hiperhealth.pipeline import BaseSkill, SkillMetadata, Stage

class AyurvedaSkill(BaseSkill):
    def __init__(self):
        super().__init__(SkillMetadata(
            name="ayurveda",
            stages=(Stage.DIAGNOSIS, Stage.TREATMENT),
        ))

    def pre(self, stage, ctx):
        fragments = ctx.extras.setdefault("prompt_fragments", {})
        fragments[stage] = "Also consider Ayurvedic perspectives."
        return ctx
```

## Repository layout

- `src/hiperhealth/pipeline/`: stage runner, context, session, skill base
  classes, registry, discovery
- `src/hiperhealth/skills/`: built-in skills (diagnostics, extraction, privacy)
- `src/hiperhealth/agents/`: backward-compatible re-exports and shared utilities
- `src/hiperhealth/schema/`: Pydantic schemas
- `src/hiperhealth/models/`: SQLAlchemy FHIR models
- `tests/`: unit and integration tests
- `docs/`: Douki documentation source

## Development

### Create development environment

```bash
conda env create -f conda/dev.yaml -n hiperhealth
conda activate hiperhealth
./scripts/install-dev.sh
```

### Run tests

```bash
pytest -vv
```

### Run quality checks

```bash
pre-commit run --all-files
ruff check .
mypy .
```

### Build docs locally

```bash
makim docs.preview
```

(This will build the documentation and serve it locally at
http://127.0.0.1:8000)

## License

BSD 3-Clause. See [LICENSE](LICENSE).
