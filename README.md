# hiperhealth

Core Python library for HiperHealth clinical AI workflows.

This repository is the **library/SDK** package (`hiperhealth`) and not the web
application.

- Software License: BSD 3-Clause
- Documentation: https://hiperhealth.com
- Source: https://github.com/hiperhealth/hiperhealth

## What this library provides

- LLM-powered clinical assistance utilities:
  - Differential diagnosis suggestions
  - Exam/procedure suggestions
- Data extraction utilities:
  - Medical reports (PDF/image) to extracted text plus metadata
  - Wearable data (CSV/JSON) parsing and normalization
- Privacy utilities:
  - PII detection and de-identification
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

### 1. Differential diagnosis and exam suggestions

```python
from hiperhealth.agents.diagnostics import core as diag

patient = {
    "age": 45,
    "gender": "M",
    "symptoms": "chest pain, shortness of breath",
    "previous_tests": "ECG normal"
}

dx = diag.differential(patient, language="en", session_id="demo-1")
print(dx.summary)
print(dx.options)

exams = diag.exams(["Acute coronary syndrome"], language="en", session_id="demo-1")
print(exams.summary)
print(exams.options)
```

Supported languages for diagnostics/exam prompts are `en`, `pt`, `es`, `fr`, and
`it`. Unknown values fall back to English.

To configure the backend directly in code instead of environment variables:

```python
from hiperhealth.agents.diagnostics import core as diag
from hiperhealth.llm import LLMSettings

settings = LLMSettings(
    provider="ollama",
    model="llama3.2:3b",
    api_params={"base_url": "http://localhost:11434/v1"},
)

dx = diag.differential(
    patient,
    language="en",
    session_id="demo-2",
    llm_settings=settings,
)
```

### 2. Wearable data extraction (CSV/JSON)

```python
from hiperhealth.agents.extraction.wearable import WearableDataFileExtractor

extractor = WearableDataFileExtractor()
data = extractor.extract_wearable_data("tests/data/wearable/wearable_data.csv")
print(data[:2])
```

### 3. Medical report extraction (PDF/image -> text)

```python
from hiperhealth.agents.extraction.medical_reports import MedicalReportFileExtractor

extractor = MedicalReportFileExtractor()
report = extractor.extract_report_data("tests/data/reports/pdf_reports/report-1.pdf")
print(report["mime_type"])
print(report["text"][:200])
```

### 4. De-identification

```python
from hiperhealth.privacy.deidentifier import Deidentifier, deidentify_patient_record

engine = Deidentifier()
record = {
    "symptoms": "Patient John Doe reports severe headache.",
    "mental_health": "Lives at 123 Main St"
}
clean = deidentify_patient_record(record, engine)
print(clean)
```

## Repository layout

- `src/hiperhealth/agents`: AI interaction and extraction modules
- `src/hiperhealth/privacy`: de-identification tools
- `src/hiperhealth/schema`: Pydantic schemas
- `src/hiperhealth/models`: SQLAlchemy models
- `tests`: unit and integration tests
- `docs`: MkDocs documentation source

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
mkdocs serve --watch docs --config-file mkdocs.yaml
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).
