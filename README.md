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
  - Medical reports (PDF/image) to structured FHIR-like resources
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

Set these environment variables before using LLM-dependent features:

- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (optional, defaults to `o4-mini`)

Example:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="o4-mini"
```

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

### 2. Wearable data extraction (CSV/JSON)

```python
from hiperhealth.agents.extraction.wearable import WearableDataFileExtractor

extractor = WearableDataFileExtractor()
data = extractor.extract_wearable_data("tests/data/wearable/wearable_data.csv")
print(data[:2])
```

### 3. Medical report extraction (PDF/image -> structured output)

```python
from hiperhealth.agents.extraction.medical_reports import MedicalReportFileExtractor

extractor = MedicalReportFileExtractor()
report = extractor.extract_report_data("tests/data/reports/pdf_reports/report-1.pdf")
print(report.keys())
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
