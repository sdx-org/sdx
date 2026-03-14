# Usage

## Diagnostics

The diagnostics helpers return `LLMDiagnosis` objects with:

- `summary`: short summary text
- `options`: suggested diagnoses or exam/procedure names

Supported prompt languages are:

- `en`
- `pt`
- `es`
- `fr`
- `it`

Unknown language values fall back to English.

### Differential diagnosis

```python
from hiperhealth.agents.diagnostics import core as diag

patient = {
    'age': 45,
    'gender': 'M',
    'symptoms': 'chest pain, shortness of breath',
    'previous_tests': 'ECG normal',
}

result = diag.differential(patient, language='en', session_id='demo-1')
print(result.summary)
print(result.options)
```

### Suggested exams and procedures

```python
from hiperhealth.agents.diagnostics import core as diag

result = diag.exams(
    ['Acute coronary syndrome'],
    language='en',
    session_id='demo-1',
)
print(result.summary)
print(result.options)
```

### Runtime configuration in code

```python
from hiperhealth.agents.diagnostics import core as diag
from hiperhealth.llm import LLMSettings

settings = LLMSettings(
    provider='ollama',
    model='llama3.2:3b',
    api_params={'base_url': 'http://localhost:11434/v1'},
)

result = diag.differential(
    {'symptoms': 'headache'},
    llm_settings=settings,
)
```

## Medical report extraction

Medical reports are extracted locally from PDF or image files. The extractor
returns text and metadata, not FHIR resources.

Supported inputs:

- `pdf`
- `png`
- `jpg`
- `jpeg`

Example:

```python
from hiperhealth.agents.extraction.medical_reports import (
    MedicalReportFileExtractor,
)

extractor = MedicalReportFileExtractor()
report = extractor.extract_report_data(
    'tests/data/reports/pdf_reports/report-1.pdf'
)

print(report['source_name'])
print(report['mime_type'])
print(report['text'][:200])
```

Returned payload keys:

- `source_name`
- `source_type`
- `mime_type`
- `text`

If you only need the raw text:

```python
text = extractor.extract_text('tests/data/reports/pdf_reports/report-1.pdf')
```

## Wearable data extraction

Wearable data extraction supports CSV and JSON inputs and returns a normalized
list of dictionaries.

```python
from hiperhealth.agents.extraction.wearable import WearableDataFileExtractor

extractor = WearableDataFileExtractor()
data = extractor.extract_wearable_data(
    'tests/data/wearable/wearable_data.csv'
)
print(data[:2])
```

## Raw LLM output capture

Diagnostics responses are normalized and then written to `data/llm_raw/` using
the supplied `session_id` when present.
