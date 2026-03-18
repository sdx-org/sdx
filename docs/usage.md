# Usage

## Pipeline

The pipeline is the recommended way to use hiperhealth. It runs clinical stages
independently through composable skills.

### Running a single stage

```python
from hiperhealth.pipeline import PipelineContext, Stage, create_default_runner

runner = create_default_runner()

ctx = PipelineContext(
    patient={'symptoms': 'chest pain, shortness of breath', 'age': 45},
    language='en',
    session_id='visit-1',
)

ctx = runner.run(Stage.DIAGNOSIS, ctx)
print(ctx.results['diagnosis'].summary)
print(ctx.results['diagnosis'].options)
```

### Running multiple stages

```python
ctx = runner.run_many([Stage.SCREENING, Stage.DIAGNOSIS, Stage.EXAM], ctx)
```

### Persisting context between sessions

Stages can be executed at different times by different actors. Serialize the
context to JSON between invocations:

```python
# Monday — nurse runs screening
ctx = PipelineContext(
    patient={'symptoms': 'Patient John has fever and cough', 'age': 30},
    language='pt',
    session_id='encounter-42',
)
runner = create_default_runner()
ctx = runner.run(Stage.SCREENING, ctx)

# Save to database, file, or message queue
saved_json = ctx.model_dump_json()

# Wednesday — physician restores context and runs diagnosis
ctx = PipelineContext.model_validate_json(saved_json)
ctx = runner.run(Stage.DIAGNOSIS, ctx)
```

### Available stages

| Stage          | Description                                            |
| -------------- | ------------------------------------------------------ |
| `screening`    | Initial triage, PII de-identification                  |
| `intake`       | Data extraction from reports and wearable files        |
| `diagnosis`    | LLM-powered differential diagnosis                     |
| `exam`         | Exam/procedure suggestions based on diagnosis          |
| `treatment`    | Treatment planning (extensible via custom skills)      |
| `prescription` | Prescription generation (extensible via custom skills) |

### Built-in skills

The `create_default_runner()` factory registers three built-in skills in this
order:

| Skill              | Stages            | Description                                          |
| ------------------ | ----------------- | ---------------------------------------------------- |
| `PrivacySkill`     | screening, intake | De-identifies PII in patient data                    |
| `ExtractionSkill`  | intake            | Extracts text from medical reports and wearable data |
| `DiagnosticsSkill` | diagnosis, exam   | LLM-powered diagnosis and exam suggestions           |

Skills run in registration order, so `PrivacySkill` always runs before
`ExtractionSkill` within the same stage.

### Installing and registering custom skills

Custom skills can be installed from local paths or Git URLs using the
`SkillRegistry`, then activated in a runner by name:

```python
from hiperhealth.pipeline import SkillRegistry, create_default_runner, Stage

# Install skills into the internal registry (~/.hiperhealth/skills/)
registry = SkillRegistry()
registry.install('/path/to/ayurveda_skill/')
registry.install('https://github.com/my_org/tcm_skill')

# Create a runner and register installed skills
runner = create_default_runner()
runner.register('ayurveda', index=0)       # insert before built-ins
runner.register('traditional-chinese-medicine')  # append at end
```

See [Creating Skills](skills.md) for details on writing skill projects.

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
from hiperhealth.skills.diagnostics.core import differential

patient = {
    'age': 45,
    'gender': 'M',
    'symptoms': 'chest pain, shortness of breath',
    'previous_tests': 'ECG normal',
}

result = differential(patient, language='en', session_id='demo-1')
print(result.summary)
print(result.options)
```

### Suggested exams and procedures

```python
from hiperhealth.skills.diagnostics.core import exams

result = exams(
    ['Acute coronary syndrome'],
    language='en',
    session_id='demo-1',
)
print(result.summary)
print(result.options)
```

### Runtime configuration in code

```python
from hiperhealth.skills.diagnostics.core import differential
from hiperhealth.llm import LLMSettings

settings = LLMSettings(
    provider='ollama',
    model='llama3.2:3b',
    api_params={'base_url': 'http://localhost:11434/v1'},
)

result = differential(
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
from hiperhealth.skills.extraction.medical_reports import (
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
from hiperhealth.skills.extraction.wearable import WearableDataFileExtractor

extractor = WearableDataFileExtractor()
data = extractor.extract_wearable_data(
    'tests/data/wearable/wearable_data.csv'
)
print(data[:2])
```

## De-identification

```python
from hiperhealth.skills.privacy.deidentifier import (
    Deidentifier,
    deidentify_patient_record,
)

engine = Deidentifier()
record = {
    'symptoms': 'Patient John Doe reports severe headache.',
    'mental_health': 'Lives at 123 Main St',
}
clean = deidentify_patient_record(record, engine)
print(clean)
```

## Raw LLM output capture

Diagnostics responses are normalized and then written to `data/llm_raw/` using
the supplied `session_id` when present.

## Backward compatibility

The old import paths continue to work:

```python
# These still work
from hiperhealth.agents.diagnostics.core import differential, exams
from hiperhealth.agents.extraction.medical_reports import MedicalReportFileExtractor
from hiperhealth.agents.extraction.wearable import WearableDataFileExtractor
from hiperhealth.privacy.deidentifier import Deidentifier
```

The canonical locations are now under `hiperhealth.skills.*` and
`hiperhealth.pipeline`.
