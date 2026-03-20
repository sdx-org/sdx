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

Custom skills are typically published through channel repositories. Register a
channel once, inspect its skills, then install and register the canonical skill
id you want:

```python
from hiperhealth.pipeline import SkillRegistry, create_default_runner

registry = SkillRegistry()
registry.add_channel(
    'https://github.com/my-org/traditional-medicine.git',
    local_name='tm',
)

registry.list_channels()
registry.list_channel_skills('tm')
registry.install_skill('tm.ayurveda')

runner = create_default_runner()
runner.register('tm.ayurveda', index=0)
```

In notebooks and scripts, `registry.list_skills()` is useful for exploring all
registered channel skills and built-ins from one place.

```python
registry.list_skills()
registry.list_skills(channel='tm')
registry.list_skills(channel='tm', installed_only=True)
```

To compare results with and without a specific skill, temporarily disable it at
the runner layer without uninstalling or unregistering it:

```python
with runner.disabled({'tm.ayurveda'}):
    ctx_without_ayurveda = runner.run(Stage.TREATMENT, ctx)

ctx_with_ayurveda = runner.run(Stage.TREATMENT, ctx)
```

For one-off calls, `run()` also accepts `disabled_skills=`.

For channel lifecycle operations such as `update_channel()`, `remove_channel()`,
`install_channel(include_disabled=True)`, and the `skills-channel.yaml` /
`skill.yaml` manifest layout, see [Creating Skills](skills.md).

See [Creating Skills](skills.md) for the full channel repository layout and
manifest schema.

## Session-based workflow

For multi-visit clinical scenarios, sessions provide a parquet-backed event log
that persists the full interaction history. The calling system manages the
session file lifecycle (storage, deletion, retention).

### Creating and loading sessions

```python
from hiperhealth.pipeline import Session

# Create a new session
session = Session.create('/data/sessions/patient-visit.parquet', language='en')

# Provide clinical data (no PII — only clinical information)
session.set_clinical_data({
    'symptoms': 'chronic bloating, fatigue',
    'age': 34,
    'biological_sex': 'female',
})

# Load an existing session (e.g., days later)
session = Session.load('/data/sessions/patient-visit.parquet')
```

### Checking requirements before execution

Skills declare what information they need before a stage can run. Use
`check_requirements()` to gather inquiries from all relevant skills:

```python
from hiperhealth.pipeline import Session, Stage, create_default_runner

runner = create_default_runner()
session = Session.load('/data/sessions/patient-visit.parquet')

inquiries = runner.check_requirements(Stage.DIAGNOSIS, session)
for inq in inquiries:
    print(f'[{inq.priority}] {inq.field}: {inq.label}')
```

Inquiries have three priority levels reflecting clinical data availability:

| Priority        | Meaning                                     | Example                          |
| --------------- | ------------------------------------------- | -------------------------------- |
| `required`      | Must have before this stage can run         | Basic symptoms for diagnosis     |
| `supplementary` | Would improve results, available now        | Dietary history, medication list |
| `deferred`      | Only available after a future pipeline step | Lab results (after exam stage)   |

### Providing answers and running stages

```python
# Patient provides answers
session.provide_answers({'dietary_history': 'High carb, low fiber...'})

# Re-check — are required fields satisfied?
inquiries = runner.check_requirements(Stage.DIAGNOSIS, session)
required = [i for i in inquiries if i.priority == 'required']

if not required:
    runner.run_session(Stage.DIAGNOSIS, session, llm=my_llm)
```

### Multi-visit workflow

Not all data is available at the same time. A typical multi-visit flow:

```python
# Visit 1: Preliminary diagnosis with available data
runner.run_session(Stage.DIAGNOSIS, session, llm=my_llm)
runner.run_session(Stage.EXAM, session, llm=my_llm)  # requests lab work

# Visit 2: Lab results arrive, re-run with enriched data
session = Session.load('/data/sessions/patient-visit.parquet')
session.provide_answers({'stool_analysis': lab_results})
runner.run_session(Stage.DIAGNOSIS, session, llm=my_llm)  # complete diagnosis

# Visit 3: Treatment plan
runner.check_requirements(Stage.TREATMENT, session)
runner.run_session(Stage.TREATMENT, session, llm=my_llm)
```

### Inspecting session state

```python
session.clinical_data       # all patient data (merged from events)
session.results             # stage results keyed by stage name
session.pending_inquiries   # unanswered inquiries
session.stages_completed    # which stages have run
session.events              # raw event log
```

## Interactive analysis in Jupyter notebooks

hiperhealth is designed to work as a data science framework for clinical
analysis. Physicians can use it directly from Jupyter notebooks to study patient
cases:

```python
from hiperhealth.pipeline import Session, Stage, create_default_runner

runner = create_default_runner()

session = Session.create('/tmp/case-study.parquet')
session.set_clinical_data({
    'symptoms': 'chronic fatigue, joint pain, morning stiffness',
    'age': 52,
    'biological_sex': 'female',
    'family_history': 'rheumatoid arthritis (mother)',
})

# Check what information skills need
inquiries = runner.check_requirements(Stage.DIAGNOSIS, session)
for inq in inquiries:
    print(f'  [{inq.priority}] {inq.label}')

# Provide supplementary data interactively
session.provide_answers({
    'rheumatoid_factor': 'positive, 45 IU/mL',
    'anti_ccp': 'positive, 120 U/mL',
    'esr': '38 mm/hr',
})

# Run diagnosis
runner.run_session(Stage.DIAGNOSIS, session, llm=my_llm)
print(session.results[Stage.DIAGNOSIS])
```

### Analyzing session data with pandas or polars

The session parquet file is a standard parquet that can be queried directly:

```python
import polars as pl

df = pl.read_parquet('/tmp/case-study.parquet')

# See all events
df

# Filter to specific event types
df.filter(pl.col('event_type') == 'inquiries_raised')

# See what stages have been completed
df.filter(pl.col('event_type') == 'stage_completed').select('stage', 'timestamp')
```

## Diagnostics

The diagnostics helpers return `LLMDiagnosis` objects with:

- `summary`: short summary text
- `options`: suggested diagnoses or exam/procedure names

Supported output languages are:

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
