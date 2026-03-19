# Creating Skills

Skills are composable plugins that extend the clinical pipeline. Each skill can
affect one or more stages and is a Python class that subclasses `BaseSkill`.

## Architecture overview

```
StageRunner
    |
    +-- PrivacySkill        (screening, intake)
    +-- ExtractionSkill     (intake)
    +-- DiagnosticsSkill    (diagnosis, exam)
    +-- YourCustomSkill     (diagnosis, treatment)
```

When a stage runs, the runner finds all skills that declare that stage in their
metadata and calls their hooks in **registration order** (the order you pass
them to `StageRunner`):

0. Optional assessment pass via `StageRunner.check_requirements(stage, session)`
   which calls `check_requirements()` for matching skills in registration order
1. All `pre()` hooks (in registration order)
2. All `execute()` hooks (in registration order)
3. All `post()` hooks (in registration order)

The system integrator controls execution order — not the skill author.

## Skill project structure

Every skill project is a directory containing at minimum a `hiperhealth.yaml`
metadata file and a Python module with the skill class:

```
my_skill/
├── hiperhealth.yaml          # required: skill metadata
├── skill.py                  # required: contains the BaseSkill subclass
├── prompts/                  # optional: prompt templates
│   └── diagnosis.txt
├── data/                     # optional: reference data, lookup tables
│   └── herbs.json
└── requirements.txt          # optional: extra pip dependencies
```

### `hiperhealth.yaml`

```yaml
# Required fields
name: my_org.skill_name # unique identifier
version: 1.0.0 # semver
entry_point: "skill:MySkillClass" # module:ClassName within the folder
stages:
  - diagnosis
  - treatment

# Human-readable (optional)
description: >-
  A brief description of what this skill does.
author: "Your Name <email@example.com>"
license: MIT
homepage: https://github.com/my_org/my_skill

# Compatibility (optional)
min_hiperhealth_version: "0.4.0"

# Extra pip dependencies this skill needs (optional)
dependencies:
  - some-package>=1.0
```

| Field                     | Required | Description                                         |
| ------------------------- | -------- | --------------------------------------------------- |
| `name`                    | yes      | Unique skill identifier. Used in `register("name")` |
| `version`                 | yes      | Semver string                                       |
| `entry_point`             | yes      | `module:ClassName` relative to the skill folder     |
| `stages`                  | yes      | List of stage names this skill participates in      |
| `description`             | no       | Human-readable description                          |
| `author`                  | no       | Author name and contact                             |
| `license`                 | no       | License identifier                                  |
| `homepage`                | no       | URL for documentation or source                     |
| `min_hiperhealth_version` | no       | Minimum compatible hiperhealth version              |
| `dependencies`            | no       | Extra pip packages the skill requires               |

## Minimal skill

```python
from hiperhealth.pipeline import BaseSkill, SkillMetadata, Stage
from hiperhealth.pipeline.context import PipelineContext


class GreetingSkill(BaseSkill):
    def __init__(self):
        super().__init__(
            SkillMetadata(
                name='my_org.greeting',
                version='1.0.0',
                stages=(Stage.SCREENING,),
                description='Adds a greeting to the context.',
            )
        )

    def execute(self, stage, ctx):
        name = ctx.patient.get('name', 'Patient')
        ctx.extras['greeting'] = f'Welcome, {name}!'
        return ctx
```

## SkillMetadata fields

| Field         | Type              | Default   | Description                                 |
| ------------- | ----------------- | --------- | ------------------------------------------- |
| `name`        | `str`             | required  | Unique identifier, e.g. `my_org.skill_name` |
| `version`     | `str`             | `"0.1.0"` | Semantic version of the skill               |
| `stages`      | `tuple[str, ...]` | `()`      | Which stages this skill participates in     |
| `description` | `str`             | `""`      | Human-readable description                  |

## Hooks

Each skill has four hooks. Override only the ones you need — the base class
provides no-op defaults.

### `check_requirements(stage, ctx) -> list[Inquiry]`

Called before execution to determine what information is needed. Return a list
of `Inquiry` objects describing what data the skill needs. The default returns
an empty list (no extra data needed).

In the session workflow, callers do not invoke this hook directly. They call
`StageRunner.check_requirements(stage, session, **kwargs)`, which:

- Builds `ctx` from `session.to_context()`
- Merges both `session.set_clinical_data()` and `session.provide_answers()` into
  `ctx.patient`
- Stores extra keyword arguments in `ctx.extras['_run_kwargs']`
- Records `check_requirements_started`, `inquiries_raised`, and
  `check_requirements_completed` events in the session file

That means skill authors should treat `ctx.patient` as the current merged
clinical state and only return inquiries for fields that are still missing.
`ctx.results` also contains outputs from previously completed stages. The only
thing not passed into the hook is the raw session event log itself.

Each inquiry has a **priority** reflecting clinical data availability:

| Priority        | Meaning                                     | Example                          |
| --------------- | ------------------------------------------- | -------------------------------- |
| `required`      | Must have before this stage can run         | Basic symptoms for diagnosis     |
| `supplementary` | Would improve results, available now        | Dietary history, medication list |
| `deferred`      | Only available after a future pipeline step | Lab results (after exam stage)   |

Example:

```python
import json

from typing import Literal

from pydantic import BaseModel

from hiperhealth.agents.client import chat_structured
from hiperhealth.pipeline import BaseSkill, Inquiry, SkillMetadata, Stage
from hiperhealth.pipeline.context import PipelineContext


class InquiryDraft(BaseModel):
    field: str
    label: str
    description: str = ''
    priority: Literal['required', 'supplementary', 'deferred'] = (
        'supplementary'
    )
    input_type: str = 'text'
    choices: list[str] | None = None


class InquiryDraftList(BaseModel):
    inquiries: list[InquiryDraft]


class GutMicrobiomeSkill(BaseSkill):
    def __init__(self):
        super().__init__(SkillMetadata(
            name='gut_microbiome',
            stages=(Stage.DIAGNOSIS, Stage.TREATMENT),
        ))

    def check_requirements(
        self, stage: str, ctx: PipelineContext
    ) -> list[Inquiry]:
        if stage != Stage.DIAGNOSIS:
            return []

        run_kwargs = ctx.extras.get('_run_kwargs', {})
        llm = run_kwargs.get('llm')
        llm_settings = run_kwargs.get('llm_settings')

        system_prompt = (
            'You are a clinical assistant specialized in gut microbiome care. '
            'Review the full anamnesis and prior stage outputs. '
            'Return only the additional information that would be most useful '
            'for this skill and is not already present. '
            'Prioritize safety-critical items as "required", useful but '
            'non-blocking items as "supplementary", and items that naturally '
            'arrive later as "deferred".'
        )

        payload = {
            'patient': ctx.patient,
            'results': ctx.results,
            'stage': stage,
        }
        response = chat_structured(
            system_prompt,
            json.dumps(payload, ensure_ascii=False),
            InquiryDraftList,
            session_id=ctx.session_id,
            llm=llm,
            llm_settings=llm_settings,
        )

        existing_fields = set(ctx.patient.keys())
        return [
            Inquiry(
                skill_name=self.metadata.name,
                stage=stage,
                field=item.field,
                label=item.label,
                description=item.description,
                priority=item.priority,
                input_type=item.input_type,
                choices=item.choices,
            )
            for item in response.inquiries
            if item.field not in existing_fields
        ]
```

### Requirement / answer loop

`provide_answers()` does not call any skill hooks by itself. It appends an
`answers_provided` event, and those fields become part of
`session.clinical_data` the next time the runner builds a context.

```python
from hiperhealth.pipeline import Session, Stage, StageRunner

runner = StageRunner(skills=[GutMicrobiomeSkill()])
session = Session.create('/tmp/case.parquet')

session.set_clinical_data({'symptoms': 'bloating'})
inquiries = runner.check_requirements(Stage.DIAGNOSIS, session)

session.provide_answers({'dietary_history': 'high carb, low fiber'})

# ctx.patient will now include both symptoms and dietary_history
inquiries = runner.check_requirements(Stage.DIAGNOSIS, session)
required = [i for i in inquiries if i.priority == 'required']

if not required:
    runner.run_session(Stage.DIAGNOSIS, session)
```

You can inspect `session.pending_inquiries` at any point to see which recorded
inquiries still do not have matching fields in `session.clinical_data`.

### `pre(stage, ctx) -> PipelineContext`

Called before the main execution. Use it to prepare data, inject prompt
fragments, or validate preconditions.

### `execute(stage, ctx) -> PipelineContext`

The main work of the skill. Read from `ctx.patient`, `ctx.results`, or
`ctx.extras`, and write results to `ctx.results[stage]`.

### `post(stage, ctx) -> PipelineContext`

Called after execution. Use it for logging, cleanup, or result transformation.

## PipelineContext

The context is a Pydantic model that flows between stages:

```python
class PipelineContext(BaseModel):
    patient: dict[str, Any] = {}        # Patient data
    language: str = 'en'                # Prompt language
    session_id: str | None = None       # Session tracking
    results: dict[str, Any] = {}        # Stage results, keyed by stage name
    audit: list[AuditEntry] = []        # Execution audit log
    extras: dict[str, Any] = {}         # Skill-specific data, prompt fragments
```

### Serialization

The context serializes to JSON for persistence between invocations:

```python
# Save
json_str = ctx.model_dump_json()

# Restore
ctx = PipelineContext.model_validate_json(json_str)
```

This allows stages to run hours or days apart, by different actors.

## Modifying prompts from a skill

Skills can inject additional instructions into the prompts used by
`DiagnosticsSkill` via **prompt fragments**:

```python
class AyurvedaSkill(BaseSkill):
    def __init__(self):
        super().__init__(
            SkillMetadata(
                name='ayurveda',
                stages=(Stage.DIAGNOSIS, Stage.TREATMENT),
            )
        )

    def pre(self, stage, ctx):
        fragments = ctx.extras.setdefault('prompt_fragments', {})
        fragments[stage] = (
            'Also consider Ayurvedic perspectives and traditional '
            'Indian medicine approaches.'
        )
        return ctx
```

The `DiagnosticsSkill` checks `ctx.extras['prompt_fragments']` in two places:

- `{stage}` for the main execution prompt
- `{stage}_requirements` for the requirement-gathering prompt used by
  `check_requirements()`

## Skill registry

The `SkillRegistry` manages skill installation and loading. Skills can be
installed from local paths or Git URLs, and are stored in an internal registry
directory (`~/.hiperhealth/skills/`).

### Installing skills

```python
from hiperhealth.pipeline import SkillRegistry

registry = SkillRegistry()

# Install from a local directory
registry.install('/path/to/my_skill/')

# Install from a Git repository
registry.install('https://github.com/my_org/my_skill')

# List all available skills (built-in + installed)
for manifest in registry.list_skills():
    print(f'{manifest.name} v{manifest.version}: {manifest.description}')

# Remove an installed skill
registry.uninstall('my_org.skill_name')
```

### Registering skills in the runner

Use `register()` to activate an installed skill by name:

```python
from hiperhealth.pipeline import SkillRegistry, StageRunner, Stage

registry = SkillRegistry()
runner = StageRunner(registry=registry)

# Register skills — order defines execution order
runner.register('hiperhealth.privacy')
runner.register('hiperhealth.diagnostics')

# Insert a custom skill at the beginning
runner.register('my_org.greeting', index=0)
```

### Using the default runner with custom skills

```python
from hiperhealth.pipeline import create_default_runner

runner = create_default_runner()

# Add an externally installed skill
runner.register('ayurveda')

ctx = runner.run(Stage.DIAGNOSIS, ctx)
```

### Internal registry layout

```
~/.hiperhealth/
└── skills/
    ├── manifest.json                    # index of installed skills
    ├── ayurveda/
    │   ├── hiperhealth.yaml
    │   └── skill.py
    └── traditional-chinese-medicine/
        ├── hiperhealth.yaml
        ├── skill.py
        └── data/
            └── herbs.json
```

## Using the runner

### Register skills at construction time

The list order defines execution order:

```python
from hiperhealth.pipeline import StageRunner, Stage

runner = StageRunner(skills=[
    PrivacySkill(),       # runs first
    ExtractionSkill(),    # runs second
    DiagnosticsSkill(),   # runs third
    AyurvedaSkill(),      # runs last
])

ctx = runner.run(Stage.DIAGNOSIS, ctx)
```

### Run multiple stages

```python
ctx = runner.run_many(
    [Stage.SCREENING, Stage.INTAKE, Stage.DIAGNOSIS],
    ctx,
)
```

### Pass extra arguments

Extra keyword arguments to `run()` are available to skills via
`ctx.extras['_run_kwargs']`:

```python
ctx = runner.run(Stage.DIAGNOSIS, ctx, llm_settings=my_settings)
```

## Stages

The built-in stages are defined as a string enum:

| Stage                | Value            | Typical use                           |
| -------------------- | ---------------- | ------------------------------------- |
| `Stage.SCREENING`    | `"screening"`    | Initial triage, PII de-identification |
| `Stage.INTAKE`       | `"intake"`       | Data extraction from files            |
| `Stage.DIAGNOSIS`    | `"diagnosis"`    | Differential diagnosis                |
| `Stage.EXAM`         | `"exam"`         | Exam/procedure suggestions            |
| `Stage.TREATMENT`    | `"treatment"`    | Treatment planning                    |
| `Stage.PRESCRIPTION` | `"prescription"` | Prescription generation               |

Custom string stage names also work — the runner accepts any string, not only
enum values.

## Skill discovery via entry points

Third-party skills can also be auto-discovered if they register as Python entry
points:

```toml
# In the skill package's pyproject.toml
[project.entry-points."hiperhealth.skills"]
ayurveda = "my_package:AyurvedaSkill"
```

Then discover and use them:

```python
from hiperhealth.pipeline import discover_skills, StageRunner

third_party = discover_skills()
runner = StageRunner(skills=third_party)
```

## Example: full custom skill

Here is a complete example of a skill that adds intake data enrichment:

```python
from hiperhealth.pipeline import BaseSkill, SkillMetadata, Stage
from hiperhealth.pipeline.context import PipelineContext


class BMICalculatorSkill(BaseSkill):
    """Calculates BMI from height and weight in patient data."""

    def __init__(self):
        super().__init__(
            SkillMetadata(
                name='my_clinic.bmi_calculator',
                version='1.0.0',
                stages=(Stage.INTAKE,),
                description='Calculates BMI from patient height and weight.',
            )
        )

    def execute(self, stage, ctx):
        height = ctx.patient.get('height_m')
        weight = ctx.patient.get('weight_kg')

        if height and weight and height > 0:
            bmi = weight / (height ** 2)
            intake = ctx.results.setdefault(Stage.INTAKE, {})
            intake['bmi'] = round(bmi, 1)
            intake['bmi_category'] = self._categorize(bmi)

        return ctx

    def _categorize(self, bmi):
        if bmi < 18.5:
            return 'underweight'
        elif bmi < 25:
            return 'normal'
        elif bmi < 30:
            return 'overweight'
        return 'obese'
```

With `hiperhealth.yaml`:

```yaml
name: my_clinic.bmi_calculator
version: 1.0.0
entry_point: "skill:BMICalculatorSkill"
stages:
  - intake
description: Calculates BMI from patient height and weight.
```

Usage:

```python
from hiperhealth.pipeline import (
    PipelineContext, SkillRegistry, Stage, create_default_runner,
)

# Install the skill
registry = SkillRegistry()
registry.install('/path/to/bmi_calculator/')

# Use it in a pipeline
runner = create_default_runner()
runner.register('my_clinic.bmi_calculator')

ctx = PipelineContext(
    patient={'height_m': 1.75, 'weight_kg': 70},
)
ctx = runner.run(Stage.INTAKE, ctx)
print(ctx.results['intake']['bmi'])        # 22.9
print(ctx.results['intake']['bmi_category'])  # normal
```

## Testing skills

Skills are plain Python classes, so they are straightforward to test:

```python
from hiperhealth.pipeline import PipelineContext, Stage, StageRunner


def test_bmi_calculator():
    skill = BMICalculatorSkill()
    runner = StageRunner(skills=[skill])

    ctx = PipelineContext(
        patient={'height_m': 1.80, 'weight_kg': 90},
    )
    ctx = runner.run(Stage.INTAKE, ctx)

    assert ctx.results[Stage.INTAKE]['bmi'] == 27.8
    assert ctx.results[Stage.INTAKE]['bmi_category'] == 'overweight'
```
