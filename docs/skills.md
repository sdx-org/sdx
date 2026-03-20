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

Every skill project is a directory containing at minimum a `skill.yaml` metadata
file and a Python module with the skill class:

```
my_skill/
├── skill.yaml                # required: skill metadata
├── skill.py                  # required: contains the BaseSkill subclass
├── prompts/                  # optional: prompt templates
│   └── diagnosis.txt
├── data/                     # optional: reference data, lookup tables
│   └── herbs.json
└── requirements.txt          # optional: extra pip dependencies
```

### `skill.yaml`

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

The `SkillRegistry` treats a repository or folder as a **channel** and a folder
inside `skills/` as an installable **skill**. Channel-based skills use canonical
ids in the form `<local_channel_name>.<skill_name>`, such as `tm.ayurveda`.

### Recommended channel repository layout

```text
channel-repo/
├── skills-channel.yaml
├── README.md
├── LICENSE
├── docs/
├── infra/
├── scripts/
├── shared/
├── skills/
│   ├── ayurveda/
│   │   ├── skill.yaml
│   │   └── skill.py
│   ├── nutrition/
│   │   ├── skill.yaml
│   │   └── skill.py
│   └── triage/
│       ├── skill.yaml
│       └── skill.py
└── tests/
```

Only skills explicitly declared in `skills-channel.yaml` are installable.
Channel metadata does not need `discovery`, `ignore`, `path`, or `manifest`
fields, because every declared skill is resolved automatically to
`skills/<name>/skill.yaml`.

### `skills-channel.yaml`

```yaml
api_version: 1
channel:
  name: traditional-medicine
  display_name: Traditional Medicine
  default_alias: tm
  version: 0.1.0
  description: Complementary and traditional medicine skills
  homepage: https://github.com/my-org/traditional-medicine
  license: BSD-3-Clause
  min_hiperhealth_version: ">=0.5.0"

skills:
  - name: ayurveda
    enabled: true
    tags: [traditional-medicine, treatment]

  - name: nutrition
    enabled: true
    tags: [nutrition]

  - name: triage
    enabled: false
    tags: [screening]
```

Each declared skill is expected to live at `skills/<name>/skill.yaml`, which
keeps the per-skill manifest format explicit while removing repeated path
boilerplate from the channel manifest.

### Python API for scripts and notebooks

```python
from hiperhealth.pipeline import SkillRegistry

registry = SkillRegistry()
registry.add_channel(
    'https://github.com/my-org/traditional-medicine.git',
    local_name='tm',
)

registry.list_channels()
registry.list_channel_skills('tm')
registry.list_skills()
registry.list_skills(channel='tm')

registry.install_skill('tm.ayurveda')
registry.install_channel('tm')
registry.update_skill('tm.ayurveda')
registry.remove_skill('tm.ayurveda')
```

### Channel registration and source detection

`SkillRegistry.add_channel()` accepts:

- Local folder paths whose root contains `skills-channel.yaml`
- GitHub URLs
- GitLab URLs
- Generic git URLs

The source root is interpreted with these rules:

- If the root contains `skills-channel.yaml`, it is treated as a channel.
- If the root does not contain `skills-channel.yaml`, registration fails with a
  validation error.
- Local folders are copied as-is into the local registry checkout.
- Remote git sources are cloned into the local registry checkout.
- `ref=` is supported for remote git sources only.

Local channel aliases follow these rules:

- `local_name` can be provided explicitly, or derived from
  `channel.default_alias`, or finally from `channel.name`.
- Local aliases must be unique within the local registry.
- Local aliases may contain letters, numbers, `_`, and `-`.
- Local aliases cannot contain `.` because canonical skill ids use
  `<local_name>.<skill_name>`.
- The alias `hiperhealth` is reserved for built-in skills.

### Full Python API

The public registry API is intended to be comfortable in scripts and notebooks:

```python
from hiperhealth.pipeline import SkillRegistry

registry = SkillRegistry()

registry.add_channel(
    'https://github.com/my-org/traditional-medicine.git',
    local_name='tm',
    ref='main',
)
registry.add_channel('/srv/system-x/channels/traditional-medicine')

registry.list_channels()
registry.list_channel_skills('tm')
registry.update_channel('tm')
registry.update_channel('tm', ref='main')
registry.remove_channel('tm')

registry.list_skills()
registry.list_skills(channel='tm')
registry.list_skills(channel='tm', installed_only=True)

registry.install_skill('tm.ayurveda')
registry.install_channel('tm')
registry.install_channel('tm', include_disabled=True)
registry.update_skill('tm.ayurveda')
registry.update_skill('tm.ayurveda', pull_channel=True)
registry.remove_skill('tm.ayurveda')
registry.load('tm.ayurveda')
```

`StageRunner.register()` uses the same canonical ids:

```python
from hiperhealth.pipeline import StageRunner, create_default_runner

runner = create_default_runner()
runner.register('tm.ayurveda', index=0)
```

Built-in skills continue to use the built-in canonical ids:

```python
runner.register('hiperhealth.privacy')
runner.register('hiperhealth.extraction')
runner.register('hiperhealth.diagnostics')
```

External channel skills must be installed before `load()` or
`StageRunner.register()` can activate them.

### CLI

The CLI is a thin wrapper over the same registry API:

```bash
hiperhealth channel add https://github.com/my-org/traditional-medicine.git --name tm
hiperhealth channel add /srv/system-x/channels/traditional-medicine --name tm
hiperhealth channel list
hiperhealth channel skills tm
hiperhealth channel update tm
hiperhealth channel update tm --ref main
hiperhealth channel remove tm
hiperhealth channel install tm --all
hiperhealth channel install tm --all --include-disabled
hiperhealth skill list --channel tm
hiperhealth skill install tm.ayurveda
hiperhealth skill update tm.ayurveda --pull
hiperhealth skill remove tm.ayurveda
```

### Update semantics

Channel and skill updates are intentionally separate:

- `registry.update_channel('tm')` refreshes the local channel checkout from its
  original source, then refreshes every currently installed skill from that
  channel.
- `registry.update_skill('tm.ayurveda')` refreshes one installed skill against
  the current local checkout of channel `tm` without first refreshing the
  channel source.
- `registry.update_skill('tm.ayurveda', pull_channel=True)` first refreshes the
  owning channel checkout, then updates the installed skill metadata.
- `hiperhealth channel update tm` is the CLI equivalent of
  `registry.update_channel('tm')`.
- `hiperhealth skill update tm.ayurveda --pull` is the CLI equivalent of
  `registry.update_skill('tm.ayurveda', pull_channel=True)`.

### Validation and error behavior

The registry enforces these rules:

- All skill names in a channel must be unique within that channel.
- Every declared skill must have a matching `skills/<name>/` directory.
- Every declared skill must have a matching `skills/<name>/skill.yaml`.
- Each declared skill manifest must be a valid `skill.yaml`.
- Built-in skills cannot be installed or removed with channel skill commands.
- Channel skill operations use canonical ids such as `tm.ayurveda`.
- Skill names only need to be unique within a channel, not globally.

### Local storage layout

```text
~/.hiperhealth/
├── registry/
│   ├── channels.json
│   ├── skills.json
│   └── state.json
├── channels/
│   └── tm/
│       ├── repo/
│       ├── skills-channel.yaml
│       └── channel.json
└── artifacts/
    └── skills/
```

Each registered channel keeps a single local checkout under
`~/.hiperhealth/channels/<local_name>/repo`. Installed channel skills reference
that checkout instead of copying the full repository per skill.

### Channel-only registry sources

The external registry flow is channel-based:

- Use `registry.add_channel(...)` for either a local folder or a remote git
  source.
- Use `registry.install_skill(...)` or `registry.install_channel(...)` to
  activate skills from that channel.
- The channel root must contain `skills-channel.yaml`.
- Each skill folder must contain `skill.yaml`.

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

### Temporarily disable skills

Use `runner.disabled(...)` when you want to compare a stage with and without
specific registered skills, without uninstalling them or changing the registry:

```python
with runner.disabled({'tm.ayurveda'}):
    ctx_without_ayurveda = runner.run(Stage.TREATMENT, ctx)

ctx_with_ayurveda = runner.run(Stage.TREATMENT, ctx)
```

For one-off calls, `run()`, `run_many()`, `run_session()`, and
`check_requirements()` also accept `disabled_skills=`.

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

With `skill.yaml`:

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

# Register the channel and install the skill
registry = SkillRegistry()
registry.add_channel('/path/to/my_clinic_channel_repo', local_name='clinic')
registry.install_skill('clinic.bmi_calculator')

# Use it in a pipeline
runner = create_default_runner()
runner.register('clinic.bmi_calculator')

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
