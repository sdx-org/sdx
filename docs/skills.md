# Creating Skills

Skills are composable plugins that extend the clinical pipeline. Each skill can
affect one or more stages and is a Python class that subclasses `BaseSkill`.

## Architecture overview

```
StageRunner
    |
    +-- PrivacySkill        (screening, intake)   priority=50
    +-- ExtractionSkill     (intake)              priority=100
    +-- DiagnosticsSkill    (diagnosis, exam)     priority=100
    +-- YourCustomSkill     (diagnosis, treatment) priority=150
```

When a stage runs, the runner finds all skills that declare that stage in their
metadata, orders them by priority (lower runs first), and calls their hooks:

1. All `pre()` hooks (in priority order)
2. All `execute()` hooks (in priority order)
3. All `post()` hooks (in priority order)

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
                priority=200,
                description='Adds a greeting to the context.',
            )
        )

    def execute(self, stage, ctx):
        name = ctx.patient.get('name', 'Patient')
        ctx.extras['greeting'] = f'Welcome, {name}!'
        return ctx
```

## SkillMetadata fields

| Field         | Type              | Default   | Description                                    |
| ------------- | ----------------- | --------- | ---------------------------------------------- |
| `name`        | `str`             | required  | Unique identifier, e.g. `my_org.skill_name`    |
| `version`     | `str`             | `"0.1.0"` | Semantic version of the skill                  |
| `stages`      | `tuple[str, ...]` | `()`      | Which stages this skill participates in        |
| `priority`    | `int`             | `100`     | Execution order within a stage (lower = first) |
| `description` | `str`             | `""`      | Human-readable description                     |

## Hooks

Each skill has three hooks that are called per stage. Override only the ones you
need — the base class provides no-op defaults.

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
                priority=150,
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

The `DiagnosticsSkill` checks `ctx.extras['prompt_fragments']` and appends
matching fragments to the system prompt for each stage.

## Using the runner

### Register skills at construction time

```python
from hiperhealth.pipeline import StageRunner, Stage

runner = StageRunner(skills=[
    PrivacySkill(),
    ExtractionSkill(),
    DiagnosticsSkill(),
    AyurvedaSkill(),
])

ctx = runner.run(Stage.DIAGNOSIS, ctx)
```

### Install skills at runtime

```python
runner = create_default_runner()
runner.install(AyurvedaSkill())
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

Third-party skills can be auto-discovered if they register as Python entry
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
                priority=110,
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

Usage:

```python
from hiperhealth.pipeline import PipelineContext, Stage, create_default_runner

runner = create_default_runner()
runner.install(BMICalculatorSkill())

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
