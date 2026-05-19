# Notebook UI

`hiperhealth.notebook` provides an optional Jupyter widget interface for the
session-backed clinical pipeline workflow.

Install the optional dependencies:

```bash
pip install "hiperhealth[notebook]"
```

Launch the UI from a notebook:

```python
from hiperhealth.notebook import ui

ui.show()
ui.show(data_dir="/my/data/path")
```

The UI follows the workflow from `docs/example.qmd`:

1. create or load a parquet-backed session
2. enter de-identified screening data
3. run screening and intake
4. check diagnosis requirements, answer inquiries, and run diagnosis
5. run exam suggestions
6. reload the same session after lab results arrive
7. rerun enriched diagnosis
8. run treatment and prescription stages
9. inspect results and the session event log

## Skill app views

Skills may declare optional app views in their `skill.yaml` metadata. The
notebook UI discovers active skill views and renders them on the **Skill apps**
page. The declaration uses JSON Schema-style `data_schema` plus JSON Forms-style
`ui_schema`, so the same metadata can be reused by future notebook or JupyterLab
renderers.

Values saved through skill app views are persisted through session events and
replayed into `PipelineContext.extras['skill_ui']` for the skill to consume.

Session files may contain sensitive clinical artifacts. Protect the configured
`data_dir` like clinical records, do not store API keys in it, and prefer
synthetic or de-identified data in examples.
