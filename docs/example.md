# Gut Leak Patient — Full Pipeline Example

# Gut Leak Patient: End-to-End Pipeline Example

This example walks through a complete clinical workflow for a patient presenting
with gut leak symptoms. It demonstrates all six pipeline stages, the
session-based workflow with requirement checking, and the multi-visit pattern
where deferred data (lab results) arrives later.

## Setup

```python
from pathlib import Path
from dotenv import load_dotenv

# Load env vars — try both project root and docs/ as possible working dirs
load_dotenv(Path("tests/.env")) or load_dotenv(Path("../tests/.env"))

from hiperhealth.pipeline import (
    Session,
    Stage,
    create_default_runner,
)
```

## Create a session and runner

The session file is the single source of truth for all interactions. It is a
parquet-backed event log — every action is recorded as a row.

```python
import tempfile
from pathlib import Path

session_dir = Path(tempfile.mkdtemp())
session_path = session_dir / "gut-leak-visit.parquet"

session = Session.create(session_path, language="en")
runner = create_default_runner()
```

## Stage 1 — Screening

The screening stage performs initial triage and de-identifies any PII in the
patient record. The clinical system provides data without personally
identifiable information — only clinical content.

```python
session.set_clinical_data({
    "symptoms": (
        "Patient reports chronic bloating, abdominal discomfort after meals, "
        "fatigue, brain fog, and intermittent diarrhea for the past 6 months. "
        "Symptoms worsen with gluten and dairy intake."
    ),
    "age": 38,
    "biological_sex": "female",
    "medical_history": "Irritable bowel syndrome diagnosed 3 years ago",
    "medications": "Omeprazole 20mg daily",
    "allergies": "None known",
})
```

Run screening to de-identify any PII that may have been inadvertently included:

```python
runner.run_session(Stage.SCREENING, session)
print("Screening complete.")
print("Stages completed:", session.stages_completed)
```

    Screening complete.
    Stages completed: [<Stage.SCREENING: 'screening'>]

## Stage 2 — Intake

The intake stage extracts structured data from medical reports and wearable
files. In this example we simulate providing a previous lab report.

```python
# In a real workflow, you would provide file paths:
# session.provide_answers({"lab_report": "/path/to/previous_labs.pdf"})

# For this example, we provide the extracted data directly
session.provide_answers({
    "previous_labs": {
        "CBC": "within normal limits",
        "CRP": "slightly elevated (5.2 mg/L)",
        "vitamin_D": "low (18 ng/mL)",
        "iron": "low-normal (45 mcg/dL)",
        "ferritin": "low (15 ng/mL)",
    },
})

runner.run_session(Stage.INTAKE, session)
print("Intake complete.")
```

    Intake complete.

## Stage 3 — Diagnosis (first pass)

Before running diagnosis, check what information the skills need. This is the
requirement checking cycle: assess → provide → execute.

### Check requirements

```python
inquiries = runner.check_requirements(Stage.DIAGNOSIS, session)

print(f"Total inquiries: {len(inquiries)}\n")
for inq in inquiries:
    print(f"  [{inq.priority}] {inq.field}: {inq.label}")
    if inq.description:
        print(f"    → {inq.description}")
```

    Total inquiries: 0

### Provide answers to required and supplementary inquiries

The clinical system collects answers from the patient or provider and feeds them
back into the session. Deferred inquiries (like lab results not yet ordered) are
skipped for now.

```python
session.provide_answers({
    "dietary_history": (
        "High carbohydrate diet, frequent processed foods, low fiber intake. "
        "Symptoms worsen significantly after gluten-containing meals and dairy. "
        "Patient has tried eliminating gluten for 2 weeks with partial improvement."
    ),
    "bowel_habits": (
        "2-4 loose stools daily, occasional mucus. Bristol stool scale type 5-6. "
        "Urgency after meals, especially breakfast."
    ),
    "stress_level": "High — work-related stress, poor sleep quality (5-6 hours/night)",
    "supplement_history": "Probiotics (generic, 1 month, no improvement), multivitamin",
})
```

### Re-check requirements — are we ready?

```python
inquiries = runner.check_requirements(Stage.DIAGNOSIS, session)
required = [i for i in inquiries if i.priority == "required"]

print(f"Remaining required: {len(required)}")
if not required:
    print("All required information is available — ready to run diagnosis.")
```

    Remaining required: 0
    All required information is available — ready to run diagnosis.

### Run preliminary diagnosis

This is a first-pass diagnosis with available data. Deferred information (like
stool analysis or zonulin levels) is not yet available.

```python
runner.run_session(Stage.DIAGNOSIS, session)

diagnosis = session.results.get(Stage.DIAGNOSIS, {})
print("Preliminary diagnosis complete.")
if hasattr(diagnosis, "summary"):
    print(f"\nSummary: {diagnosis.summary}")
    print(f"\nDifferential:\n{diagnosis.options}")
else:
    print(f"\nResults: {diagnosis}")
```

    Preliminary diagnosis complete.

    Results: {'summary': 'A 38-year-old female with a history of IBS presents with six months of chronic bloating, postprandial abdominal discomfort, loose stools, fatigue, and brain fog that worsen with gluten and dairy intake and have partially improved with gluten elimination. Laboratory evaluation shows low vitamin D and ferritin with mildly elevated CRP, raising concern for both functional and organic etiologies.', 'options': ['Celiac disease', 'Non-celiac gluten sensitivity', 'Lactose intolerance', 'Irritable bowel syndrome–diarrhea predominant', 'Small intestinal bacterial overgrowth', 'Inflammatory bowel disease', 'Microscopic colitis']}

## Stage 4 — Exam

The exam stage suggests laboratory tests and procedures based on the preliminary
diagnosis. This is where the system requests the deferred data.

```python
runner.run_session(Stage.EXAM, session)

exam_results = session.results.get(Stage.EXAM, {})
print("Exam suggestions complete.")
if hasattr(exam_results, "summary"):
    print(f"\nSuggested exams: {exam_results.summary}")
    print(f"\nDetails:\n{exam_results.options}")
else:
    print(f"\nResults: {exam_results}")
```

    Exam suggestions complete.

    Results: {'summary': 'The differential for chronic diarrhea includes celiac disease, non-celiac gluten sensitivity, lactose intolerance, IBS-D, small intestinal bacterial overgrowth (SIBO), inflammatory bowel disease, and microscopic colitis. Initial evaluation should target celiac serology and duodenal histology, lactose and SIBO breath testing, colonoscopy with random biopsies plus stool inflammatory markers, and imaging if IBD is suspected. Trial of dietary modifications may also aid in diagnosis.', 'options': ['Serum tissue transglutaminase IgA and total IgA', 'Anti-endomysial antibody testing', 'Upper endoscopy with duodenal biopsy', 'Lactose hydrogen breath test', 'Glucose or lactulose hydrogen breath test for SIBO', 'Colonoscopy with random colonic biopsies', 'Stool calprotectin and lactoferrin', 'CT or MR enterography', 'Fecal ova and parasite examination', 'Trial of gluten-free diet under dietitian supervision']}

## Multi-visit gap — Lab results arrive

In a real clinical workflow, days or weeks may pass while lab work is performed.
The session file persists on disk, and the system reloads it when new data
arrives.

```python
# Simulate reloading the session (as would happen days later)
session = Session.load(session_path)

# Lab results arrive from the laboratory
session.provide_answers({
    "stool_analysis": {
        "zonulin": "elevated (78 ng/mL, ref: <30)",
        "calprotectin": "mildly elevated (95 mcg/g, ref: <50)",
        "secretory_IgA": "low (42 mg/dL, ref: 70-400)",
        "parasitology": "negative",
        "occult_blood": "negative",
    },
    "food_sensitivity_panel": {
        "IgG_gluten": "highly reactive",
        "IgG_casein": "moderately reactive",
        "IgG_soy": "mildly reactive",
        "IgG_eggs": "non-reactive",
    },
    "intestinal_permeability_test": {
        "lactulose_mannitol_ratio": "elevated (0.09, ref: <0.03)",
        "interpretation": "consistent with increased intestinal permeability",
    },
})

print("Lab results recorded in session.")
print(f"Total events in session: {len(session.events)}")
```

    Lab results recorded in session.
    Total events in session: 16

## Stage 3 (re-run) — Enriched diagnosis

With lab results now available, re-run the diagnosis stage for a complete
clinical picture. The runner uses all accumulated data.

```python
# Check if any requirements are still pending
inquiries = runner.check_requirements(Stage.DIAGNOSIS, session)
deferred = [i for i in inquiries if i.priority == "deferred"]
print(f"Remaining deferred inquiries: {len(deferred)}")

# Run enriched diagnosis
runner.run_session(Stage.DIAGNOSIS, session)

diagnosis = session.results.get(Stage.DIAGNOSIS, {})
print("\nEnriched diagnosis complete (with lab results).")
if hasattr(diagnosis, "summary"):
    print(f"\nSummary: {diagnosis.summary}")
    print(f"\nDifferential:\n{diagnosis.options}")
else:
    print(f"\nResults: {diagnosis}")
```

    Remaining deferred inquiries: 0

    Enriched diagnosis complete (with lab results).

    Results: {'summary': 'A 38-year-old female presents with chronic postprandial bloating, abdominal discomfort, intermittent diarrhea, fatigue, and brain fog aggravated by gluten and dairy. Laboratory findings of elevated zonulin, calprotectin, lactulose-mannitol ratio, low secretory IgA, and mild nutrient deficiencies point toward an organic disorder beyond functional IBS.', 'options': ['Celiac disease', 'Non-celiac gluten sensitivity', 'Irritable bowel syndrome with dysbiosis', 'Small intestinal bacterial overgrowth (SIBO)', 'Lactose and casein intolerance', 'Microscopic colitis', 'Leaky gut syndrome (increased intestinal permeability)']}

## Stage 5 — Treatment

The treatment stage generates a plan based on the enriched diagnosis.

```python
# Check treatment requirements
inquiries = runner.check_requirements(Stage.TREATMENT, session)
for inq in inquiries:
    print(f"  [{inq.priority}] {inq.field}: {inq.label}")

# Provide any additional treatment-relevant answers
session.provide_answers({
    "treatment_preferences": (
        "Patient prefers integrative approach. Open to dietary changes "
        "and supplements. Wants to minimize pharmaceutical interventions."
    ),
    "budget_constraints": "Standard insurance coverage, willing to pay OOP for supplements",
})

runner.run_session(Stage.TREATMENT, session)

treatment = session.results.get(Stage.TREATMENT, {})
print("Treatment plan complete.")
if isinstance(treatment, dict):
    print(f"\nResults: {treatment}")
```

    Treatment plan complete.

    Results: {}

## Stage 6 — Prescription

The prescription stage generates specific prescriptions or supplement
recommendations based on the treatment plan.

```python
runner.run_session(Stage.PRESCRIPTION, session)

prescription = session.results.get(Stage.PRESCRIPTION, {})
print("Prescription complete.")
if isinstance(prescription, dict):
    print(f"\nResults: {prescription}")
```

    Prescription complete.

    Results: {}

## Inspect the full session

The session parquet is a standard file that can be analyzed with any data tool.

### Session summary

```python
print(f"Session file: {session.path}")
print(f"Language: {session.language}")
print(f"Stages completed: {session.stages_completed}")
print(f"Total events: {len(session.events)}")
print(f"Pending inquiries: {len(session.pending_inquiries)}")
```

    Session file: /tmp/tmppjix08r9/gut-leak-visit.parquet
    Language: en
    Stages completed: ['screening', 'intake', 'diagnosis', 'exam', <Stage.DIAGNOSIS: 'diagnosis'>, <Stage.TREATMENT: 'treatment'>, <Stage.PRESCRIPTION: 'prescription'>]
    Total events: 27
    Pending inquiries: 0

### Clinical data accumulated

```python
import json

clinical = session.clinical_data
print(json.dumps({k: v for k, v in clinical.items()
                   if k not in ("previous_labs", "stool_analysis",
                                "food_sensitivity_panel",
                                "intestinal_permeability_test")},
                  indent=2))
```

    {
      "symptoms": "Patient reports chronic bloating, abdominal discomfort after meals, fatigue, brain fog, and intermittent diarrhea for the past 6 months. Symptoms worsen with gluten and dairy intake.",
      "age": 38,
      "biological_sex": "female",
      "medical_history": "Irritable bowel syndrome diagnosed 3 years ago",
      "medications": "Omeprazole 20mg daily",
      "allergies": "None known",
      "dietary_history": "High carbohydrate diet, frequent processed foods, low fiber intake. Symptoms worsen significantly after gluten-containing meals and dairy. Patient has tried eliminating gluten for 2 weeks with partial improvement.",
      "bowel_habits": "2-4 loose stools daily, occasional mucus. Bristol stool scale type 5-6. Urgency after meals, especially breakfast.",
      "stress_level": "High \u2014 work-related stress, poor sleep quality (5-6 hours/night)",
      "supplement_history": "Probiotics (generic, 1 month, no improvement), multivitamin",
      "treatment_preferences": "Patient prefers integrative approach. Open to dietary changes and supplements. Wants to minimize pharmaceutical interventions.",
      "budget_constraints": "Standard insurance coverage, willing to pay OOP for supplements"
    }

### Event log with polars

```python
import polars as pl

df = pl.read_parquet(session_path)

# All events
print(df.select("event_id", "event_type", "stage", "skill_name"))

# Filter to stage completions
completed = df.filter(pl.col("event_type") == "stage_completed")
print(completed.select("stage", "timestamp"))

# Filter to inquiries
raised = df.filter(pl.col("event_type") == "inquiries_raised")
print(raised.select("stage", "skill_name", "data"))
```

### Event log with pandas

```python
import pandas as pd

df = pd.read_parquet(session_path)
df[["event_id", "event_type", "stage", "timestamp"]]
```

## Cleanup

```python
import shutil
shutil.rmtree(session_dir)
```

## Summary

This example demonstrated the full hiperhealth pipeline for a gut leak patient:

1.  **Screening** — PII de-identification on intake data
2.  **Intake** — structured data extraction from reports
3.  **Diagnosis (preliminary)** — first-pass differential with available data
4.  **Exam** — suggested lab tests and procedures
5.  **Diagnosis (enriched)** — re-run with lab results for a complete picture
6.  **Treatment** — integrative treatment plan
7.  **Prescription** — specific supplement and medication recommendations

Key patterns shown:

- **Requirement checking** — `check_requirements()` before execution, with three
  priority levels (required, supplementary, deferred)
- **Multi-visit workflow** — session persists across days; deferred data (lab
  results) arrives later and triggers a re-run
- **Session persistence** — parquet event log as single source of truth,
  queryable with polars/pandas/DuckDB
- **No PII** — only clinical data in the session file; the external system maps
  sessions to patients
