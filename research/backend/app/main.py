"""
FastAPI application exposing a multi-step consultation wizard for physicians.

The workflow reproduces *exactly* the CLI steps:

1. Demographics
2. Lifestyle
3. Symptoms
4. Mental health
5. Previous tests
6. AI differential diagnosis → physician selects
7. AI exam suggestions → physician selects
8. Persist record & show confirmation

This refactored version ensures data is persisted at each step using a
repository pattern, preventing data loss on server restart. State is
derived from the patient data itself.
"""

# ruff: noqa: E402
import io
import logging
import uuid

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).resolve().parents[3] / '.envs' / '.env'
load_dotenv(env_path)

from app.database import SessionLocal
from app.models.repositories import ResearchRepository
from app.models.ui import Patient
from app.reports import (
    load_fhir_reports,
    process_uploaded_reports,
    save_fhir_reports,
)
from app.schemas import (
    ConsultationStatusResponse,
    CreatePatientRequest,
    CreatePatientResponse,
    DeleteResponse,
    DemographicsRequest,
    DiagnosisGetResponse,
    DiagnosisOption,
    DiagnosisSubmitRequest,
    DiagnosisSubmitResponse,
    ExamGetResponse,
    ExamOption,
    ExamSubmitRequest,
    ExamSubmitResponse,
    HealthResponse,
    LifestyleRequest,
    MedicalReportListResponse,
    MedicalReportSkipResponse,
    MedicalReportUploadResponse,
    MentalHealthRequest,
    PatientSummary,
    ReportSummary,
    StepResponse,
    SymptomsRequest,
    WearableDataSkipResponse,
    WearableDataUploadResponse,
)
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from hiperhealth.agents.diagnostics import core as diag
from hiperhealth.agents.extraction.medical_reports import (
    MedicalReportFileExtractor,
)
from hiperhealth.agents.extraction.wearable import WearableDataFileExtractor
from hiperhealth.privacy.deidentifier import (
    Deidentifier,
    deidentify_patient_record,
)
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).parent


# --- Database Dependency Setup ---
def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@lru_cache(maxsize=None)
def get_deidentifier() -> Deidentifier:
    """Get a cached deidentifier instance."""
    return Deidentifier()


def get_repository(
    db: Session = Depends(get_db),
) -> ResearchRepository:
    """Get a repository instance with a database session."""
    return ResearchRepository(db_session=db)


# --- App Initialization ---
_STATIC = StaticFiles(directory=APP_DIR / 'static')
app = FastAPI(title='TeleHealthCareAI Physician Portal API')
app.mount('/static', _STATIC, name='static')


app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://localhost:3000', '*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# --- Helper Functions ---
def extract_medical_reports_for_ui(consultation) -> List[Dict[str, Any]]:
    """Help to format reports for the UI."""
    formatted_reports = []
    fhir_reports = load_fhir_reports(consultation)
    for report in fhir_reports:
        if isinstance(report, dict):
            file_name = report.get('filename', 'unknown')
            resource_keys = [
                k for k in report.keys() if k[0].isupper() and k != 'filename'
            ]
            if resource_keys:
                resource_type = ', '.join(sorted(resource_keys))
            else:
                resource_type = 'Bundle'

            formatted_reports.append(
                {
                    'file_name': file_name,
                    'resource_type': resource_type,
                    'fhir_content': report,
                }
            )
        else:
            formatted_reports.append(
                {
                    'filename': 'Unknown',
                    'resource_type': 'Unknown',
                    'fhir_content': report,
                }
            )
    return formatted_reports


def patient_to_dict(patient: Patient) -> Dict[str, Any]:
    """Convert a Patient ORM object to a dictionary for template rendering."""
    if not patient:
        return {}

    consultation = patient.consultations[-1] if patient.consultations else None

    patient_dict = {
        'meta': {
            'uuid': patient.uuid,
            'lang': consultation.lang if consultation else None,
            'timestamp': (
                consultation.timestamp.isoformat()
                if consultation and consultation.timestamp
                else None
            ),
        },
        'patient': {
            'age': patient.age,
            'gender': patient.gender,
        },
        'selected_diagnoses': [
            assoc.diagnosis.name for assoc in consultation.selected_diagnoses
        ]
        if consultation
        else [],
        'selected_exams': [
            assoc.exam.name for assoc in consultation.selected_exams
        ]
        if consultation
        else [],
        'ai_diag': consultation.ai_diag_raw if consultation else {},
        'ai_exam': consultation.ai_exam_raw if consultation else {},
        'evaluations': {
            'ai_diag': {
                assoc.diagnosis.name: {
                    'ratings': {
                        'accuracy': assoc.accuracy,
                        'relevance': assoc.relevance,
                        'usefulness': assoc.usefulness,
                        'coherence': assoc.coherence,
                        'comments': assoc.comments,
                    }
                }
                for assoc in consultation.selected_diagnoses
            }
            if consultation
            else {},
            'ai_exam': {
                assoc.exam.name: {
                    'ratings': {
                        'accuracy': assoc.accuracy,
                        'relevance': assoc.relevance,
                        'usefulness': assoc.usefulness,
                        'coherence': assoc.coherence,
                        'safety': assoc.safety,
                        'comments': assoc.comments,
                    }
                }
                for assoc in consultation.selected_exams
            }
            if consultation
            else {},
        },
    }

    if consultation:
        # consultation fields
        consultation_fields = [
            'weight_kg',
            'height_cm',
            'diet',
            'sleep_hours',
            'physical_activity',
            'mental_exercises',
            'symptoms',
            'mental_health',
            'previous_tests',
            'wearable_data',
        ]
        for field in consultation_fields:
            if hasattr(consultation, field):
                patient_dict['patient'][field] = getattr(consultation, field)

    return patient_dict


def _get_next_step(patient: Patient) -> str:
    """Determine the next step by checking for missing data."""
    if not patient.consultations:
        return 'demographics'

    consultation = patient.consultations[-1]

    if patient.age is None:
        return 'demographics'
    if consultation.diet is None:
        return 'lifestyle'
    if consultation.symptoms is None:
        return 'symptoms'
    if consultation.mental_health is None:
        return 'mental'
    if consultation.previous_tests is None:
        return 'medical-reports'
    if consultation.wearable_data is None:
        return 'wearable-data'
    if not consultation.selected_diagnoses:
        return 'diagnosis'
    if not consultation.selected_exams:
        return 'exams'
    return 'confirmation'


# --- FastAPI Endpoints ---


# Health check
@app.get('/api/health', response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status='ok',
        service='TeleHealthCareAI Physician Portal API',
    )


# Create a new patient
@app.post('/api/patients', response_model=CreatePatientResponse)
def create_new_patient(
    req: CreatePatientRequest,
    repo: ResearchRepository = Depends(get_repository),
):
    """Create a new patient and start a new consultation."""
    patient_uuid = str(uuid.uuid4())
    new_patient_record = {
        'meta': {'uuid': patient_uuid, 'lang': req.lang},
        'patient': {},
    }
    repo.create_patient_and_consultation(new_patient_record)
    return CreatePatientResponse(
        patient_id=patient_uuid,
        lang=req.lang,
        created_at=datetime.utcnow().isoformat(),
    )


# Get all patients
@app.get('/api/patients', response_model=List[PatientSummary])
def get_all_patients(repo: ResearchRepository = Depends(get_repository)):
    """Get list of all patients for dashboard."""
    patients = repo.list_patients()
    patients_data = []
    for p in patients:
        next_step = _get_next_step(p)
        patients_data.append(
            PatientSummary(
                patient_id=p.uuid,
                created_at=p.consultations[-1].timestamp.isoformat()
                if p.consultations and p.consultations[-1].timestamp
                else None,
                current_step=next_step,
                is_complete=next_step == 'confirmation',
            )
        )
    return patients_data


# Delete a patient
@app.delete('/api/patients/{patient_id}', response_model=DeleteResponse)
def delete_patient(
    patient_id: str,
    repo: ResearchRepository = Depends(get_repository),
):
    """Delete a patient record."""
    success = repo.delete_patient(patient_id)
    if not success:
        raise HTTPException(status_code=404, detail='Patient not found')
    return DeleteResponse(success=True, patient_id=patient_id)


# Get consultation status
@app.get(
    '/api/consultations/{patient_id}/status',
    response_model=ConsultationStatusResponse,
)
def get_consultation_status(
    patient_id: str, repo: ResearchRepository = Depends(get_repository)
):
    """Get the current step and all patient data."""
    patient = repo.get_patient_by_uuid(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail='Patient not found')

    record = patient_to_dict(patient)
    next_step = _get_next_step(patient)

    completed_steps = []
    if patient.age is not None:
        completed_steps.append('demographics')
    if patient.consultations:
        c = patient.consultations[-1]
        if c.diet:
            completed_steps.append('lifestyle')
        if c.symptoms:
            completed_steps.append('symptoms')
        if c.mental_health:
            completed_steps.append('mental')
        if c.previous_tests is not None:
            completed_steps.append('medical-reports')
        if c.wearable_data is not None:
            completed_steps.append('wearable-data')
        if c.selected_diagnoses:
            completed_steps.append('diagnosis')
        if c.selected_exams:
            completed_steps.append('exams')

    return ConsultationStatusResponse(
        patient_id=patient_id,
        current_step=next_step,
        completed_steps=completed_steps,
        is_complete=next_step == 'confirmation',
        patient_dict=record,
        lang=record.get('meta', {}).get('lang', 'en'),
    )


# demographics
@app.post(
    '/api/consultations/{patient_id}/demographics', response_model=StepResponse
)
def submit_demographics(
    patient_id: str,
    data: DemographicsRequest,
    repo: ResearchRepository = Depends(get_repository),
):
    """Save demographics and return next step."""
    patient = repo.get_patient_by_uuid(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail='Patient not found')

    patient.age = data.age
    patient.gender = data.gender
    consultation = patient.consultations[-1]
    consultation.weight_kg = data.weight_kg
    consultation.height_cm = data.height_cm
    repo.db.commit()

    next_step = _get_next_step(patient)
    return StepResponse(
        success=True, next_step=next_step, patient_id=patient_id
    )


# Lifestyle
@app.post(
    '/api/consultations/{patient_id}/lifestyle', response_model=StepResponse
)
def submit_lifestyle(
    patient_id: str,
    data: LifestyleRequest,
    repo: ResearchRepository = Depends(get_repository),
):
    """Save lifestyle data and return next step."""
    patient = repo.get_patient_by_uuid(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail='Patient not found')

    consultation = patient.consultations[-1]
    consultation.diet = data.diet
    consultation.sleep_hours = data.sleep_hours
    consultation.physical_activity = data.physical_activity
    consultation.mental_exercises = data.mental_exercises
    repo.db.commit()

    next_step = _get_next_step(patient)
    return StepResponse(
        success=True, next_step=next_step, patient_id=patient_id
    )


# Symptoms
@app.post(
    '/api/consultations/{patient_id}/symptoms', response_model=StepResponse
)
def submit_symptoms(
    patient_id: str,
    data: SymptomsRequest,
    repo: ResearchRepository = Depends(get_repository),
):
    """Save symptoms data and return next step."""
    patient = repo.get_patient_by_uuid(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail='Patient not found')

    consultation = patient.consultations[-1]
    consultation.symptoms = data.symptoms
    repo.db.commit()

    next_step = _get_next_step(patient)
    return StepResponse(
        success=True, next_step=next_step, patient_id=patient_id
    )


# Mental Health
@app.post(
    '/api/consultations/{patient_id}/mental', response_model=StepResponse
)
def submit_mental_health(
    patient_id: str,
    data: MentalHealthRequest,
    repo: ResearchRepository = Depends(get_repository),
):
    """Save mental health data and return next step."""
    patient = repo.get_patient_by_uuid(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail='Patient not found')

    consultation = patient.consultations[-1]
    consultation.mental_health = data.mental_health
    repo.db.commit()

    next_step = _get_next_step(patient)
    return StepResponse(
        success=True, next_step=next_step, patient_id=patient_id
    )


# Upload Previous Medical Reports
@app.post(
    '/api/consultations/{patient_id}/medical-reports/upload',
    response_model=MedicalReportUploadResponse,
)
async def upload_medical_reports(
    patient_id: str,
    files: List[UploadFile] = File(...),
    repo: ResearchRepository = Depends(get_repository),
):
    """Upload and process previous medical reports."""
    patient = repo.get_patient_by_uuid(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail='Patient not found')

    consultation = patient.consultations[-1]

    # load existing reports
    fhir_reports = load_fhir_reports(consultation)
    seen_filenames = {
        r.get('filename', '').lower()
        for r in fhir_reports
        if isinstance(r, dict) and 'filename' in r
    }

    # extract FHIR from uploaded reports
    extractor = MedicalReportFileExtractor()
    new_reports, error = await process_uploaded_reports(
        files, seen_filenames, extractor
    )

    if error:
        raise HTTPException(status_code=400, detail=error)

    fhir_reports.extend(new_reports)

    try:
        save_fhir_reports(consultation, fhir_reports, repo)
    except Exception as e:
        raise HTTPException(
            status_code=422, detail=f'Report validation failed: {e}'
        )

    next_step = _get_next_step(patient)
    return MedicalReportUploadResponse(
        success=True,
        uploaded_files=[r.filename for r in files],
        total_reports=len(fhir_reports),
        next_step=next_step,
    )


# Skip Uploading Medical Report Step
@app.post(
    '/api/consultations/{patient_id}/medical-reports/skip',
    response_model=MedicalReportSkipResponse,
)
def skip_medical_reports(
    patient_id: str,
    repo: ResearchRepository = Depends(get_repository),
):
    """Skip medical reports upload step."""
    patient = repo.get_patient_by_uuid(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail='Patient not found')

    consultation = patient.consultations[-1]
    save_fhir_reports(consultation, [], repo)

    next_step = _get_next_step(patient)
    return MedicalReportSkipResponse(
        success=True, skipped=True, next_step=next_step
    )


# Get all the previous medical Reports
@app.get(
    '/api/consultations/{patient_id}/medical-reports',
    response_model=MedicalReportListResponse,
)
def get_medical_reports(
    patient_id: str,
    repo: ResearchRepository = Depends(get_repository),
):
    """Get summary of uploaded medical reports."""
    patient = repo.get_patient_by_uuid(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail='Patient not found')

    consultation = patient.consultations[-1]
    formatted_reports = extract_medical_reports_for_ui(consultation)

    reports_summary = [
        ReportSummary(
            file_name=report.get('file_name')
            or report.get('filename', 'Unknown'),
            resource_type=report.get('resource_type', 'Unknown'),
            fhir_content=report.get('fhir_content', 'Unknown'),
        )
        for report in formatted_reports
    ]

    return MedicalReportListResponse(
        total_reports=len(reports_summary), reports=reports_summary
    )


# Wearable data Upload
@app.post(
    '/api/consultations/{patient_id}/wearable-data/upload',
    response_model=WearableDataUploadResponse,
)
async def upload_wearable_data(
    patient_id: str,
    file: UploadFile = File(...),
    repo: ResearchRepository = Depends(get_repository),
):
    """Upload and Process wearable data."""
    patient = repo.get_patient_by_uuid(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detai='Patient not found')

    consultation = patient.consultations[-1]

    if not file or file.size == 0:
        raise HTTPException(status_code=400, detail='No file provided')

    extractor = WearableDataFileExtractor()
    try:
        file_content = await file.read()
        wearable_data = extractor.extract_wearable_data(
            io.BytesIO(file_content)
        )
        consultation.wearable_data = wearable_data
        repo.db.commit()

        next_step = _get_next_step(patient)
        return WearableDataUploadResponse(
            success=True, file_name=file.filename, next_step=next_step
        )
    except Exception as e:
        raise HTTPException(
            status_code=422, detail=f'Failed to process wearable data:{e}'
        )


# Wearable data upload skip
@app.post(
    '/api/consultations/{patient_id}/wearable-data/skip',
    response_model=WearableDataSkipResponse,
)
def skip_wearable_data(
    patient_id: str,
    repo: ResearchRepository = Depends(get_repository),
):
    """Skip wearable data upload."""
    patient = repo.get_patient_by_uuid(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail='Patient not found')
    consultation = patient.consultations[-1]
    consultation.wearable_data = []
    repo.db.commit()

    next_step = _get_next_step(patient)
    return WearableDataSkipResponse(
        success=True, skipped=True, next_step=next_step
    )


# AI diagnosis
@app.get(
    '/api/consultations/{patient_id}/diagnosis',
    response_model=DiagnosisGetResponse,
)
def get_diagnosis_suggestions(
    patient_id: str, repo: ResearchRepository = Depends(get_repository)
):
    """Get AI-generated differential diagnosis suggestions."""
    patient = repo.get_patient_by_uuid(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail='Patient not found')

    record = patient_to_dict(patient)
    lang = record['meta']['lang']

    # Call AI diagnostic engine
    ai = diag.differential(
        record['patient'], language=lang, session_id=patient_id
    )

    consultation = patient.consultations[-1]
    consultation.ai_diag_raw = ai.model_dump()
    repo.db.commit()

    if isinstance(ai.options, list):
        diagnosis_options = [
            DiagnosisOption(name=opt, description='') for opt in ai.options
        ]
    else:
        diagnosis_options = [
            DiagnosisOption(name=name, description='')
            for name in ai.options.keys()
        ]
    return DiagnosisGetResponse(
        patient_id=patient_id, summary=ai.summary, options=diagnosis_options
    )


# AI diagnosis selected
@app.post(
    '/api/consultations/{patient_id}/diagnosis',
    response_model=DiagnosisSubmitResponse,
)
def submit_diagnosis_selection(
    patient_id: str,
    req: DiagnosisSubmitRequest,
    repo: ResearchRepository = Depends(get_repository),
):
    """Save selected diagnosis and evaluations."""
    patient = repo.get_patient_by_uuid(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail='Patient not found')

    record = patient_to_dict(patient)
    record['selected_diagnoses'] = req.selected_diagnoses
    record['evaluations'] = {'ai_diag': req.evaluations, 'ai_exam': {}}

    repo.update_consultation(patient_id, record)
    next_step = _get_next_step(patient)
    return DiagnosisSubmitResponse(
        success=True, next_step=next_step, patient_id=patient_id
    )


# AI exams
@app.get(
    '/api/consultations/{patient_id}/exams', response_model=ExamGetResponse
)
def get_exam_suggestions(
    patient_id: str, repo: ResearchRepository = Depends(get_repository)
):
    """Get AI-generated exam suggestions."""
    patient = repo.get_patient_by_uuid(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail='Patient not found')

    record = patient_to_dict(patient)
    lang = record['meta']['lang']
    selected_diagnoses = record.get('selected_diagnoses', [])

    ai = diag.exams(selected_diagnoses, language=lang, session_id=patient_id)

    consultation = patient.consultations[-1]
    consultation.ai_exam_raw = ai.model_dump()
    repo.db.commit()

    if isinstance(ai.options, list):
        exam_options = [
            ExamOption(name=opt, description='') for opt in ai.options
        ]
    else:
        exam_options = [
            ExamOption(name=name, description='') for name in ai.options.keys()
        ]
    return ExamGetResponse(
        patient_id=patient_id, summary=ai.summary, options=exam_options
    )


# Upload selected exams rating
@app.post(
    '/api/consultations/{patient_id}/exams', response_model=ExamSubmitResponse
)
def submit_exams_selection(
    patient_id: str,
    req: ExamSubmitRequest,
    deidentifier: Deidentifier = Depends(get_deidentifier),
    repo: ResearchRepository = Depends(get_repository),
):
    """Save selected exams and finalize record with deidentification."""
    patient = repo.get_patient_by_uuid(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail='Patient not found')

    record = patient_to_dict(patient)
    record['selected_exams'] = req.selected_exams
    record['evaluations']['ai_exam'] = req.evaluations
    record['meta']['timestamp'] = datetime.utcnow().isoformat()

    deidentified_record = deidentify_patient_record(record, deidentifier)
    repo.update_consultation(patient_id, deidentified_record)

    return ExamSubmitResponse(
        success=True, patient_id=patient_id, is_complete=True
    )
