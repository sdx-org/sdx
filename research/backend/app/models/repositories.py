"""Repositories for reading and saving the web app data."""

from datetime import datetime
from typing import Any, Dict, List
from uuid import UUID

from app.models.ui import (
    Consultation,
    ConsultationDiagnosis,
    ConsultationExam,
    Diagnosis,
    Exam,
    Patient,
)
from schema.ui import ConsultationCreate, PatientCreate
from sqlalchemy.orm import Session

# Supported language codes
SUPPORTED_LANGS: set[str] = {'en', 'es', 'pt'}


def _parse_iso_datetime(value: object) -> datetime | None:
    """Parse ISO 8601 timestamp safely.

    Parameters
    ----------
        value: Object to parse (typically a string).

    Returns
    -------
        Parsed datetime or None if value is None, not a string, or invalid.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _normalize_lang(value: object) -> str | None:
    """Normalize and validate ISO 639-1 language code."""
    if not isinstance(value, str):
        return None
    code = value.strip().lower()
    # Accept codes like 'pt-BR' or 'pt_BR' and take the primary subtag
    for sep in ('-', '_'):
        if sep in code:
            code = code.split(sep, 1)[0]
            break

    if code in SUPPORTED_LANGS:
        return code

    # unsupported or invalid codes -> explicit fallback to None
    return None


class ResearchRepository:
    """
    Handle all database operations for the research application.

    Manages patients, consultations, and their related data.
    """

    def __init__(self, db_session: Session):
        """Initialize the repository with a database session."""
        self.db = db_session

    def get_patient_by_uuid(self, patient_uuid: UUID) -> Patient | None:
        """Retrieve a single patient by their UUID."""
        return (
            self.db.query(Patient)
            .filter(Patient.uuid == str(patient_uuid))
            .first()
        )

    def list_patients(self) -> List[Patient]:
        """List all patients in the database."""
        return self.db.query(Patient).all()

    def create_patient_and_consultation(
        self, patient_data: Dict[str, Any]
    ) -> Patient:
        """Create a new patient and their initial consultation record."""
        patient_schema = PatientCreate(
            uuid=patient_data['meta']['uuid'],
            age=patient_data['patient'].get('age'),
            gender=patient_data['patient'].get('gender'),
        )
        new_patient = Patient(**patient_schema.model_dump())
        self.db.add(new_patient)
        self.db.commit()
        self.db.refresh(new_patient)

        # Parse the timestamp string safely as ISO 8601.
        timestamp_str = patient_data['meta'].get('timestamp')
        timestamp_obj = _parse_iso_datetime(timestamp_str)

        # Normalize language before persisting.
        # Only supported codes (en, es, pt) will be saved.
        normalized_lang = _normalize_lang(
            patient_data.get('meta', {}).get('lang')
        )
        consultation_schema = ConsultationCreate(
            patient_id=new_patient.id,
            timestamp=timestamp_obj,
            lang=normalized_lang,
            **patient_data['patient'],
        )
        new_consultation = Consultation(
            **consultation_schema.model_dump(exclude_unset=True)
        )
        self.db.add(new_consultation)
        self.db.commit()

        self.db.refresh(new_patient)
        return new_patient

    def update_consultation(
        self, patient_uuid: UUID, full_patient_record: Dict[str, Any]
    ) -> Patient | None:
        """Update the comprehensive record for a patient's consultation."""
        patient = self.get_patient_by_uuid(patient_uuid)
        if not patient:
            return None

        consultation = (
            patient.consultations[-1] if patient.consultations else None
        )
        if not consultation:
            consultation = Consultation(patient_id=patient.id)
            self.db.add(consultation)

        consultation_data = full_patient_record.get('patient', {})
        meta_data: Dict[str, Any] | None = full_patient_record.get('meta')

        for key, value in consultation_data.items():
            if hasattr(consultation, key):
                setattr(consultation, key, value)

        # Parse and update timestamp: trim whitespace, clear on empty/None.
        if meta_data and isinstance(meta_data, dict):
            if 'timestamp' in meta_data:
                raw_ts = meta_data.get('timestamp')
                # Trim whitespace; empty or None clears the timestamp
                if isinstance(raw_ts, str):
                    raw_ts = raw_ts.strip()
                if not raw_ts:
                    # Empty/whitespace-only/None clears timestamp
                    consultation.timestamp = None
                else:
                    # Parse the trimmed value
                    ts = _parse_iso_datetime(raw_ts)
                    if ts is not None:
                        consultation.timestamp = ts

            # Update language consistently (even if None) for predictability
            lang = _normalize_lang(meta_data.get('lang'))
            if consultation.lang != lang:
                consultation.lang = lang

        consultation.ai_diag_raw = full_patient_record.get('ai_diag')
        consultation.ai_exam_raw = full_patient_record.get('ai_exam')

        # Clear old evaluation data to prevent duplicates
        self.db.query(ConsultationDiagnosis).filter(
            ConsultationDiagnosis.consultation_id == consultation.id
        ).delete()
        self.db.query(ConsultationExam).filter(
            ConsultationExam.consultation_id == consultation.id
        ).delete()

        evaluations = full_patient_record.get('evaluations', {})
        if (
            'ai_diag' in evaluations
            and 'selected_diagnoses' in full_patient_record
        ):
            for diag_name in full_patient_record['selected_diagnoses']:
                diagnosis_obj = self.get_or_create_diagnosis(diag_name)
                rating_obj = evaluations.get('ai_diag', {}).get(diag_name)
                if rating_obj:
                    rating_dict = (
                        rating_obj.model_dump()
                        if hasattr(rating_obj, 'model_dump')
                        else rating_obj
                    )
                    eval_data = (
                        rating_obj.get('ratings', {})
                        if 'ratings' in rating_dict
                        else rating_dict
                    )
                else:
                    eval_data = {}

                assoc = ConsultationDiagnosis(
                    consultation_id=consultation.id,
                    diagnosis_id=diagnosis_obj.id,
                    **eval_data,
                )
                self.db.add(assoc)

        if (
            'ai_exam' in evaluations
            and 'selected_exams' in full_patient_record
        ):
            for exam_name in full_patient_record['selected_exams']:
                exam_obj = self.get_or_create_exam(exam_name)
                rating_obj = evaluations.get('ai_exam', {}).get(exam_name)
                if rating_obj:
                    rating_dict = (
                        rating_obj.model_dump()
                        if hasattr(rating_obj, 'model_dump')
                        else rating_obj
                    )
                    eval_data = (
                        rating_obj.get('ratings', {})
                        if 'ratings' in rating_dict
                        else rating_dict
                    )
                else:
                    eval_data = {}

                assoc = ConsultationExam(
                    consultation_id=consultation.id,
                    exam_id=exam_obj.id,
                    **eval_data,
                )
                self.db.add(assoc)

        self.db.commit()
        self.db.refresh(patient)
        return patient

    def get_or_create_diagnosis(self, diagnosis_name: str) -> Diagnosis:
        """Find a diagnosis by name or create it if it does not exist."""
        db_diagnosis = (
            self.db.query(Diagnosis)
            .filter(Diagnosis.name == diagnosis_name)
            .first()
        )
        if db_diagnosis:
            return db_diagnosis

        new_diagnosis = Diagnosis(name=diagnosis_name)
        self.db.add(new_diagnosis)
        self.db.commit()
        self.db.refresh(new_diagnosis)
        return new_diagnosis

    def get_or_create_exam(self, exam_name: str) -> Exam:
        """Find an exam by name or create it if it does not exist."""
        db_exam = self.db.query(Exam).filter(Exam.name == exam_name).first()
        if db_exam:
            return db_exam

        new_exam = Exam(name=exam_name)
        self.db.add(new_exam)
        self.db.commit()
        self.db.refresh(new_exam)
        return new_exam

    def delete_patient(self, patient_uuid: UUID) -> bool:
        """Delete a patient record by their UUID."""
        patient = self.get_patient_by_uuid(patient_uuid)
        if patient:
            self.db.delete(patient)
            self.db.commit()
            return True
        return False
