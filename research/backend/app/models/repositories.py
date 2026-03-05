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
    PatientConsent,
    PatientConsentAuditLog,
)
from schema.ui import ConsultationCreate, PatientCreate
from sqlalchemy.orm import Session


class ResearchRepository:
    """
    Handle all database operations for the research application.

    Manages patients, consultations, and their related data.
    """

    def __init__(self, db_session: Session):
        """Initialize the repository with a database session."""
        self.db = db_session
        self._consent_keys = (
            'allow_diagnostics',
            'allow_exam_recommendations',
            'allow_medical_reports',
            'allow_wearable_data',
            'allow_research_sharing',
            'allow_recontact',
        )

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

        # Parse the timestamp string into a datetime object
        timestamp_str = patient_data['meta'].get('timestamp')
        timestamp_obj = (
            datetime.fromisoformat(timestamp_str) if timestamp_str else None
        )

        consultation_schema = ConsultationCreate(
            patient_id=new_patient.id,
            timestamp=timestamp_obj,
            lang=patient_data['meta'].get('lang'),
            **patient_data['patient'],
        )
        new_consultation = Consultation(
            **consultation_schema.model_dump(exclude_unset=True)
        )
        self.db.add(new_consultation)
        self.db.commit()

        self.get_or_create_patient_consent(
            new_patient.id, actor='system', reason='auto-created on enrollment'
        )
        self.db.refresh(new_patient)
        return new_patient

    def _create_consent_audit_log(
        self,
        patient_id: int,
        consent_id: int,
        action: str,
        actor: str,
        reason: str | None = None,
        details: Dict[str, Any] | None = None,
    ) -> None:
        """Insert a consent audit log entry."""
        entry = PatientConsentAuditLog(
            patient_id=patient_id,
            consent_id=consent_id,
            action=action,
            actor=actor,
            reason=reason,
            details=details or {},
            created_at=datetime.utcnow(),
        )
        self.db.add(entry)

    def get_or_create_patient_consent(
        self, patient_id: int, actor: str = 'system', reason: str | None = None
    ) -> PatientConsent:
        """Fetch patient consent row, creating one with defaults if needed."""
        consent = (
            self.db.query(PatientConsent)
            .filter(PatientConsent.patient_id == patient_id)
            .first()
        )
        if consent:
            return consent

        now = datetime.utcnow()
        consent = PatientConsent(
            patient_id=patient_id, granted_at=now, updated_at=now
        )
        self.db.add(consent)
        self.db.flush()
        self._create_consent_audit_log(
            patient_id=patient_id,
            consent_id=consent.id,
            action='consent_created',
            actor=actor,
            reason=reason,
            details={k: getattr(consent, k) for k in self._consent_keys},
        )
        self.db.commit()
        self.db.refresh(consent)
        return consent

    def get_patient_consent(self, patient_uuid: UUID) -> PatientConsent | None:
        """Return patient consent object."""
        patient = self.get_patient_by_uuid(patient_uuid)
        if not patient:
            return None
        return self.get_or_create_patient_consent(patient.id)

    def update_patient_consent(
        self,
        patient_uuid: UUID,
        updates: Dict[str, Any],
        actor: str,
        reason: str | None = None,
    ) -> PatientConsent | None:
        """Apply consent changes and persist an audit entry."""
        patient = self.get_patient_by_uuid(patient_uuid)
        if not patient:
            return None

        consent = self.get_or_create_patient_consent(patient.id)
        old_values = {k: getattr(consent, k) for k in self._consent_keys}

        changed: Dict[str, Dict[str, Any]] = {}
        for key in self._consent_keys:
            if key in updates and updates[key] is not None:
                new_value = bool(updates[key])
                if old_values[key] != new_value:
                    changed[key] = {'old': old_values[key], 'new': new_value}
                setattr(consent, key, new_value)

        if updates.get('revoke_all'):
            for key in self._consent_keys:
                if getattr(consent, key):
                    changed[key] = {'old': True, 'new': False}
                setattr(consent, key, False)
            consent.revoked_at = datetime.utcnow()

        if updates.get('grant_all'):
            for key in self._consent_keys:
                if not getattr(consent, key):
                    changed[key] = {'old': False, 'new': True}
                setattr(consent, key, True)
            consent.revoked_at = None
            consent.granted_at = datetime.utcnow()

        consent.updated_at = datetime.utcnow()
        if changed or reason:
            self._create_consent_audit_log(
                patient_id=patient.id,
                consent_id=consent.id,
                action='consent_updated',
                actor=actor,
                reason=reason,
                details=changed,
            )

        self.db.commit()
        self.db.refresh(consent)
        return consent

    def log_consent_access(
        self,
        patient_uuid: UUID,
        action: str,
        actor: str,
        details: Dict[str, Any] | None = None,
    ) -> None:
        """Log consent-permission checks for sensitive operations."""
        patient = self.get_patient_by_uuid(patient_uuid)
        if not patient:
            return
        consent = self.get_or_create_patient_consent(patient.id)
        self._create_consent_audit_log(
            patient_id=patient.id,
            consent_id=consent.id,
            action=action,
            actor=actor,
            details=details or {},
        )
        self.db.commit()

    def list_consent_audit_logs(
        self, patient_uuid: UUID, limit: int = 100
    ) -> List[PatientConsentAuditLog]:
        """List most recent consent audit entries for a patient."""
        patient = self.get_patient_by_uuid(patient_uuid)
        if not patient:
            return []
        return (
            self.db.query(PatientConsentAuditLog)
            .filter(PatientConsentAuditLog.patient_id == patient.id)
            .order_by(PatientConsentAuditLog.created_at.desc())
            .limit(limit)
            .all()
        )

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
        meta_data = full_patient_record.get('meta', {})

        for key, value in consultation_data.items():
            if hasattr(consultation, key):
                setattr(consultation, key, value)

        # Parse the timestamp string into a datetime object
        timestamp_str = meta_data.get('timestamp')
        consultation.timestamp = (
            datetime.fromisoformat(timestamp_str) if timestamp_str else None
        )

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
