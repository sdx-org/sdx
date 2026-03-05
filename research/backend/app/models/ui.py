"""SQLAlchemy models for research application."""

from hiperhealth.models.sqla.fhirx import (
    Base,
)
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship


class Patient(Base):
    """Patient model storing core demographics."""

    __tablename__ = 'patients'
    id = Column(Integer, primary_key=True)
    uuid = Column(String(36), unique=True, nullable=False, index=True)
    age = Column(Integer)
    gender = Column(String(50))

    consultations = relationship('Consultation', back_populates='patient')
    consent = relationship(
        'PatientConsent', back_populates='patient', uselist=False
    )
    consent_audit_logs = relationship(
        'PatientConsentAuditLog', back_populates='patient'
    )


class Consultation(Base):
    """Consultation model for patient visits."""

    __tablename__ = 'consultations'
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.id'))
    timestamp = Column(DateTime)
    lang = Column(String(10))

    # Consultation-specific data
    weight_kg = Column(Float)
    height_cm = Column(Float)
    diet = Column(Text)
    sleep_hours = Column(Float)
    physical_activity = Column(Text)
    mental_exercises = Column(Text)
    symptoms = Column(Text)
    mental_health = Column(Text)

    # Store complex, semi-structured data as JSON
    previous_tests = Column(JSON)
    wearable_data = Column(JSON)
    ai_diag_raw = Column(JSON)
    ai_exam_raw = Column(JSON)

    patient = relationship(Patient, back_populates='consultations')
    selected_diagnoses = relationship(
        'ConsultationDiagnosis', back_populates='consultation'
    )
    selected_exams = relationship(
        'ConsultationExam', back_populates='consultation'
    )


class Diagnosis(Base):
    """Diagnosis model for medical conditions."""

    __tablename__ = 'diagnoses'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)


class Exam(Base):
    """Exam model for medical examinations."""

    __tablename__ = 'exams'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)


class ConsultationDiagnosis(Base):
    """Junction table linking consultations to diagnoses with evaluations."""

    __tablename__ = 'consultation_diagnoses'
    consultation_id = Column(
        Integer, ForeignKey('consultations.id'), primary_key=True
    )
    diagnosis_id = Column(
        Integer, ForeignKey('diagnoses.id'), primary_key=True
    )

    # Evaluation fields
    accuracy = Column(Integer)
    relevance = Column(Integer)
    usefulness = Column(Integer)
    coherence = Column(Integer)
    comments = Column(Text)

    consultation = relationship(
        Consultation, back_populates='selected_diagnoses'
    )
    diagnosis = relationship(Diagnosis)


class ConsultationExam(Base):
    """Junction table linking consultations to exams with evaluations."""

    __tablename__ = 'consultation_exams'
    consultation_id = Column(
        Integer, ForeignKey('consultations.id'), primary_key=True
    )
    exam_id = Column(Integer, ForeignKey('exams.id'), primary_key=True)

    # Evaluation fields
    accuracy = Column(Integer)
    relevance = Column(Integer)
    usefulness = Column(Integer)
    coherence = Column(Integer)
    safety = Column(String(50))
    comments = Column(Text)

    consultation = relationship(Consultation, back_populates='selected_exams')
    exam = relationship(Exam)


class PatientConsent(Base):
    """Consent settings with granular patient permissions."""

    __tablename__ = 'patient_consents'
    id = Column(Integer, primary_key=True)
    patient_id = Column(
        Integer, ForeignKey('patients.id'), unique=True, nullable=False
    )
    consent_version = Column(String(20), nullable=False, default='v1')

    allow_diagnostics = Column(Boolean, nullable=False, default=True)
    allow_exam_recommendations = Column(Boolean, nullable=False, default=True)
    allow_medical_reports = Column(Boolean, nullable=False, default=True)
    allow_wearable_data = Column(Boolean, nullable=False, default=True)
    allow_research_sharing = Column(Boolean, nullable=False, default=False)
    allow_recontact = Column(Boolean, nullable=False, default=False)

    granted_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime)
    updated_at = Column(DateTime, nullable=False)

    patient = relationship(Patient, back_populates='consent')
    audit_logs = relationship(
        'PatientConsentAuditLog', back_populates='consent'
    )


class PatientConsentAuditLog(Base):
    """Audit log for consent changes and permission checks."""

    __tablename__ = 'patient_consent_audit_logs'
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patients.id'), nullable=False)
    consent_id = Column(Integer, ForeignKey('patient_consents.id'))
    action = Column(String(50), nullable=False)
    actor = Column(String(100), nullable=False)
    reason = Column(Text)
    details = Column(JSON)
    created_at = Column(DateTime, nullable=False)

    patient = relationship(Patient, back_populates='consent_audit_logs')
    consent = relationship(PatientConsent, back_populates='audit_logs')
