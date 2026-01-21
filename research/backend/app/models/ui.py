"""SQLAlchemy models for research application."""

from hiperhealth.models.sqla.fhirx import (
    Base,
)
from sqlalchemy import (
    JSON,
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
