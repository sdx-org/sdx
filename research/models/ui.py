"""SQLAlchemy models for research application."""

from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.hiperhealth.models.sqla.fhirx import (
    Base,
)

class Patient(Base):
    """Patient model storing core demographics."""

    __tablename__ = 'patients'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    uuid: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    gender: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    consultations: Mapped[List['Consultation']] = relationship('Consultation', back_populates='patient')


class Consultation(Base):
    """Consultation model for patient visits."""

    __tablename__ = 'consultations'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(Integer, ForeignKey('patients.id'), nullable=False)
    timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    lang: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)

    # Consultation-specific data
    weight_kg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    height_cm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    diet: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sleep_hours: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    physical_activity: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mental_exercises: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    symptoms: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mental_health: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Store complex, semi-structured data as JSON
    previous_tests: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    wearable_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    ai_diag_raw: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    ai_exam_raw: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    patient: Mapped['Patient'] = relationship(Patient, back_populates='consultations')
    selected_diagnoses: Mapped[List['ConsultationDiagnosis']] = relationship(
        'ConsultationDiagnosis', back_populates='consultation'
    )
    selected_exams: Mapped[List['ConsultationExam']] = relationship(
        'ConsultationExam', back_populates='consultation'
    )


class Diagnosis(Base):
    """Diagnosis model for medical conditions."""

    __tablename__ = 'diagnoses'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)


class Exam(Base):
    """Exam model for medical examinations."""

    __tablename__ = 'exams'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)


class ConsultationDiagnosis(Base):
    """Junction table linking consultations to diagnoses with evaluations."""

    __tablename__ = 'consultation_diagnoses'
    consultation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey('consultations.id'), primary_key=True
    )
    diagnosis_id: Mapped[int] = mapped_column(
        Integer, ForeignKey('diagnoses.id'), primary_key=True
    )

    # Evaluation fields
    accuracy: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    relevance: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    usefulness: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    coherence: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    comments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    consultation: Mapped['Consultation'] = relationship(
        Consultation, back_populates='selected_diagnoses'
    )
    diagnosis: Mapped['Diagnosis'] = relationship(Diagnosis)


class ConsultationExam(Base):
    """Junction table linking consultations to exams with evaluations."""

    __tablename__ = 'consultation_exams'
    consultation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey('consultations.id'), primary_key=True
    )
    exam_id: Mapped[int] = mapped_column(Integer, ForeignKey('exams.id'), primary_key=True)

    # Evaluation fields
    accuracy: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    relevance: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    usefulness: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    coherence: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    safety: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    comments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    consultation: Mapped['Consultation'] = relationship(Consultation, back_populates='selected_exams')
    exam: Mapped['Exam'] = relationship(Exam)
