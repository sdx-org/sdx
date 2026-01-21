"""Pydantic schemas for research application."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class PatientBase(BaseModel):
    """Base model for core patient demographics."""

    age: Optional[int] = None
    gender: Optional[str] = Field(None, max_length=50)


class PatientCreate(PatientBase):
    """Schema for creating a new patient."""

    uuid: str = Field(..., max_length=36)


class Patient(PatientBase):
    """Complete patient schema with relationships."""

    id: int
    uuid: str = Field(..., max_length=36)
    consultations: List[Consultation] = []

    class Config:
        """Pydantic config."""

        from_attributes = True


class ConsultationBase(BaseModel):
    """Base model for a single patient consultation."""

    timestamp: Optional[datetime] = None
    lang: Optional[str] = Field(None, max_length=10)
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    diet: Optional[str] = None
    sleep_hours: Optional[float] = None
    physical_activity: Optional[str] = None
    mental_exercises: Optional[str] = None
    symptoms: Optional[str] = None
    mental_health: Optional[str] = None
    previous_tests: Optional[str] = None
    wearable_data: Optional[Any] = None
    ai_diag_raw: Optional[Any] = None
    ai_exam_raw: Optional[Any] = None


class ConsultationCreate(ConsultationBase):
    """Schema for creating a new consultation."""

    patient_id: int


class Consultation(ConsultationBase):
    """Complete consultation schema with relationships."""

    id: int
    patient_id: int
    patient: Patient
    selected_diagnoses: List[ConsultationDiagnosis] = []
    selected_exams: List[ConsultationExam] = []

    class Config:
        """Pydantic config."""

        from_attributes = True


class DiagnosisBase(BaseModel):
    """Base model for medical diagnoses."""

    name: str = Field(..., max_length=255)


class DiagnosisCreate(DiagnosisBase):
    """Schema for creating a new diagnosis."""

    pass


class Diagnosis(DiagnosisBase):
    """Complete diagnosis schema."""

    id: int

    class Config:
        """Pydantic config."""

        from_attributes = True


class ExamBase(BaseModel):
    """Base model for medical examinations."""

    name: str = Field(..., max_length=255)


class ExamCreate(ExamBase):
    """Schema for creating a new exam."""

    pass


class Exam(ExamBase):
    """Complete exam schema."""

    id: int

    class Config:
        """Pydantic config."""

        from_attributes = True


class ConsultationDiagnosisBase(BaseModel):
    """Base model for consultation-diagnosis evaluation."""

    accuracy: Optional[int] = None
    relevance: Optional[int] = None
    usefulness: Optional[int] = None
    coherence: Optional[int] = None
    comments: Optional[str] = None


class ConsultationDiagnosisCreate(ConsultationDiagnosisBase):
    """Schema for creating consultation-diagnosis relationship."""

    consultation_id: int
    diagnosis_id: int


class ConsultationDiagnosis(ConsultationDiagnosisBase):
    """Complete consultation-diagnosis schema with relationships."""

    consultation: Consultation
    diagnosis: Diagnosis

    class Config:
        """Pydantic config."""

        from_attributes = True


class ConsultationExamBase(BaseModel):
    """Base model for consultation-exam evaluation."""

    accuracy: Optional[int] = None
    relevance: Optional[int] = None
    usefulness: Optional[int] = None
    coherence: Optional[int] = None
    safety: Optional[str] = Field(None, max_length=50)
    comments: Optional[str] = None


class ConsultationExamCreate(ConsultationExamBase):
    """Schema for creating consultation-exam relationship."""

    consultation_id: int
    exam_id: int


class ConsultationExam(ConsultationExamBase):
    """Complete consultation-exam schema with relationships."""

    consultation: Consultation
    exam: Exam

    class Config:
        """Pydantic config."""

        from_attributes = True


# Rebuild models to resolve forward references
Patient.model_rebuild()
Consultation.model_rebuild()
ConsultationDiagnosis.model_rebuild()
ConsultationExam.model_rebuild()
