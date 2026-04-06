"""
title: UI schema definitions for patient and consultation data.
summary: |-
  Pydantic v2 compatible models for frontend-backend data exchange.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Patient(BaseModel):
    """
    title: Patient demographic and clinical information.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    date_of_birth: datetime
    gender: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    medical_record_number: Optional[str] = None


class Consultation(BaseModel):
    """
    title: A single consultation encounter.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    patient_id: str
    date: datetime
    reason: str
    notes: Optional[str] = None
    provider_id: Optional[str] = None


class Diagnosis(BaseModel):
    """
    title: Clinical diagnosis associated with a consultation.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    consultation_id: str
    code: str
    description: str
    severity: Optional[str] = None
    status: Optional[str] = None


class Exam(BaseModel):
    """
    title: Clinical examination or test result.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    consultation_id: str
    type: str
    result: str
    date: datetime
    reference_range: Optional[str] = None


class ConsultationDiagnosis(BaseModel):
    """
    title: Linking table between consultation and diagnosis.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    consultation_id: str
    diagnosis_id: str
    primary: bool = False
    notes: Optional[str] = None


class ConsultationExam(BaseModel):
    """
    title: Linking table between consultation and exam.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    consultation_id: str
    exam_id: str
    ordered_date: datetime
    completed_date: Optional[datetime] = None


__all__ = [
    'Patient',
    'Consultation',
    'Diagnosis',
    'Exam',
    'ConsultationDiagnosis',
    'ConsultationExam',
]
