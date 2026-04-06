"""
title: Application schema definitions and API models.
summary: |-
  Pydantic v2 compatible schemas for request/response validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class BaseResponse(BaseModel):
    """
    title: Base response model for API endpoints.
    """

    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PatientCreate(BaseModel):
    """
    title: Request payload for creating a new patient.
    """

    name: str = Field(..., min_length=1, max_length=200)
    date_of_birth: datetime
    gender: str
    email: Optional[str] = Field(None, json_schema_extra={"populate_by_name": True})
    phone: Optional[str] = None
    address: Optional[str] = None


class PatientUpdate(BaseModel):
    """
    title: Request payload for updating patient information.
    """

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    email: Optional[str] = Field(
        None,
        json_schema_extra={
            "populate_by_name": True,
            "description": "Patient email address",
        },
    )
    phone: Optional[str] = None
    address: Optional[str] = None


class ConsultationCreate(BaseModel):
    """
    title: Request payload for creating a consultation.
    """

    patient_id: str = Field(..., json_schema_extra={"populate_by_name": True})
    date: datetime
    reason: str = Field(..., min_length=1, max_length=500)
    notes: Optional[str] = None
    provider_id: Optional[str] = None


class DiagnosticRequest(BaseModel):
    """
    title: Request for diagnostic analysis.
    """

    consultation_id: str = Field(
        ...,
        json_schema_extra={
            "populate_by_name": True,
            "description": "Reference to the consultation",
        },
    )
    symptoms: List[str] = Field(
        ...,
        min_length=1,
        json_schema_extra={"description": "List of reported symptoms"},
    )
    duration_days: Optional[int] = Field(
        None,
        ge=0,
        json_schema_extra={"description": "Symptom duration in days"},
    )
    severity: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        json_schema_extra={"description": "Severity scale 1-10"},
    )


class DiagnosticResponse(BaseModel):
    """
    title: Response containing diagnostic recommendations.
    """

    model_config = ConfigDict(from_attributes=True)

    consultation_id: str
    primary_diagnosis: Optional[str] = None
    differential_diagnoses: List[str] = Field(
        default_factory=list,
        json_schema_extra={"description": "List of possible conditions"},
    )
    recommended_exams: List[str] = Field(
        default_factory=list,
        json_schema_extra={"description": "Suggested examinations or tests"},
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        json_schema_extra={"description": "AI confidence in diagnosis"},
    )


class ExamResult(BaseModel):
    """
    title: Examination result data.
    """

    exam_type: str = Field(..., json_schema_extra={"populate_by_name": True})
    value: Any
    unit: Optional[str] = None
    reference_min: Optional[float] = Field(
        None,
        json_schema_extra={
            "populate_by_name": True,
            "description": "Minimum normal value",
        },
    )
    reference_max: Optional[float] = Field(
        None,
        json_schema_extra={
            "populate_by_name": True,
            "description": "Maximum normal value",
        },
    )
    interpretation: Optional[str] = None


class BatchExamResults(BaseModel):
    """
    title: Multiple examination results submitted together.
    """

    consultation_id: str = Field(..., json_schema_extra={"populate_by_name": True})
    results: List[ExamResult] = Field(
        ...,
        min_length=1,
        json_schema_extra={"description": "List of exam results"},
    )
    performed_date: datetime = Field(
        default_factory=datetime.utcnow,
        json_schema_extra={"description": "Date exams were performed"},
    )


class SearchQuery(BaseModel):
    """
    title: Generic search query parameters.
    """

    query: str = Field(..., min_length=1, max_length=500)
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        json_schema_extra={
            "populate_by_name": True,
            "description": "Additional filter criteria",
        },
    )
    limit: int = Field(
        20,
        ge=1,
        le=100,
        json_schema_extra={"description": "Maximum results to return"},
    )
    offset: int = Field(
        0,
        ge=0,
        json_schema_extra={"description": "Number of results to skip"},
    )


__all__ = [
    'BaseResponse',
    'PatientCreate',
    'PatientUpdate',
    'ConsultationCreate',
    'DiagnosticRequest',
    'DiagnosticResponse',
    'ExamResult',
    'BatchExamResults',
    'SearchQuery',
]
