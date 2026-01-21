"""API Request/ Response schemas."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Generic response schemas
class StepResponse(BaseModel):
    """Generic response returned after submitting a form step."""

    success: bool = Field(
        ..., description='Indicates if the step submission was successful.'
    )
    next_step: str = Field(..., description='The next step to proceed to.')
    patient_id: str = Field(
        ..., description='The unique identifier for the patient.'
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(..., description='Error message detailing the issue.')


# Start/ Create Patient Schemas
class CreatePatientRequest(BaseModel):
    """Request schema for creating a new patient."""

    lang: str = Field(..., description="Language code, e.g. 'en' or 'es'.")


class CreatePatientResponse(BaseModel):
    """Response schema after creating a new patient."""

    patient_id: str = Field(
        ..., description='The unique identifier for the newly created patient.'
    )
    lang: str = Field(..., description="Language code, e.g. 'en' or 'es'.")
    created_at: str = Field(
        ..., description='Timestamp when the patient was created.'
    )


# Demopgraphics, Lifestyle, Symptoms Mental Health Schemas
class DemographicsRequest(BaseModel):
    """Schema for demographics step request."""

    age: int = Field(..., description='Age of the patient.')
    gender: str = Field(..., description='Gender of the patient.')
    weight_kg: float = Field(
        ..., description='Weight of the patient in kilograms.'
    )
    height_cm: float = Field(
        ..., description='Height of the patient in centimeters.'
    )


class LifestyleRequest(BaseModel):
    """Schema for lifestyle step request."""

    diet: str = Field(..., description='Dietary habits of the patient.')
    sleep_hours: float = Field(
        ..., description='Average hours of sleep per night.'
    )
    physical_activity: str = Field(
        ..., description='Physical activity routines of the patient.'
    )
    mental_exercises: str = Field(
        ..., description='Mental exercises or activities performed.'
    )


class SymptomsRequest(BaseModel):
    """Schema for symptoms step request."""

    symptoms: str = Field(
        ..., description='Current symptoms reported by the patient.'
    )


class MentalHealthRequest(BaseModel):
    """Schema for mental health step request."""

    mental_health: str = Field(
        ..., description='Overall mental health status of the patient.'
    )


# Medical Report and Wearable Data Schemas
class ReportSummary(BaseModel):
    """Schema for the final patient report summary."""

    file_name: str = Field(..., description='Name of the report file.')
    resource_type: str = Field(..., description='Type of the resource.')
    fhir_content: Optional[Dict[str, Any]] = Field(
        ..., description='The full extracted FHIR JSON content.'
    )


class MedicalReportListResponse(BaseModel):
    """Response schema for listing medical reports."""

    total_reports: int = Field(
        ..., description='Total number of medical reports.'
    )
    reports: List[ReportSummary] = Field(
        ..., description='List of medical report summaries.'
    )


class MedicalReportUploadResponse(BaseModel):
    """Response schema after uploading a medical report."""

    success: bool = Field(
        ..., description='Indicates if the upload was successful.'
    )
    uploaded_files: List[str] = Field(
        ..., description='List of names of the uploaded files.'
    )
    total_reports: int = Field(
        ..., description='Total number of medical reports after upload.'
    )
    next_step: str = Field(..., description='The next step to proceed to.')


class MedicalReportSkipResponse(BaseModel):
    """Response schema after skipping medical report upload."""

    success: bool = Field(
        ..., description='Indicates if the skip action was successful.'
    )
    skipped: bool = Field(
        ..., description='Indicates if the medical report upload was skipped.'
    )
    next_step: str = Field(..., description='The next step to proceed to.')


class WearableDataUploadResponse(BaseModel):
    """Response schema after uploading wearable data."""

    success: bool = Field(
        ...,
        description='Indicates if the wearable data upload was successful.',
    )
    file_name: str = Field(
        ..., description='Name of the uploaded wearable data file.'
    )
    next_step: str = Field(..., description='The next step to proceed to.')


class WearableDataSkipResponse(BaseModel):
    """Response schema after skipping wearable data upload."""

    success: bool = Field(
        ..., description='Indicates if the skip action was successful.'
    )
    skipped: bool = Field(
        ..., description='Indicates if the wearable data upload was skipped.'
    )
    next_step: str = Field(..., description='The next step to proceed to.')


# Diagnosis Schemas
class DiagnosisOption(BaseModel):
    """Schema for a single diagnosis option."""

    name: str = Field(..., description='Diagnosis name.')
    description: str = Field(..., description='Description of the diagnosis.')


class DiagnosisGetResponse(BaseModel):
    """Response containing AI suggestions to be displayed to the user."""

    patient_id: str = Field(
        ..., description='The unique identifier for the patient.'
    )
    summary: str = Field(
        ..., description='Summary of the AI-generated diagnosis suggestions.'
    )
    options: List[DiagnosisOption] = Field(
        ..., description='List of diagnosis options suggested by AI.'
    )


class DiagnosisRating(BaseModel):
    """Ratings provided by the physician for a specific diagnosis option."""

    accuracy: int = Field(..., description='Accuracy rating (1-10).')
    relevance: int = Field(..., description='Relevance rating (1-10).')
    usefulness: int = Field(..., description='Usefulness rating (1-10).')
    coherence: int = Field(..., description='Coherence rating (1-10).')
    comments: Optional[str] = Field(
        None, description='Additional comments by the physician.'
    )


class DiagnosisSubmitRequest(BaseModel):
    """Request schema for submitting selected diagnoses with ratings."""

    selected_diagnoses: List[str] = Field(
        ..., description='List of selected diagnosis names.'
    )
    evaluations: Dict[str, DiagnosisRating] = Field(
        ...,
        description='Mapping of diagnosis names to their respective ratings.',
    )


class DiagnosisSubmitResponse(BaseModel):
    """Response schema after submitting selected diagnoses."""

    success: bool = Field(
        ..., description='Indicates if the submission was successful.'
    )
    patient_id: str = Field(
        ..., description='The unique identifier for the patient.'
    )
    next_step: str = Field(..., description='The next step to proceed to.')


# AI Suggested Exam Schemas
class ExamOption(BaseModel):
    """Schema for a single exam suggestion."""

    name: str = Field(..., description='Exam name.')
    description: Optional[str] = Field(
        ..., description='Description of the exam.'
    )


class ExamGetResponse(BaseModel):
    """Response containing exam suggestions to be displayed to the user."""

    patient_id: str = Field(
        ..., description='The unique identifier for the patient.'
    )
    summary: str = Field(
        ..., description='Summary of the AI-generated exam suggestions.'
    )
    options: List[ExamOption] = Field(
        ..., description='List of exam options suggested.'
    )


class ExamRating(BaseModel):
    """Ratings provided by the physician for a specific exam option."""

    accuracy: int = Field(..., description='accuracy rating (1-10).')
    relevance: int = Field(..., description='Relevance rating (1-10).')
    usefulness: int = Field(..., description='usefulness rating (1-10).')
    coherence: int = Field(..., description='coherence rating (1-10).')
    safety: int = Field(..., description='Relevance rating (1-10).')
    comments: Optional[str] = Field(
        None, description='Additional comments by the physician.'
    )


class ExamSubmitRequest(BaseModel):
    """Request schema for submitting selected exams with ratings."""

    selected_exams: List[str] = Field(
        ..., description='List of selected exam names.'
    )
    evaluations: Dict[str, ExamRating] = Field(
        ...,
        description='Mapping of exam names to their respective ratings.',
    )


class ExamSubmitResponse(BaseModel):
    """Response schema after submitting selected exams."""

    success: bool = Field(
        ..., description='Indicates if the submission was successful.'
    )
    patient_id: str = Field(
        ..., description='The unique identifier for the patient.'
    )
    is_complete: bool = Field(
        ..., description='Marks the end of the workflow.'
    )


# Dashboard/ Status Models
class ConsultationStatusResponse(BaseModel):
    """Used to resotre state or show progress."""

    patient_id: str = Field(
        ..., description='The unique identifier for the patient.'
    )
    current_step: str = Field(
        ..., description='The current step in the consultation workflow.'
    )
    completed_steps: List[str] = Field(
        ..., description='List of completed steps in the workflow.'
    )
    is_complete: bool = Field(
        ..., description='Indicates if the consultation workflow is complete.'
    )
    patient_dict: Dict[str, Any] = Field(
        ..., description='Dictionary representation of the patient data.'
    )
    lang: str = Field(..., description="Language code, e.g. 'en' or 'es'.")


class PatientSummary(BaseModel):
    """Schema for patient summary in dashboard list."""

    patient_id: str = Field(
        ..., description='The unique identifier for the patient.'
    )
    created_at: Optional[str] = Field(
        None, description='Timestamp when the patient was created.'
    )
    current_step: str = Field(
        ..., description='The current step in the consultation workflow.'
    )
    is_complete: bool = Field(
        ..., description='Indicates if the consultation workflow is complete.'
    )


class DeleteResponse(BaseModel):
    """Response schema after deleting a patient record."""

    success: bool = Field(
        ..., description='Indicates if the deletion was successful.'
    )
    patient_id: str = Field(
        ..., description='The unique identifier for the deleted patient.'
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description='Health status of the API service.')
    service: str = Field(..., description='Name of the service being checked.')
