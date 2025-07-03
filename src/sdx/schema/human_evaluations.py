"""Domain-specific (non-FHIR) Pydantic models used across the platform."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Literal, Optional

from pydantic import BaseModel

from sdx.schema.fhir import BaseLanguage


class AIOutput(BaseLanguage, BaseModel):
    """Full AI-generated text associated with a particular encounter."""

    id: str
    encounter_id: str
    type: Literal['anamnesis', 'diagnosis', 'treatment']
    content: str
    model_version: str
    timestamp: datetime


class Evaluation(BaseLanguage, BaseModel):
    """Structured physician rating of an AIOutput instance."""

    id: str
    aioutput_id: str
    output_type: Literal['anamnesis', 'diagnosis', 'treatment']
    ratings: Dict[
        Literal['accuracy', 'relevance', 'usefulness', 'coherence'], int
    ]
    safety: Literal['safe', 'needs_review', 'unsafe']
    comments: Optional[str] = None
    timestamp: datetime


class DeIdentifiedDatasetDescriptor(BaseLanguage, BaseModel):
    """Metadata describing a dataset produced for open publication."""

    dataset_id: str
    generation_date: datetime
    version: str
    records: int
    license: str
    url: Optional[str] = None
