"""Domain-specific (non-FHIR) Pydantic models used across the platform."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from sdx.models.fhir import BaseLanguage


class AIOutput(BaseModel, BaseLanguage):
    """Full AI-generated text associated with a particular encounter."""

    id: str
    encounter_id: str
    type: Literal['anamnesis', 'diagnosis', 'treatment']
    content: str
    model_version: str
    timestamp: datetime


class Evaluation(BaseModel, BaseLanguage):
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


class DeIdentifiedDatasetDescriptor(BaseModel, BaseLanguage):
    """Metadata describing a dataset produced for open publication."""

    dataset_id: str
    generation_date: datetime
    version: str
    records: int
    license: str
    url: Optional[str] = None
