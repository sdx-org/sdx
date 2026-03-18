"""
title: Domain-specific (non-FHIR) Pydantic models used across the platform.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

from hiperhealth.schema.fhirx import BaseLanguage


class AIOutput(BaseLanguage, BaseModel):
    """
    title: Full AI-generated text associated with a particular encounter.
    attributes:
      id:
        type: str
        description: Value for id.
      encounter_id:
        type: str
        description: Value for encounter_id.
      type:
        type: Literal[anamnesis, diagnosis, treatment]
        description: Value for type.
      content:
        type: str
        description: Value for content.
      model_version:
        type: str
        description: Value for model_version.
      timestamp:
        type: datetime
        description: Value for timestamp.
    """

    id: str
    encounter_id: str
    type: Literal['anamnesis', 'diagnosis', 'treatment']
    content: str
    model_version: str
    timestamp: datetime


class Evaluation(BaseLanguage, BaseModel):
    """
    title: Structured physician rating of an AIOutput instance.
    attributes:
      id:
        type: str
        description: Value for id.
      aioutput_id:
        type: str
        description: Value for aioutput_id.
      output_type:
        type: Literal[anamnesis, diagnosis, treatment]
        description: Value for output_type.
      ratings:
        type: dict[Literal[accuracy, relevance, usefulness, coherence], int]
        description: Value for ratings.
      safety:
        type: Literal[safe, needs_review, unsafe]
        description: Value for safety.
      comments:
        type: Optional[str]
        description: Value for comments.
      timestamp:
        type: datetime
        description: Value for timestamp.
    """

    id: str
    aioutput_id: str
    output_type: Literal['anamnesis', 'diagnosis', 'treatment']
    ratings: dict[
        Literal['accuracy', 'relevance', 'usefulness', 'coherence'], int
    ]
    safety: Literal['safe', 'needs_review', 'unsafe']
    comments: Optional[str] = None
    timestamp: datetime


class DeIdentifiedDatasetDescriptor(BaseLanguage, BaseModel):
    """
    title: Metadata describing a dataset produced for open publication.
    attributes:
      dataset_id:
        type: str
        description: Value for dataset_id.
      generation_date:
        type: datetime
        description: Value for generation_date.
      version:
        type: str
        description: Value for version.
      records:
        type: int
        description: Value for records.
      license:
        type: str
        description: Value for license.
      url:
        type: Optional[str]
        description: Value for url.
    """

    dataset_id: str
    generation_date: datetime
    version: str
    records: int
    license: str
    url: Optional[str] = None
