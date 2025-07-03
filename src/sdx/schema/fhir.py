"""
FHIR-compatible resource definitions extended for TeleHealthCareAI.

All extensions preserve FHIR element names and validation rules
via subclassing from `fhir.resources` Pydantic models.
"""

from __future__ import annotations

import abc

from typing import Optional

from fhir.resources.annotation import Annotation as FhirAnnotation
from fhir.resources.clinicalimpression import (
    ClinicalImpression as FhirClinicalImpression,
)
from fhir.resources.condition import Condition as FhirCondition
from fhir.resources.encounter import Encounter as FhirEncounter
from fhir.resources.observation import Observation as FhirObservation
from fhir.resources.patient import Patient as FhirPatient
from fhir.resources.procedure import Procedure as FhirProcedure
from public import public
from pydantic import BaseModel, Field


@public
class BaseLanguage(BaseModel, abc.ABC):
    """Base class for language."""

    language: Optional[str] = Field(
        default=...,
        alias='language',
        description='IETF language tag representing the default language',
        examples=['en-US'],
    )


@public
class Patient(FhirPatient, BaseLanguage):
    """FHIR Patient with optional preferred language for text content."""

    ...


@public
class Encounter(FhirEncounter, BaseLanguage):
    """FHIR Encounter representing one clinical episode."""

    canonicalEpisodeId: Optional[str] = Field(
        None,
        alias='canonicalEpisodeId',
        description=(
            'Stable ID used across AI, physician and data-publishing modules.'
        ),
    )


@public
class Observation(
    FhirObservation, BaseLanguage
):  # No change, subclass kept for future hooks
    """FHIR Observation for symptoms or clinical findings."""

    pass


@public
class Condition(
    FhirCondition, BaseLanguage
):  # Subclass preserved for custom search helpers
    """FHIR Condition for AI-generated or physician-confirmed diagnoses."""

    pass


@public
class Procedure(FhirProcedure, BaseLanguage):
    """FHIR Procedure for treatment recommendations."""

    pass


@public
class ClinicalImpression(FhirClinicalImpression, BaseLanguage):
    """FHIR ClinicalImpression produced by the AI engine."""


@public
class Annotation(FhirAnnotation, BaseLanguage):
    """FHIR Annotation storing physician corrections or comments."""

    pass
