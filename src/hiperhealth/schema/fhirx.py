"""
title: FHIR-compatible resource definitions extended for TeleHealthCareAI.
summary: |-
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
    """
    title: Base class for language.
    attributes:
      language:
        type: Optional[str]
        description: Value for language.
    """

    language: Optional[str] = Field(
        default=None,
        alias='language',
        description='IETF language tag representing the default language',
        examples=['en-US'],
    )


@public
class Patient(FhirPatient, BaseLanguage):
    """
    title: FHIR Patient with optional preferred language for text content.
    attributes:
      language:
        type: Optional[str]
        description: Value for language.
    """

    ...


@public
class Encounter(FhirEncounter, BaseLanguage):
    """
    title: FHIR Encounter representing one clinical episode.
    attributes:
      language:
        type: Optional[str]
        description: Value for language.
      canonicalEpisodeId:
        type: Optional[str]
        description: Value for canonicalEpisodeId.
    """

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
    """
    title: FHIR Observation for symptoms or clinical findings.
    attributes:
      language:
        type: Optional[str]
        description: Value for language.
    """

    pass


@public
class Condition(
    FhirCondition, BaseLanguage
):  # Subclass preserved for custom search helpers
    """
    title: FHIR Condition for AI-generated or physician-confirmed diagnoses.
    attributes:
      language:
        type: Optional[str]
        description: Value for language.
    """

    pass


@public
class Procedure(FhirProcedure, BaseLanguage):
    """
    title: FHIR Procedure for treatment recommendations.
    attributes:
      language:
        type: Optional[str]
        description: Value for language.
    """

    pass


@public
class ClinicalImpression(FhirClinicalImpression, BaseLanguage):
    """
    title: FHIR ClinicalImpression produced by the AI engine.
    attributes:
      language:
        type: Optional[str]
        description: Value for language.
    """


@public
class Annotation(FhirAnnotation, BaseLanguage):
    """
    title: FHIR Annotation storing physician corrections or comments.
    attributes:
      language:
        type: Optional[str]
        description: Value for language.
    """

    pass
