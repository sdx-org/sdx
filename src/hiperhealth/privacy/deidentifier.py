"""
title: A module for PII detection and de-identification.
summary: |-
  Re-exports from ``hiperhealth.skills.privacy.deidentifier``
  for backward compatibility.
"""

from hiperhealth.skills.privacy.deidentifier import (
    Deidentifier,
    PrivacySkill,
    deidentify_patient_record,
)

__all__ = ['Deidentifier', 'PrivacySkill', 'deidentify_patient_record']
