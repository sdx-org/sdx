"""
title: Privacy models package.
summary: |-
  Re-exports from ``hiperhealth.skills.privacy``
  for backward compatibility.
"""

from hiperhealth.skills.privacy.deidentifier import (
    Deidentifier,
    deidentify_patient_record,
)

__all__ = ['Deidentifier', 'deidentify_patient_record']
