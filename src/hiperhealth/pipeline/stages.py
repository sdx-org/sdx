"""
title: Pipeline stage definitions.
"""

from __future__ import annotations

from enum import Enum


class Stage(str, Enum):
    """
    title: Named phases of a clinical encounter.
    summary: |-
      Custom string stage names are also accepted by the StageRunner;
      this enum provides the standard built-in stages.
    """

    SCREENING = 'screening'
    INTAKE = 'intake'
    DIAGNOSIS = 'diagnosis'
    EXAM = 'exam'
    TREATMENT = 'treatment'
    PRESCRIPTION = 'prescription'
