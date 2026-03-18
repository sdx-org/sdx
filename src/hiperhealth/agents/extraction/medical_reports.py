"""
title: Module for extracting text from PDF documents and images.
summary: |-
  Re-exports from ``hiperhealth.skills.extraction.medical_reports``
  for backward compatibility.
"""

from hiperhealth.skills.extraction.medical_reports import (
    BaseMedicalReportExtractor,
    FileExtension,
    FileInput,
    MedicalReportExtractorError,
    MedicalReportFileExtractor,
    MimeType,
    TextExtractionError,
    get_medical_report_extractor,
)

__all__ = [
    'BaseMedicalReportExtractor',
    'FileExtension',
    'FileInput',
    'MedicalReportExtractorError',
    'MedicalReportFileExtractor',
    'MimeType',
    'TextExtractionError',
    'get_medical_report_extractor',
]
