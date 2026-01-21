"""Helper functions for managing and processing medical reports."""

import logging

from typing import List, Optional, Tuple

from fastapi import UploadFile
from hiperhealth.agents.extraction.medical_reports import (
    MedicalReportExtractorError,
    MedicalReportFileExtractor,
)

logger = logging.getLogger(__name__)


def load_fhir_reports(consultation) -> List[dict]:
    """Load and deserialize FHIR reports from consultation."""
    reports = consultation.previous_tests

    if not reports:
        return []

    if not isinstance(reports, list):
        logger.warning('Loaded reports is not a list')
        return []

    return reports


def save_fhir_reports(consultation, reports: List[dict], repo) -> None:
    """Save FHIR reports to consultation."""
    try:
        consultation.previous_tests = reports
        repo.db.commit()
        logger.info(f'Saved {len(reports)} reports')
    except Exception:
        repo.db.rollback()
        logger.error('Failed to save fhir_reports', exc_info=True)
        raise ValueError('Failed to save reports')


def validate_report_file(
    report: UploadFile,
    seen_filenames: set,
    extractor: MedicalReportFileExtractor,
) -> Tuple[bool, Optional[str]]:
    """Validate uploaded report file."""
    filename_lower = report.filename.lower()

    if filename_lower in seen_filenames:
        return False, 'File already uploaded'

    # Accept if EITHER mimetype OR extension matches
    if report.content_type not in extractor.allowed_mimetypes and not any(
        filename_lower.endswith(f'.{ext}')
        for ext in extractor.allowed_extensions
    ):
        return False, 'Only PDF, PNG, JPEG, JPG files allowed'

    return True, None


async def process_uploaded_reports(
    reports: List[UploadFile],
    seen_filenames: set,
    extractor: MedicalReportFileExtractor,
) -> Tuple[List[dict], Optional[str]]:
    """Process uploaded medical reports and extract FHIR data."""
    fhir_reports = []

    for report in reports:
        if not report.filename:
            continue

        try:
            # Validate file
            valid, error_msg = validate_report_file(
                report, seen_filenames, extractor
            )
            if not valid:
                return [], error_msg

            # Use stream directly (memory efficient)
            await report.seek(0)
            fhir = extractor.extract_report_data(report.file)

            # Type validation: ensure dict
            if not isinstance(fhir, dict):
                logger.warning(
                    f'Unexpected extractor output type: {type(fhir).__name__}'
                )
                return [], 'Failed to process report'

            fhir['filename'] = report.filename
            fhir_reports.append(fhir)
            seen_filenames.add(report.filename.lower())

        except MedicalReportExtractorError:
            logger.error('Medical report extraction failed', exc_info=True)
            return [], 'Failed to extract report data'
        except Exception:
            logger.error('Error processing uploaded report', exc_info=True)
            return [], 'Error processing report'
        finally:
            await report.close()

    return fhir_reports, None
