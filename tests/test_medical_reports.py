"""Tests for the medical_reports module."""

import os

import pytest

from sdx.medical_reports import extract_text_from_pdf, get_report_data_from_pdf


def test_extract_text_from_pdf(reports_data_dir):
    """Test PDF text extraction functionality."""
    test_files = list(reports_data_dir.glob('*.pdf'))
    if not test_files:
        pytest.skip('No test PDF files available in the reports directory')

    pdf_path = test_files[0]

    text = extract_text_from_pdf(pdf_path)

    assert isinstance(text, str)
    assert len(text) > 0, 'Extracted text should not be empty'


def test_extract_text_nonexistent_file():
    """Test error handling for non-existent PDF files."""
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf('nonexistent_file.pdf')


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='OpenAI API key not available'
)
def test_get_report_data_from_pdf(reports_data_dir, openai_api_key):
    """Test FHIR data extraction from PDF."""
    test_files = list(reports_data_dir.glob('*.pdf'))
    if not test_files:
        pytest.skip('No test PDF files available in the reports directory')

    pdf_path = test_files[0]

    fhir_data = get_report_data_from_pdf(pdf_path, api_key=openai_api_key)

    assert isinstance(fhir_data, dict)
    assert len(fhir_data) > 0, 'No FHIR resources were extracted'

    # Check that at least one FHIR resource was extracted
    # Common FHIR resource types include Patient, Condition, Observation, etc.
    expected_resource_types = {
        'Patient',
        'Condition',
        'Observation',
        'DiagnosticReport',
    }
    assert any(
        resource_type in fhir_data for resource_type in expected_resource_types
    ), 'No expected FHIR resource types found'
