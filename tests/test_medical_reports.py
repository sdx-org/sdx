"""Test the extraction of medical report data."""

import io
import os
import shutil

from pathlib import Path

import pytest

from hiperhealth.agents.extraction.medical_reports import (
    MedicalReportExtractorError,
    MedicalReportFileExtractor,
    TextExtractionError,
)

TEST_DATA_PATH = Path(__file__).parent / 'data' / 'reports'
PDF_FILE = TEST_DATA_PATH / 'pdf_reports' / 'report-1.pdf'
IMAGE_FILE = TEST_DATA_PATH / 'image_reports' / 'image-1.png'
UNSUPPORTED_FILE = TEST_DATA_PATH / 'pdf_reports' / 'unsupported_file.txt'
CORRUPT_PDF_FILE = TEST_DATA_PATH / 'pdf_reports' / 'corrupt_report.txt'
HAS_TESSERACT = shutil.which('tesseract') is not None


@pytest.fixture
def extractor():
    """Return a MedicalReportFileExtractor instance for testing."""
    return MedicalReportFileExtractor()


def test_only_supported_files_can_be_extracted(extractor):
    """Test that only supported files can be validated successfully."""
    extractor._validate_or_raise(PDF_FILE)
    extractor._validate_or_raise(IMAGE_FILE)
    with pytest.raises(MedicalReportExtractorError):
        extractor._validate_or_raise(UNSUPPORTED_FILE)


def test_extract_text_from_pdf_file(extractor):
    """Test text extraction from PDF files returns valid string."""
    text = extractor._extract_text_from_pdf(PDF_FILE)
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.skipif(not HAS_TESSERACT, reason='tesseract is not installed')
def test_extract_text_from_image_file(extractor):
    """Test text extraction from image files using OCR."""
    text = extractor._extract_text_from_image(IMAGE_FILE)
    assert isinstance(text, str)
    assert len(text) > 0


def test_extract_unsupported_file_raises(extractor):
    """Test that unsupported file types raise appropriate errors."""
    with pytest.raises(MedicalReportExtractorError):
        extractor._validate_or_raise(UNSUPPORTED_FILE)


def test_extract_corrupt_pdf_raises(extractor):
    """Test that corrupt PDF files raise TextExtractionError."""
    with pytest.raises(TextExtractionError):
        extractor._extract_text_from_pdf(CORRUPT_PDF_FILE)


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='OpenAI API key not available'
)
def test_extract_report_data_from_pdf_file(extractor):
    """Test FHIR data extraction from PDF files."""
    api_key = os.environ.get('OPENAI_API_KEY')  # use environment key
    fhir_data = extractor.extract_report_data(PDF_FILE, api_key)
    assert isinstance(fhir_data, dict)
    assert len(fhir_data) > 0
    expected_keys = {'Patient', 'Condition', 'Observation', 'DiagnosticReport'}
    assert any(key in fhir_data for key in expected_keys)


@pytest.mark.skipif(
    (not os.environ.get('OPENAI_API_KEY')) or (not HAS_TESSERACT),
    reason='OpenAI API key or tesseract not available',
)
def test_extract_report_data_from_image_file(extractor):
    """Test FHIR data extraction from image files."""
    api_key = os.environ.get('OPENAI_API_KEY')  # use environment key
    fhir_data = extractor.extract_report_data(IMAGE_FILE, api_key)
    assert isinstance(fhir_data, dict)
    assert len(fhir_data) > 0
    expected_keys = {'Patient', 'Condition', 'Observation', 'DiagnosticReport'}
    assert any(key in fhir_data for key in expected_keys)


def test_support_inmemory_pdf(extractor):
    """Test text extraction from in-memory PDF BytesIO objects."""
    with open(PDF_FILE, 'rb') as f:
        pdf_bytes = io.BytesIO(f.read())
    text = extractor._extract_text_from_pdf(pdf_bytes)
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.skipif(not HAS_TESSERACT, reason='tesseract is not installed')
def test_support_inmemory_image(extractor):
    """Test text extraction from in-memory image BytesIO objects."""
    with open(IMAGE_FILE, 'rb') as f:
        image_bytes = io.BytesIO(f.read())
    text = extractor._extract_text_from_image(image_bytes)
    assert isinstance(text, str)
    assert len(text) > 0


def test_empty_inmemory_file_raises(extractor):
    """Test that empty in-memory streams raise FileNotFoundError."""
    empty_stream = io.BytesIO(b'')
    with pytest.raises(FileNotFoundError):
        extractor._validate_or_raise(empty_stream)
