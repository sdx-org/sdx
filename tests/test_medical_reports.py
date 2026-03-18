"""
title: Test the extraction of medical report data.
"""

import io
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
    """
    title: Return a MedicalReportFileExtractor instance for testing.
    """
    return MedicalReportFileExtractor()


def test_only_supported_files_can_be_extracted(extractor):
    """
    title: Test that only supported files can be validated successfully.
    parameters:
      extractor:
        description: Value for extractor.
    """
    extractor._validate_or_raise(PDF_FILE)
    extractor._validate_or_raise(IMAGE_FILE)
    with pytest.raises(MedicalReportExtractorError):
        extractor._validate_or_raise(UNSUPPORTED_FILE)


def test_extract_text_from_pdf_file(extractor):
    """
    title: Test text extraction from PDF files returns valid string.
    parameters:
      extractor:
        description: Value for extractor.
    """
    text = extractor._extract_text_from_pdf(PDF_FILE)
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.skipif(not HAS_TESSERACT, reason='tesseract is not installed')
def test_extract_text_from_image_file(extractor):
    """
    title: Test text extraction from image files using OCR.
    parameters:
      extractor:
        description: Value for extractor.
    """
    text = extractor._extract_text_from_image(IMAGE_FILE)
    assert isinstance(text, str)
    assert len(text) > 0


def test_extract_unsupported_file_raises(extractor):
    """
    title: Test that unsupported file types raise appropriate errors.
    parameters:
      extractor:
        description: Value for extractor.
    """
    with pytest.raises(MedicalReportExtractorError):
        extractor._validate_or_raise(UNSUPPORTED_FILE)


def test_extract_corrupt_pdf_raises(extractor):
    """
    title: Test that corrupt PDF files raise TextExtractionError.
    parameters:
      extractor:
        description: Value for extractor.
    """
    with pytest.raises(TextExtractionError):
        extractor._extract_text_from_pdf(CORRUPT_PDF_FILE)


def test_extract_report_data_from_pdf_file(extractor):
    """
    title: Test structured text extraction payload from PDF files.
    parameters:
      extractor:
        description: Value for extractor.
    """
    report = extractor.extract_report_data(PDF_FILE)
    assert report['source_name'] == PDF_FILE.name
    assert report['source_type'] == 'pdf'
    assert report['mime_type'] == 'application/pdf'
    assert isinstance(report['text'], str)
    assert len(report['text']) > 0


@pytest.mark.skipif(not HAS_TESSERACT, reason='tesseract is not installed')
def test_extract_report_data_from_image_file(extractor):
    """
    title: Test structured text extraction payload from image files.
    parameters:
      extractor:
        description: Value for extractor.
    """
    report = extractor.extract_report_data(IMAGE_FILE)
    assert report['source_name'] == IMAGE_FILE.name
    assert report['source_type'] == 'image'
    assert report['mime_type'] == 'image/png'
    assert isinstance(report['text'], str)
    assert len(report['text']) > 0


def test_support_inmemory_pdf(extractor):
    """
    title: Test text extraction from in-memory PDF BytesIO objects.
    parameters:
      extractor:
        description: Value for extractor.
    """
    with open(PDF_FILE, 'rb') as f:
        pdf_bytes = io.BytesIO(f.read())
    text = extractor._extract_text_from_pdf(pdf_bytes)
    assert isinstance(text, str)
    assert len(text) > 0


@pytest.mark.skipif(not HAS_TESSERACT, reason='tesseract is not installed')
def test_support_inmemory_image(extractor):
    """
    title: Test text extraction from in-memory image BytesIO objects.
    parameters:
      extractor:
        description: Value for extractor.
    """
    with open(IMAGE_FILE, 'rb') as f:
        image_bytes = io.BytesIO(f.read())
    text = extractor._extract_text_from_image(image_bytes)
    assert isinstance(text, str)
    assert len(text) > 0


def test_empty_inmemory_file_raises(extractor):
    """
    title: Test that empty in-memory streams raise FileNotFoundError.
    parameters:
      extractor:
        description: Value for extractor.
    """
    empty_stream = io.BytesIO(b'')
    with pytest.raises(FileNotFoundError):
        extractor._validate_or_raise(empty_stream)


def test_extract_text_public_helper_matches_payload_text(extractor):
    """
    title: Public raw-text helper should match the structured payload content.
    parameters:
      extractor:
        description: Value for extractor.
    """
    text = extractor.extract_text(PDF_FILE)
    report = extractor.extract_report_data(PDF_FILE)
    assert text == report['text']
