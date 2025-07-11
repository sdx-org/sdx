"""Tests for the Deidentifier class."""

import pytest

from research.models.deidenitfier import Deidentifier

# Sample text containing various PII
SAMPLE_TEXT = 'My name is John Doe and my phone number is (555) 123-4567.'
CUSTOM_ID_TEXT = 'My user ID is CUST-12345.'


@pytest.fixture
def deidentifier() -> Deidentifier:
    """Provide a fresh instance of the Deidentifier class for each test.

    This ensures that tests are isolated from each other.
    """
    return Deidentifier()


def test_initialization(deidentifier: Deidentifier):
    """Test 1: Verify that the Deidentifier class initializes correctly.

    Checks if the analyzer and anonymizer engines are not None.
    """
    assert deidentifier.analyzer is not None, (
        'AnalyzerEngine should be initialized.'
    )
    assert deidentifier.anonymizer is not None, (
        'AnonymizerEngine should be initialized.'
    )


def test_analyze_finds_pii(deidentifier: Deidentifier):
    """Test 2: Check if the analyze method finds default PII entities."""
    analyzer_results = deidentifier.analyze(SAMPLE_TEXT)
    entities_found = {result.entity_type for result in analyzer_results}

    assert 'PERSON' in entities_found
    assert 'PHONE_NUMBER' in entities_found


def test_add_custom_recognizer_and_analyze(deidentifier: Deidentifier):
    """Test 3: Ensure a custom recognizer can be added & used for analysis."""
    # Define a custom entity for a specific ID format
    entity_name = 'CUSTOM_ID'
    regex_pattern = r'CUST-\d{5}'
    deidentifier.add_custom_recognizer(entity_name, regex_pattern)

    analyzer_results = deidentifier.analyze(CUSTOM_ID_TEXT)
    entities_found = {result.entity_type for result in analyzer_results}

    assert entity_name in entities_found


def test_deidentify_with_mask_strategy(deidentifier: Deidentifier):
    """Test 4: Verify the 'mask' de-identification strategy.

    Checks if the PII is replaced with masking characters.
    """
    deidentified_text = deidentifier.deidentify(SAMPLE_TEXT, strategy='mask')
    # The exact masked output can vary, so we check for the masking character
    # and ensure the original PII is gone.
    assert 'John Doe' not in deidentified_text
    assert '(555) 123-4567' not in deidentified_text
    assert '*' in deidentified_text


def test_deidentify_with_hash_strategy(deidentifier: Deidentifier):
    """Test 5: Verify the 'hash' de-identification strategy.

    The original PII should be replaced by its SHA-256 hash.
    """
    deidentified_text = deidentifier.deidentify(SAMPLE_TEXT, strategy='hash')
    assert 'John Doe' not in deidentified_text
    assert '(555) 123-4567' not in deidentified_text
    # Hashed values are long strings of hex characters
    assert len(deidentified_text.split()[-1]) > 20


def test_unsupported_strategy_raises_error(deidentifier: Deidentifier):
    """Test 6: Ensure that an unsupported strategy raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        deidentifier.deidentify(SAMPLE_TEXT, strategy='encrypt')

    assert "Unsupported strategy: 'encrypt'" in str(excinfo.value)
