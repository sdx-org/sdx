"""Tests for the Deidentifier class, updated with parameterized testing."""

import pytest

from research.models.deidenitfier import Deidentifier

PII_TEST_CASES = [
    ('T1_SIMPLE_EMAIL_NAME', 'Contact Jane Doe at jane.d@example.com.', True),
    ('T2_US_PHONE', 'My phone number is 415-555-0132.', True),
    (
        'T3_UK_PHONE_LOCATION',
        'Call from +44 20 7946 0958, located at 10 Downing St, London.',
        True,
    ),
    ('T4_CREDIT_CARD', 'Do not use card 4111-1111-1111-1111.', True),
    (
        'T5_US_DRIVER_LICENSE',
        "Driver's license number is H123-4567-8901.",
        True,
    ),
    (
        'T6_NO_PII',
        'This is a perfectly safe sentence with no sensitive data.',
        False,
    ),
    ('T7_DATE_OF_BIRTH', 'Her date of birth is 1990-01-15.', True),
    (
        'T8_IP_ADDRESS',
        "The user's IP address was 203.0.113.55.",
        True,
    ),
    (
        'T9_MULTIPLE_NAMES',
        'A meeting between Alice, Bob, and Carol.',
        True,
    ),
    (
        'T10_IBAN_CODE',
        'Please transfer to IBAN DE89 3704 0044 0532 0130 00.',
        True,
    ),
    (
        'T11_US_SSN',
        "The applicant's SSN is 987-65-4321.",
        True,
    ),
    (
        'T12_UK_PASSPORT',
        'Her UK Passport number is 500000000.',
        True,
    ),
    (
        'T13_LOCATION_ONLY',
        'He is currently in Paris for a business trip.',
        True,
    ),
    (
        'T14_COMPLEX_MIX',
        'On 2023-05-10, Mr. Smith (john.p.smith@corp.com, '
        'cell: (202) 555-0177) filed a report from IP 203.0.113.55.',
        True,
    ),
    (
        'T15_SPANISH_PII',
        'Mi nombre es María López y mi correo es maria.lopez@email.es.',
        True,
    ),
    (
        'T16_US_ADDRESS',
        'Ship to 1600 Pennsylvania Avenue, Washington, DC 20500.',
        True,
    ),
    (
        'T17_CRYPTO_ADDRESS',
        'Send 1 BTC to 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa.',
        True,
    ),
    (
        'T18_URL_WITH_CREDENTIALS',
        'Do not use ftp://user:password@ftp.example.com/',
        True,
    ),
    (
        'T19_MEDICAL_ACCOUNT',
        'Patient MRN is MED-987654321.',
        True,
    ),
    (
        'T20_MIXED_NUMBERS',
        'Order #12345 contains item #67890 but my phone is (555) 123-4567.',
        True,
    ),
]


@pytest.fixture
def deidentifier() -> Deidentifier:
    """Provide a fresh instance of the Deidentifier class for each test."""
    return Deidentifier()


@pytest.mark.parametrize(
    'test_id, text, should_find_pii',
    PII_TEST_CASES,
    ids=[case[0] for case in PII_TEST_CASES],  # test_id for clearer reporting
)
@pytest.mark.parametrize('strategy', ['mask', 'hash'])
def test_pii_detection_and_deidentification(
    deidentifier: Deidentifier,
    strategy: str,
    test_id: str,
    text: str,
    should_find_pii: bool,
):
    """
    Verify PII detection and de-identification for various data samples.

    This test checks two key things:
    * That the `analyze` method finds PII when it's expected to exist.
    * That after de-identification, the original text is successfully altered.
    """
    analyzer_results = deidentifier.analyze(text)

    if should_find_pii:
        assert len(analyzer_results) > 0, (
            f'Failed: Expected to find PII in {test_id}, but found none.'
        )

        deidentified_text = deidentifier.deidentify(text, strategy=strategy)
        assert deidentified_text != text, (
            f"De-identification failed for strategy '{strategy}' on {test_id}"
        )

        print(text)
        print(deidentified_text)
        for result in analyzer_results:
            original_pii_slice = text[result.start : result.end]
            deidentified_slice = deidentified_text[result.start : result.end]
            assert original_pii_slice != deidentified_slice, (
                f"Original PII '{original_pii_slice}' at slice "
                f'[{result.start}:{result.end}] was not altered.'
            )

    else:
        assert len(analyzer_results) == 0, (
            f'Failed: Expected no PII in {test_id}, but found '
            f'{len(analyzer_results)} entities.'
        )


def test_add_custom_recognizer_and_analyze(deidentifier: Deidentifier):
    """Test: Ensure a custom recognizer can be added and used for analysis."""
    entity_name = 'ORDER_ID'
    regex_pattern = r'ORD-\d{4}'
    text_with_custom_id = 'The order confirmation is ORD-1234.'

    deidentifier.add_custom_recognizer(entity_name, regex_pattern)
    analyzer_results = deidentifier.analyze(text_with_custom_id)
    entities_found = {result.entity_type for result in analyzer_results}

    assert entity_name in entities_found


def test_unsupported_strategy_raises_error(deidentifier: Deidentifier):
    """Test: Ensure that an unsupported strategy raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        deidentifier.deidentify('Some text', strategy='encrypt')

    assert "Unsupported strategy: 'encrypt'" in str(excinfo.value)
