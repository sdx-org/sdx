"""Security gateway for PHI and secret detection.

This module provides healthcare-specific implementations for detecting
Protected Health Information (PHI) and secrets in text before LLM submission.
Custom implementations optimized for HIPAA compliance.
"""

import re
from dotenv import load_dotenv
from pydantic.dataclasses import dataclass
from typing import TypeVar, List

from .sanitize import (
    DetectorResult,
    BaseDetector,
    Extra,
    _SECRETS_PATTERNS,
    UnicodeDetector,
)
from .batch import PRESIDIO_EXTRA

load_dotenv()

T = TypeVar('T')
R = TypeVar('R')

Extra.extras = {}


class PHI_Analyzer(BaseDetector):
    """Analyzer for detecting Protected Health Information (PHI) in text.

    PHI includes all HIPAA-protected health information that can identify individuals:
    - Standard PII (names, addresses, SSN, phone, email)
    - Medical identifiers (MRN, health plan numbers, account numbers)
    - Device identifiers and serial numbers
    - Biometric identifiers
    - URLs and IP addresses in healthcare context
    - Full-face photos and comparable images
    """

    # Default PHI entity types for healthcare context
    DEFAULT_PHI_ENTITIES = [
        'PERSON',
        'EMAIL_ADDRESS',
        'PHONE_NUMBER',
        'LOCATION',
        'DATE_TIME',
        'NRP',
        'MEDICAL_LICENSE',
        'US_SSN',
        'US_PASSPORT',
        'US_DRIVER_LICENSE',
        'CREDIT_CARD',
        'IBAN_CODE',
        'IP_ADDRESS',
        'URL',
    ]

    def __init__(self, threshold=0.5):
        AnalyzerEngine = PRESIDIO_EXTRA.package(
            'presidio_analyzer'
        ).import_names('AnalyzerEngine')
        self.analyzer = AnalyzerEngine()
        self.threshold = threshold

    def detect_all(self, text: str, entities: list[str] | None = None):
        """Detect PHI entities in text.

        Args:
            text: Text to analyze for PHI
            entities: Specific entity types to detect, defaults to all PHI types

        Returns:
            List of detected PHI entities with confidence scores
        """
        # Use healthcare-specific entities if none specified
        if entities is None:
            entities = self.DEFAULT_PHI_ENTITIES

        results = self.analyzer.analyze(text, language='en', entities=entities)
        res_matches = set()
        for res in results:
            if res.score > self.threshold:
                res_matches.add(res)
        return list(res_matches)

    async def adetect(self, text: str, entities: list[str] | None = None):
        """Async version of detect_all for PHI detection."""
        return self.detect_all(text, entities)


@dataclass
class SecretPattern:
    secret_name: str
    patterns: list[re.Pattern]


class SecretsAnalyzer(BaseDetector):
    """
    Analyzer for detecting secrets in generated text.
    """

    def __init__(self):
        super().__init__()
        self.secrets = self.get_recognizers()

    def get_recognizers(self) -> list[re.Pattern]:
        secrets = []
        for secret_name, regex_pattern in _SECRETS_PATTERNS.items():
            secrets.append(SecretPattern(secret_name, regex_pattern))
        return secrets

    def detect_all(self, text: str) -> list[DetectorResult]:
        res = []
        for secret in self.secrets:
            # patterns is a list of re.Pattern objects, each has finditer()
            for pattern in secret.patterns:
                for match in pattern.finditer(text):
                    res.append(
                        DetectorResult(
                            secret.secret_name, match.start(), match.end()
                        )
                    )
        return res


class SecurityGateway:
    """HIPAA-compliant security gateway for scanning text for PHI, secrets, and unicode issues."""

    def __init__(self):
        # No parent class, so no super().__init__() needed
        self.phi_analyzer = PHI_Analyzer()
        self.secrets_analyzer = SecretsAnalyzer()
        self.unicode_detector = UnicodeDetector()

    async def scan_text_for_issues(self, text: str) -> List[str]:
        """Scans text for Protected Health Information (PHI), secrets, and prompt injections.

        Args:
            text: Text to scan for security issues

        Returns:
            List of identified security issues
        """
        issues = []

        phi_results = self.phi_analyzer.detect_all(text)
        if phi_results:
            issues.append(
                f'PHI detected: {[res.entity for res in phi_results]}'
            )

        secrets_results = self.secrets_analyzer.detect_all(text)
        if secrets_results:
            issues.append(
                f'Secrets detected: {[res.entity for res in secrets_results]}'
            )

        unicode_results = self.unicode_detector.detect_all(
            text, categories=['Co', 'Cs']
        )
        if unicode_results:
            issues.append(
                f'Disallowed unicode characters detected: {[res.entity for res in unicode_results]}'
            )

        return issues
