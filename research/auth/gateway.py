import re
from dotenv import load_dotenv
from pydantic.dataclasses import dataclass
import re
from typing import TypeVar, List

from ._sanitize import (
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
MOCK_USER_DB = {}

Extra.extras = {}


class PII_Analyzer(BaseDetector):
    def __init__(self, threshold=0.5):
        AnalyzerEngine = PRESIDIO_EXTRA.package(
            'presidio_analyzer'
        ).import_names('AnalyzerEngine')
        self.analyzer = AnalyzerEngine()
        self.threshold = threshold

    def detect_all(self, text: str, entities: list[str] | None = None):
        results = self.analyzer.analyze(text, language='en', entities=entities)
        res_matches = set()
        for res in results:
            if res.score > self.threshold:
                res_matches.add(res)
        return list(res_matches)

    async def adetect(self, text: str, entities: list[str] | None = None):
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


class MCPSecurityGateway:
    """Security gateway for scanning text for PII, secrets, and unicode issues."""

    def __init__(self):
        # No parent class, so no super().__init__() needed
        self.pii_analyzer = PII_Analyzer()
        self.secrets_analyzer = SecretsAnalyzer()
        self.unicode_detector = UnicodeDetector()

    async def scan_text_for_issues(self, text: str) -> List[str]:
        """Scans text for PII, secrets, and prompt injections."""
        issues = []

        pii_results = self.pii_analyzer.detect_all(text)
        if pii_results:
            issues.append(
                f'PII detected: {[res.entity for res in pii_results]}'
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
