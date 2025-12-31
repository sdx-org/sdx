import asyncio
from dataclasses import dataclass, field
from typing import List, Tuple
from .gateway import (
    DetectorResult,
    PII_Analyzer,
    SecretsAnalyzer,
)
from .gateway import MCPSecurityGateway

gateway = MCPSecurityGateway()

@dataclass
class ScanResult:
    is_safe: bool
    sanitized_input: str
    findings: List[DetectorResult] = field(default_factory=list)
    is_sanitized: bool = False

class InputScanner:
    """Scans and sanitizes text inputs before they are sent to an LLM."""

    def __init__(
            self
    ):
        self.pii_detector = PII_Analyzer()
        self.secrets_detector = SecretsAnalyzer()

    async def scan(
            self,
            text: str
    ) -> ScanResult:
        """
        Scans user input for threats and redacts sensitive data.

        Args:
            text: The input string to scan.

        Returns:
            A ScanResult object containing the outcome.
        """
        try:
            # Run async detectors in parallel
            pii_results = await self.pii_detector.adetect(text)
            secrets_results = self.secrets_detector.detect_all(text)

            all_findings = (pii_results or []) + (secrets_results or [])
            if not all_findings:
                return ScanResult(is_safe=True, sanitized_input=text)

            sanitized_input = self._sanitize_text(text, all_findings)

            return ScanResult(
                is_safe=True,
                sanitized_input=sanitized_input,
                findings=all_findings,
                is_sanitized=True,
            )

        except Exception:
            return ScanResult(is_safe=False, sanitized_input="Error during security scan.")

    @staticmethod
    def _sanitize_text(
        text: str,
        findings: List[DetectorResult]
    ) -> str:
        """
        Robustly redacts findings from text.

        It sorts findings by start index in reverse to avoid issues with
        string index changes during replacement.
        """
        if not findings:
            return text
        findings.sort(key=lambda f: f.start, reverse=True)

        sanitized_text = list(text)
        for finding in findings:
            redaction_str = f"[{finding.entity}_REDACTED]"
            sanitized_text[finding.start : finding.end] = redaction_str

        return "".join(sanitized_text)


def detect(text :str):
    pii_results = gateway.pii_analyzer.detect_all(text)
    secrets_results = gateway.secrets_analyzer.detect_all(text)

    all_findings = (pii_results or []) + (secrets_results or [])
    sanitized_description = InputScanner._sanitize_text(text, all_findings)

    return sanitized_description

