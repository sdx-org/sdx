import asyncio
from dataclasses import dataclass, field
from typing import List, Tuple
from .gateway import (
    DetectorResult,
    PHI_Analyzer,
    SecretsAnalyzer,
)
from .gateway import SecurityGateway

gateway = SecurityGateway()

@dataclass
class ScanResult:
    is_safe: bool
    sanitized_input: str
    findings: List[DetectorResult] = field(default_factory=list)
    is_sanitized: bool = False

class InputScanner:
    """HIPAA-compliant scanner for detecting PHI and secrets in text before LLM submission."""

    def __init__(self):
        self.phi_detector = PHI_Analyzer()
        self.secrets_detector = SecretsAnalyzer()

    async def scan(self, text: str) -> ScanResult:
        """Scans user input for PHI and secrets, redacting sensitive data.

        Args:
            text: The input string to scan for Protected Health Information.

        Returns:
            A ScanResult object with sanitized text and findings.
        """
        try:
            # Run detectors concurrently - PHI async, secrets in thread
            phi_task = asyncio.create_task(self.phi_detector.adetect(text))
            secrets_task = asyncio.to_thread(
                self.secrets_detector.detect_all, text
            )
            phi_results, secrets_results = await asyncio.gather(
                phi_task, secrets_task
            )

            all_findings = (phi_results or []) + (secrets_results or [])
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
            return ScanResult(
                is_safe=False, sanitized_input='Error during security scan.'
            )

    @staticmethod
    def _sanitize_text(text: str, findings: List[DetectorResult]) -> str:
        """
        Robustly redacts findings from text without index drift.

        Merges overlapping findings and builds output string without
        mutating the input list or causing index misalignment.
        """
        if not findings:
            return text

        n = len(text)
        # Normalize and collect spans
        spans: List[tuple[int, int, str]] = []
        for f in findings:
            s = max(0, min(f.start, n))
            e = max(s, min(f.end, n))
            ent = f.entity or 'SENSITIVE'
            spans.append((s, e, ent))

        # Sort by start position
        spans.sort(key=lambda t: (t[0], t[1]))

        # Merge overlapping spans
        merged: List[tuple[int, int, str]] = []
        for s, e, ent in spans:
            if merged and s <= merged[-1][1]:
                # Overlapping - extend the previous span
                merged[-1] = (
                    merged[-1][0],
                    max(merged[-1][1], e),
                    merged[-1][2],
                )
            else:
                merged.append((s, e, ent))

        out: List[str] = []
        cursor = 0
        for s, e, ent in merged:
            out.append(text[cursor:s])
            out.append(f'[{ent}_REDACTED]')
            cursor = e
        out.append(text[cursor:])

        return ''.join(out)


def detect(text: str):
    """Helper function to detect and sanitize PHI in text.

    Args:
        text: Input text to scan for Protected Health Information

    Returns:
        Sanitized text with PHI redacted
    """
    phi_results = gateway.phi_analyzer.detect_all(text)
    secrets_results = gateway.secrets_analyzer.detect_all(text)
    all_findings = (phi_results or []) + (secrets_results or [])
    sanitized_description = InputScanner._sanitize_text(text, all_findings)
    return sanitized_description
