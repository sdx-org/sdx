"""
title: Unit tests for de-identification helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, ClassVar

import hiperhealth.skills.privacy.deidentifier as deid_mod
import pytest


class _NonPatternRecognizer:
    """
    title: Recognizer type that should never be removed by replacement logic.
    attributes:
      supported_entities:
        type: ClassVar[list[str]]
        description: Value for supported_entities.
    """

    supported_entities: ClassVar[list[str]] = ['PERSON']


class _FakePattern:
    """
    title: Simple stand-in for Presidio Pattern.
    attributes:
      name:
        description: Value for name.
      regex:
        description: Value for regex.
      score:
        description: Value for score.
    """

    def __init__(self, name: str, regex: str, score: float) -> None:
        self.name = name
        self.regex = regex
        self.score = score


class _FakePatternRecognizer:
    """
    title: Simple stand-in for Presidio PatternRecognizer.
    attributes:
      supported_entities:
        description: Value for supported_entities.
      patterns:
        description: Value for patterns.
    """

    def __init__(self, supported_entity: str, patterns: list[Any]) -> None:
        self.supported_entities = [supported_entity]
        self.patterns = patterns


class _FakeRegistry:
    """
    title: Minimal registry object used by fake analyzer.
    attributes:
      recognizers:
        description: Value for recognizers.
      added:
        type: list[Any]
        description: Value for added.
    """

    def __init__(self, recognizers: list[Any]) -> None:
        self.recognizers = recognizers
        self.added: list[Any] = []

    def get_recognizers(self, language: str, all_fields: bool) -> list[Any]:
        """
        title: Return recognizers currently in registry.
        parameters:
          language:
            type: str
            description: Value for language.
          all_fields:
            type: bool
            description: Value for all_fields.
        returns:
          type: list[Any]
          description: Return value.
        """
        assert language
        assert all_fields is True
        return list(self.recognizers)

    def add_recognizer(self, recognizer: Any) -> None:
        """
        title: Track newly added recognizer.
        parameters:
          recognizer:
            type: Any
            description: Value for recognizer.
        """
        self.added.append(recognizer)
        self.recognizers.append(recognizer)


class _FakeAnalyzerEngine:
    """
    title: Minimal analyzer with configurable results.
    attributes:
      _results:
        description: Value for _results.
      registry:
        description: Value for registry.
      calls:
        type: list[dict[str, Any]]
        description: Value for calls.
    """

    def __init__(self, results: list[Any], recognizers: list[Any]) -> None:
        self._results = results
        self.registry = _FakeRegistry(recognizers)
        self.calls: list[dict[str, Any]] = []

    def analyze(
        self, text: str, entities: list[str] | None, language: str
    ) -> list[Any]:
        """
        title: Return preloaded analyzer results.
        parameters:
          text:
            type: str
            description: Value for text.
          entities:
            type: list[str] | None
            description: Value for entities.
          language:
            type: str
            description: Value for language.
        returns:
          type: list[Any]
          description: Return value.
        """
        self.calls.append(
            {'text': text, 'entities': entities, 'language': language}
        )
        return list(self._results)


class _FakeAnonymizerEngine:
    """
    title: Minimal anonymizer returning static result.
    attributes:
      anonymized_text:
        description: Value for anonymized_text.
      calls:
        type: list[dict[str, Any]]
        description: Value for calls.
    """

    def __init__(self, anonymized_text: str) -> None:
        self.anonymized_text = anonymized_text
        self.calls: list[dict[str, Any]] = []

    def anonymize(
        self, text: str, analyzer_results: list[Any], operators: Any
    ) -> SimpleNamespace:
        """
        title: Return fixed anonymized text and store call args.
        parameters:
          text:
            type: str
            description: Value for text.
          analyzer_results:
            type: list[Any]
            description: Value for analyzer_results.
          operators:
            type: Any
            description: Value for operators.
        returns:
          type: SimpleNamespace
          description: Return value.
        """
        self.calls.append(
            {
                'text': text,
                'analyzer_results': analyzer_results,
                'operators': operators,
            }
        )
        return SimpleNamespace(text=self.anonymized_text)


@dataclass
class _FakeRecognizerResult:
    """
    title: Result object used by Deidentifier.deidentify.
    attributes:
      start:
        type: int
        description: Value for start.
      end:
        type: int
        description: Value for end.
      entity_type:
        type: str
        description: Value for entity_type.
    """

    start: int
    end: int
    entity_type: str = 'PII'


def _build_deidentifier(
    monkeypatch: pytest.MonkeyPatch,
    *,
    results: list[Any] | None = None,
    recognizers: list[Any] | None = None,
    anonymized_text: str = 'hashed',
) -> tuple[deid_mod.Deidentifier, _FakeAnalyzerEngine, _FakeAnonymizerEngine]:
    """
    title: Create a Deidentifier wired with in-memory fake engines.
    parameters:
      monkeypatch:
        type: pytest.MonkeyPatch
        description: Value for monkeypatch.
      results:
        type: list[Any] | None
        description: Value for results.
      recognizers:
        type: list[Any] | None
        description: Value for recognizers.
      anonymized_text:
        type: str
        description: Value for anonymized_text.
    returns:
      type: >-
        tuple[deid_mod.Deidentifier, _FakeAnalyzerEngine,
        _FakeAnonymizerEngine]
      description: Return value.
    """
    fake_analyzer = _FakeAnalyzerEngine(results or [], recognizers or [])
    fake_anonymizer = _FakeAnonymizerEngine(anonymized_text)

    monkeypatch.setattr(deid_mod, 'AnalyzerEngine', lambda: fake_analyzer)
    monkeypatch.setattr(deid_mod, 'AnonymizerEngine', lambda: fake_anonymizer)
    monkeypatch.setattr(deid_mod, 'Pattern', _FakePattern)
    monkeypatch.setattr(deid_mod, 'PatternRecognizer', _FakePatternRecognizer)

    return deid_mod.Deidentifier(), fake_analyzer, fake_anonymizer


def test_add_custom_recognizer_replaces_existing_pattern(monkeypatch):
    """
    title: Replacing recognizers keeps non-pattern recognizers untouched.
    parameters:
      monkeypatch:
        description: Value for monkeypatch.
    """
    existing = _FakePatternRecognizer('ORDER_ID', [])
    survivor = _NonPatternRecognizer()
    deid, analyzer, _ = _build_deidentifier(
        monkeypatch, recognizers=[existing, survivor]
    )

    deid.add_custom_recognizer('ORDER_ID', r'ORD-\d{4}', score=0.9)

    assert survivor in analyzer.registry.recognizers
    assert existing not in analyzer.registry.recognizers
    assert len(analyzer.registry.added) == 1
    assert analyzer.registry.added[0].supported_entities == ['ORDER_ID']


def test_add_custom_recognizer_rejects_out_of_range_score(monkeypatch):
    """
    title: Custom recognizer score must be between 0 and 1.
    parameters:
      monkeypatch:
        description: Value for monkeypatch.
    """
    deid, _, _ = _build_deidentifier(monkeypatch)
    with pytest.raises(
        ValueError, match=r'Score must be between 0\.0 and 1\.0'
    ):
        deid.add_custom_recognizer('ORDER_ID', r'ORD-\d{4}', score=1.2)


def test_analyze_delegates_to_engine(monkeypatch):
    """
    title: Analyze should pass parameters directly to analyzer engine.
    parameters:
      monkeypatch:
        description: Value for monkeypatch.
    """
    expected = [_FakeRecognizerResult(0, 4, entity_type='EMAIL')]
    deid, analyzer, _ = _build_deidentifier(monkeypatch, results=expected)

    out = deid.analyze(
        text='user@example.com', entities=['EMAIL_ADDRESS'], language='pt'
    )

    assert out == expected
    assert analyzer.calls == [
        {
            'text': 'user@example.com',
            'entities': ['EMAIL_ADDRESS'],
            'language': 'pt',
        }
    ]


def test_deidentify_rejects_unknown_strategy(monkeypatch):
    """
    title: Unknown de-identification strategy should raise ValueError.
    parameters:
      monkeypatch:
        description: Value for monkeypatch.
    """
    deid, _, _ = _build_deidentifier(monkeypatch)
    with pytest.raises(ValueError, match="Unsupported strategy: 'encrypt'"):
        deid.deidentify('text', strategy='encrypt')


def test_deidentify_returns_original_when_no_entities(monkeypatch):
    """
    title: No detected entities should keep input text unchanged.
    parameters:
      monkeypatch:
        description: Value for monkeypatch.
    """
    deid, _, _ = _build_deidentifier(monkeypatch, results=[])
    assert deid.deidentify('nothing to redact', strategy='mask') == (
        'nothing to redact'
    )


def test_deidentify_mask_strategy(monkeypatch):
    """
    title: Mask strategy should replace detected slices with same-length stars.
    parameters:
      monkeypatch:
        description: Value for monkeypatch.
    """
    pii = [_FakeRecognizerResult(0, 4), _FakeRecognizerResult(10, 13)]
    deid, _, _ = _build_deidentifier(monkeypatch, results=pii)

    assert deid.deidentify('John says Bob', strategy='mask') == '**** says ***'


def test_deidentify_hash_strategy_uses_anonymizer_engine(monkeypatch):
    """
    title: Hash strategy should delegate to anonymizer with hash operator.
    parameters:
      monkeypatch:
        description: Value for monkeypatch.
    """
    pii = [_FakeRecognizerResult(0, 4)]
    deid, _, anonymizer = _build_deidentifier(
        monkeypatch, results=pii, anonymized_text='HASHED'
    )

    out = deid.deidentify('John', strategy='hash')

    assert out == 'HASHED'
    assert len(anonymizer.calls) == 1
    call = anonymizer.calls[0]
    assert call['text'] == 'John'
    assert call['analyzer_results'] == pii
    assert 'DEFAULT' in call['operators']


def test_deidentify_patient_record_recursively():
    """
    title: Only configured free-text keys should be de-identified recursively.
    """

    class _StubDeidentifier:
        def deidentify(self, text: str) -> str:
            return f'<redacted:{text}>'

    record = {
        'symptoms': 'John Doe with headache',
        'patient_name': 'John Doe',
        'nested': {
            'comments': 'Lives at 123 Main St',
            'age': 40,
            'inner': {'summary': 'Phone 555-0100', 'safe_key': 'keep me'},
        },
    }

    out = deid_mod.deidentify_patient_record(record, _StubDeidentifier())

    assert out is record
    assert out['symptoms'] == '<redacted:John Doe with headache>'
    assert out['patient_name'] == 'John Doe'
    assert out['nested']['comments'] == '<redacted:Lives at 123 Main St>'
    assert out['nested']['inner']['summary'] == '<redacted:Phone 555-0100>'
    assert out['nested']['inner']['safe_key'] == 'keep me'
