"""
title: A module for PII detection and de-identification.
"""

import logging

from typing import Dict, List, Optional

from presidio_analyzer import (
    AnalyzerEngine,
    Pattern,
    PatternRecognizer,
    RecognizerResult,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger(__name__)


class Deidentifier:
    """
    title: A class for PII detection and de-identification using Presidio.
    attributes:
      analyzer:
        description: Value for analyzer.
      anonymizer:
        description: Value for anonymizer.
    """

    def __init__(self) -> None:
        """
        title: Initialize the Presidio Analyzer and Anonymizer engines.
        """
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()  # type: ignore[no-untyped-call]

    def add_custom_recognizer(
        self,
        entity_name: str,
        regex_pattern: str,
        score: float = 0.85,
        language: str = 'en',
    ) -> None:
        """
        title: Add a custom PII entity recognizer using a regular expression.
        summary: |-
          If a recognizer for the same entity_name already exists, this method
                  replaces the old one to prevent duplicate definitions.

                  Args:
          entity_name: The name for the new entity (e.g., "CUSTOM_ID").
                      regex_pattern: The regex pattern to detect the entity.
          score: The confidence score for the detection (0.0 to 1.0).
                      language: The language for the recognizer registry.
        parameters:
          entity_name:
            type: str
            description: Value for entity_name.
          regex_pattern:
            type: str
            description: Value for regex_pattern.
          score:
            type: float
            description: Value for score.
          language:
            type: str
            description: Value for language.
        """
        if not (0.0 <= score <= 1.0):
            raise ValueError('Score must be between 0.0 and 1.0.')

        # To prevent duplicates, remove any existing PatternRecognizer with the
        # same name. This is done by rebuilding the list of recognizers.
        existing_recognizers = self.analyzer.registry.get_recognizers(
            language=language, all_fields=True
        )

        recognizers_to_keep = []
        for rec in existing_recognizers:
            # Using type() ensures we only check generic PatternRecognizers and
            # not specialized subclasses like the built-in CreditCardRecognizer
            if type(rec) is not PatternRecognizer:
                recognizers_to_keep.append(rec)
                continue

            # Keep recognizers that don't match the name of the new one.
            if entity_name not in rec.supported_entities:
                recognizers_to_keep.append(rec)

        # Replace the registry's list with the filtered list.
        self.analyzer.registry.recognizers = recognizers_to_keep

        # Add the new recognizer to the updated list.
        custom_recognizer = PatternRecognizer(
            supported_entity=entity_name,
            patterns=[
                Pattern(name=entity_name, regex=regex_pattern, score=score)
            ],
        )
        self.analyzer.registry.add_recognizer(custom_recognizer)
        logger.info(f"Custom recognizer '{entity_name}' added successfully.")

    def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        language: str = 'en',
    ) -> List[RecognizerResult]:
        """
        title: Analyze text to detect and locate PII entities.
        parameters:
          text:
            type: str
            description: Value for text.
          entities:
            type: Optional[List[str]]
            description: Value for entities.
          language:
            type: str
            description: Value for language.
        returns:
          type: List[RecognizerResult]
          description: Return value.
        """
        return self.analyzer.analyze(
            text=text, entities=entities, language=language
        )

    def deidentify(
        self, text: str, strategy: str = 'mask', language: str = 'en'
    ) -> str:
        """
        title: Anonymize detected PII in the text using a specified strategy.
        parameters:
          text:
            type: str
            description: Value for text.
          strategy:
            type: str
            description: Value for strategy.
          language:
            type: str
            description: Value for language.
        returns:
          type: str
          description: Return value.
        """
        # First, ensure the provided strategy is supported.
        supported_strategies = ['mask', 'hash']
        if strategy not in supported_strategies:
            raise ValueError(
                f"Unsupported strategy: '{strategy}'. "
                f'Available options are: {", ".join(supported_strategies)}'
            )

        analyzer_results = self.analyze(text, language=language)

        if not analyzer_results:
            return text

        # The 'mask' strategy is handled manually to ensure the mask's length
        # dynamically matches the original PII token's length.
        if strategy == 'mask':
            sorted_results = sorted(
                analyzer_results, key=lambda x: x.end, reverse=True
            )
            anonymized_text = text
            for res in sorted_results:
                anonymized_text = (
                    anonymized_text[: res.start]
                    + '*' * (res.end - res.start)
                    + anonymized_text[res.end :]
                )
            return anonymized_text

        # Other strategies are handled by the AnonymizerEngine.
        operators = {
            'hash': {
                'DEFAULT': OperatorConfig('hash', {'hash_type': 'sha256'})
            }
        }

        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=analyzer_results,  # type: ignore[arg-type]
            operators=operators.get(strategy),
        )
        return anonymized_result.text


def deidentify_patient_record(
    record: Dict[str, object], deidentifier: Deidentifier
) -> Dict[str, object]:
    """
    title: Recursively find and de-identify string values in a patient record.
    summary: |-
      Args:
              record: The patient data dictionary.
              deidentifier: An instance of the Deidentifier class.
    parameters:
      record:
        type: Dict[str, object]
        description: Value for record.
      deidentifier:
        type: Deidentifier
        description: Value for deidentifier.
    returns:
      type: Dict[str, object]
      description: Return value.
    """
    # Define which keys contain free-text that needs to be scanned
    keys_to_deidentify = {
        'symptoms',
        'physical_activity',
        'mental_exercises',
        'mental_health',
        'previous_tests',
        'summary',
        'comments',
    }

    for key, value in record.items():
        if isinstance(value, dict):
            # If the value is a dictionary, recurse into it
            deidentify_patient_record(value, deidentifier)
        elif isinstance(value, str) and key in keys_to_deidentify:
            # If it's a string and its key is in our target list, de-identify
            record[key] = deidentifier.deidentify(value)

    return record
