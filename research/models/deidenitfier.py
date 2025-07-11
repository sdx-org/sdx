"""A module for PII detection and de-identification."""

from typing import Dict, List, Optional

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


class Deidentifier:
    """Class for PII detection and de-identification using Presidio."""

    def __init__(self):
        """Initialize the Presidio Analyzer and Anonymizer engines."""
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def add_custom_recognizer(
        self, entity_name: str, regex_pattern: str, score: float = 0.85
    ):
        """
        Add a custom PII entity recognizer using a regular expression.

        Args:
            entity_name: The name for the new entity (e.g., "CUSTOM_ID").
            regex_pattern: The regex pattern to detect the entity.
            score: The confidence score for the detection (0.0 to 1.0).
        """
        if not (0.0 <= score <= 1.0):
            raise ValueError('Score must be between 0.0 and 1.0.')

        # Create a recognizer from the provided pattern
        custom_recognizer = PatternRecognizer(
            supported_entity=entity_name,
            patterns=[
                Pattern(name=entity_name, regex=regex_pattern, score=score)
            ],
        )

        self.analyzer.registry.add_recognizer(custom_recognizer)
        print(f"Custom recognizer '{entity_name}' added successfully.")

    def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        language: str = 'en',
    ):
        """
        Analyze text to detect and locate PII entities.

        Args
            text: The input text to be analyzed.
            entities: An optional list of specific entities to search for.
                      If None, all available entities will be used.
            language: The language of the text (ISO 639-1 code).

        Returns
        -------
            A list of PII entities found by the Presidio analyzer.
        """
        return self.analyzer.analyze(
            text=text, entities=entities, language=language
        )

    def deidentify(self, text: str, strategy: str = 'redact') -> str:
        """
        Anonymize detected PII in the text using a specified strategy.

        Args
            text: The text to de-identify.
            strategy: The anonymization strategy. Supported options are:
                    'mask', 'hash', and 'redact'.

        Returns
        -------
            The de-identified text as a string.
        """
        analyzer_results = self.analyze(text)

        # A dictionary-based approach to select the operator configuration.
        # This is more scalable and readable than multiple if-elif statements.
        strategy_configs: Dict[str, Dict] = {
            'mask': {
                'DEFAULT': OperatorConfig(
                    'mask',
                    {
                        'type': 'mask',
                        'masking_char': '*',
                        # Mask a significant portion for security
                        'chars_to_mask': 15,
                        'from_end': False,
                    },
                )
            },
            'hash': {
                'DEFAULT': OperatorConfig(
                    operator_name='hash', params={'hash_type': 'sha256'}
                )
            },
            'redact': {'DEFAULT': OperatorConfig('redact')},
        }

        operators = strategy_configs.get(strategy)
        if not operators:
            raise ValueError(
                f"Unsupported strategy: '{strategy}'. "
                f'Available options are: {", ".join(strategy_configs.keys())}'
            )

        anonymized_result = self.anonymizer.anonymize(
            text=text, analyzer_results=analyzer_results, operators=operators
        )
        return anonymized_result.text
