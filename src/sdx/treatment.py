"""Treatment Suggestion Module."""

from __future__ import annotations

import logging
import os

from typing import Any, Dict, Optional, cast

from anamnesisai.openai import extract_fhir
from rago.augmented.base import AugmentedBase
from rago.generation.openai import OpenAIGen

from sdx.medical_reports import get_report_data_from_pdf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class TreatmentSuggestionError(Exception):
    """Custom exception for treatment suggestion errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class PassThroughAugmented(AugmentedBase):
    """A simple augmenter class that performs no actual augmentation."""

    def search(self, query: str, documents: Any, top_k: int = 10) -> list[str]:
        """Perform synchronous search pass-through."""
        _ = query, top_k
        return cast(list[str], documents)

    async def asearch(
        self, query: str, documents: Any, top_k: int = 10
    ) -> list[str]:
        """Perform asynchronous search pass-through."""
        _ = query, top_k
        return cast(list[str], documents)


def _format_resource_details(item: Dict[str, Any]) -> str:
    """Format a resource dictionary into a string."""
    details = [
        f'{key}: {value!s}'
        if not isinstance(value, (dict, list)) and value is not None
        else f'{key}: [complex data]'
        for key, value in item.items()
    ]
    return '; '.join(details)


def _fhir_dict_to_strings(fhir_data: Dict[str, Any]) -> list[str]:
    """Convert a dict of FHIR resources into a list of descriptive strings."""
    if not fhir_data:
        return ['No clinical data provided.']

    context_list = []

    for resource_type, resource_content in fhir_data.items():
        if isinstance(resource_content, dict):
            details = _format_resource_details(resource_content)
            context_list.append(f'ResourceType {resource_type}: {details}')

        elif isinstance(resource_content, list):
            context_list.extend(
                [
                    f'ResourceType {resource_type} [{i}]: '
                    f'{_format_resource_details(item)}'
                    if isinstance(item, dict)
                    else f'ResourceType {resource_type} [{i}]: {item!s}'
                    for i, item in enumerate(resource_content)
                ]
            )

        else:
            context_list.append(
                f'ResourceType {resource_type}: {resource_content!s}'
            )

    logger.info(f'Converted FHIR dict to {len(context_list)} context strings.')
    return context_list


GENERATION_PROMPT_TEMPLATE = """Please provide a suggested treatment plan for a
patient with the following clinical records:
---
{context}
---
Suggested Treatment Plan:"""


def validate_fhir_data(
    data: Optional[Dict[str, Any]], error_msg: Optional[str] = None
) -> Dict[str, Any]:
    """Validate FHIR data and raise custom exception if invalid."""
    if not data:
        default_msg = 'No clinical data provided.'
        raise TreatmentSuggestionError(error_msg or default_msg)
    return data


def suggest_treatment_from_pdf(
    pdf_path: str,
    openai_api_key: Optional[str] = None,
    model_name: str = 'gpt-3.5-turbo',
    augmenter: Optional[AugmentedBase] = None,
) -> Dict[str, Any]:
    """Extract FHIR, augment context, generate suggestion, convert to FHIR."""
    logger.info(f'Starting FHIR treatment suggestion for PDF: {pdf_path}')

    resolved_api_key = os.environ.get('OPENAI_API_KEY') or openai_api_key
    if not resolved_api_key:
        raise ValueError('OpenAI API Key is required.')

    if augmenter is None:
        augmenter = PassThroughAugmented()

    try:
        extracted_fhir_data = get_report_data_from_pdf(
            pdf_path, api_key=resolved_api_key
        )

        try:
            validate_fhir_data(
                extracted_fhir_data,
                'Cannot suggest treatment: No FHIR data extracted from PDF.',
            )
        except TreatmentSuggestionError as e:
            logger.warning(str(e))
            return {'error': str(e)}

        context_strings = _fhir_dict_to_strings(extracted_fhir_data)

        try:
            validate_fhir_data(
                None
                if context_strings == ['No clinical data provided.']
                else {'data': context_strings},
                'Cannot suggest treatment: No clinical data available '
                'for context.',
            )
        except TreatmentSuggestionError as e:
            logger.warning(str(e))
            return {'error': str(e)}

        augmented_context_list = augmenter.search(
            query='', documents=context_strings, top_k=len(context_strings)
        )
        context_for_generation = '\n'.join(augmented_context_list)

        generator = OpenAIGen(api_key=resolved_api_key, model_name=model_name)
        final_prompt = GENERATION_PROMPT_TEMPLATE.format(
            context=context_for_generation
        )
        text_suggestion = generator.generate(query=final_prompt, context=[])

        try:
            if not text_suggestion:
                raise TreatmentSuggestionError(
                    'Cannot produce FHIR suggestion: AI model returned '
                    'empty text.'
                )
        except TreatmentSuggestionError as e:
            logger.warning(str(e))
            return {'error': str(e)}

        fhir_suggestion = extract_fhir(
            str(text_suggestion), api_key=resolved_api_key
        )

        return fhir_suggestion

    except (FileNotFoundError, ValueError, EnvironmentError, ImportError) as e:
        logger.error(f'Error during treatment suggestion pipeline: {e}')
        raise e
    except Exception as e:
        logger.error(
            f'An unexpected error occurred during treatment suggestion: {e}'
        )
        return {'error': f'An unexpected error occurred: {e}'}
