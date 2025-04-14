"""Treatment Suggestion Module."""

from __future__ import annotations

import os

from typing import Any, Dict, Optional, cast

from anamnesisai.openai import extract_fhir
from rago.augmented.base import AugmentedBase
from rago.generation.openai import OpenAIGen

from sdx.medical_reports import get_report_data_from_pdf


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


def _fhir_dict_to_strings(fhir_data: Dict[str, Any]) -> list[str]:
    """Convert a dict of FHIR resources into a list of descriptive strings."""
    context_list = []
    if not fhir_data:
        return ['No clinical data provided.']

    for resource_type, resource_content in fhir_data.items():
        if isinstance(resource_content, dict):
            details = []
            for key, value in resource_content.items():
                if isinstance(value, (dict, list)):
                    details.append(f'{key}: [complex data]')
                elif value is not None:
                    details.append(f'{key}: {value!s}')
            context_list.append(
                f'ResourceType {resource_type}: {"; ".join(details)}'
            )

        elif isinstance(resource_content, list):
            for i, item in enumerate(resource_content):
                if isinstance(item, dict):
                    details = []
                    for key, value in item.items():
                        if isinstance(value, (dict, list)):
                            details.append(f'{key}: [complex data]')
                        elif value is not None:
                            details.append(f'{key}: {value!s}')
                    detail_str = f'ResourceType {resource_type} [{i}]: '
                    detail_str += f'{"; ".join(details)}'
                    context_list.append(detail_str)
                else:
                    context_list.append(
                        f'ResourceType {resource_type} [{i}]: {item!s}'
                    )
        else:
            context_list.append(
                f'ResourceType {resource_type}: {resource_content!s}'
            )

    print(f'Converted FHIR dict to {len(context_list)} context strings.')
    return context_list


GENERATION_PROMPT_TEMPLATE = """Please provide a suggested treatment plan for a
patient with the following clinical records:
---
{context}
---
Suggested Treatment Plan:"""


def suggest_treatment_from_pdf(
    pdf_path: str,
    openai_api_key: Optional[str] = None,
    model_name: str = 'gpt-3.5-turbo',
    augmenter: Optional[AugmentedBase] = None,
) -> Dict[str, Any]:
    """Extract FHIR, augment context, generate suggestion, convert to FHIR."""
    print(f'Starting FHIR treatment suggestion for PDF: {pdf_path}')

    resolved_api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
    if not resolved_api_key:
        raise ValueError('OpenAI API Key is required.')

    if augmenter is None:
        augmenter = PassThroughAugmented()

    try:
        extracted_fhir_data = get_report_data_from_pdf(
            pdf_path, api_key=resolved_api_key
        )
        if not extracted_fhir_data:
            print('Warning: No initial FHIR data extracted.')
            error_msg = 'Cannot suggest treatment: '
            error_msg += 'No FHIR data extracted from PDF.'
            return {'error': error_msg}

        context_strings = _fhir_dict_to_strings(extracted_fhir_data)
        if not context_strings or context_strings == [
            'No clinical data provided.'
        ]:
            error_msg = 'Cannot suggest treatment: '
            error_msg += 'No clinical data available for context.'
            return {'error': error_msg}

        augmented_context_list = augmenter.search(
            query='', documents=context_strings, top_k=len(context_strings)
        )
        context_for_generation = '\n'.join(augmented_context_list)

        generator = OpenAIGen(api_key=resolved_api_key, model_name=model_name)
        final_prompt = GENERATION_PROMPT_TEMPLATE.format(
            context=context_for_generation
        )
        text_suggestion = generator.generate(query=final_prompt, context=[])
        if not text_suggestion:
            print('Warning: AI model returned an empty suggestion.')
            error_msg = 'Cannot produce FHIR suggestion: '
            error_msg += 'AI model returned empty text.'
            return {'error': error_msg}

        fhir_suggestion = extract_fhir(
            str(text_suggestion), api_key=resolved_api_key
        )

        return fhir_suggestion

    except (FileNotFoundError, ValueError, EnvironmentError, ImportError) as e:
        print(f'Error during treatment suggestion pipeline: {e}')
        raise e
    except Exception as e:
        print(f'An unexpected error occurred during treatment suggestion: {e}')
        return {'error': f'An unexpected error occurred: {e}'}
