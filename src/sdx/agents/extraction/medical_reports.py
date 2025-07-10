"""Medical reports module for extracting FHIR data from PDF documents."""

from __future__ import annotations

import os

from pathlib import Path
from typing import Any, Dict, Optional, Union

from anamnesisai.openai import extract_fhir
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """Extract text content from a PDF file."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f'PDF file not found: {pdf_path}')

    try:
        reader = PdfReader(pdf_path)
        text_content = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content.append(page_text)

        if not text_content:
            raise ValueError(f'No extractable text found in PDF: {pdf_path}')

        return '\n'.join(text_content)

    except Exception as e:
        raise ValueError(f'Error reading PDF {pdf_path}: {e!s}') from e


def get_report_data_from_pdf(
    pdf_path: Union[str, Path], api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Extract FHIR data from a medical PDF report."""
    # Use OpenRouter API key if available, fallback to OpenAI
    api_key = api_key or os.environ.get('OPENROUTER_API_KEY') or os.environ.get('OPENAI_API_KEY')

    if not api_key:
        raise EnvironmentError(
            'API key is required. Provide it as an argument or '
            'set the OPENROUTER_API_KEY or OPENAI_API_KEY environment variable.'
        )

    text_content = extract_text_from_pdf(pdf_path)

    try:
        # Note: The anamnesisai library might need to be updated to support OpenRouter
        # For now, we'll use the existing extract_fhir function
        fhir_resources = extract_fhir(text_content, api_key)

        return {
            resource_type: resource.model_dump()
            for resource_type, resource in fhir_resources.items()
        }

    except Exception as e:
        raise ValueError(f'Failed to convert PDF to FHIR: {e!s}') from e
