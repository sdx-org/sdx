"""
title: Pytest configuration for the hiperhealth package tests.
"""

from __future__ import annotations

import os
import warnings

from pathlib import Path

import pytest

from dotenv import dotenv_values, load_dotenv
from hiperhealth.agents.extraction.medical_reports import (
    MedicalReportFileExtractor,
)
from hiperhealth.agents.extraction.wearable import WearableDataFileExtractor


@pytest.fixture
def env() -> dict[str, str | None]:
    """
    title: Return a fixture for the environment variables from .env file.
    returns:
      type: dict[str, str | None]
      description: Return value.
    """
    # This assumes a .envs/.env file at the project root
    dotenv_path = Path(__file__).parents[1] / '.envs' / '.env'
    if not dotenv_path.exists():
        warnings.warn(
            f"'.env' file not found at {dotenv_path}. Some "
            'tests requiring environment variables might fail or be skipped.'
        )
        return {}
    load_dotenv(dotenv_path=dotenv_path)
    return dotenv_values(dotenv_path)


@pytest.fixture
def api_key_openai(env: dict[str, str | None]) -> str | None:
    """
    title: Fixture providing the OpenAI API key. Skips test if not found.
    parameters:
      env:
        type: dict[str, str | None]
        description: Value for env.
    returns:
      type: str | None
      description: Return value.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        pytest.skip('OpenAI API key not available')
    return api_key


@pytest.fixture
def test_data_dir() -> Path:
    """
    title: Fixture providing the path to the test data directory.
    returns:
      type: Path
      description: Return value.
    """
    return Path(__file__).parent / 'data'


@pytest.fixture
def reports_pdf_dir(test_data_dir: Path) -> Path:
    """
    title: Fixture for the directory containing PDF report files.
    parameters:
      test_data_dir:
        type: Path
        description: Value for test_data_dir.
    returns:
      type: Path
      description: Return value.
    """
    return test_data_dir / 'reports' / 'pdf_reports'


@pytest.fixture
def reports_image_dir(test_data_dir: Path) -> Path:
    """
    title: Fixture for the directory containing image report files.
    parameters:
      test_data_dir:
        type: Path
        description: Value for test_data_dir.
    returns:
      type: Path
      description: Return value.
    """
    return test_data_dir / 'reports' / 'image_reports'


@pytest.fixture
def wearable_extractor():
    """
    title: Provide a WearableDataFileExtractor instance for tests.
    """
    return WearableDataFileExtractor()


@pytest.fixture
def medical_extractor():
    """
    title: Provide a MedicalReportFileExtractor instance for tests.
    """
    return MedicalReportFileExtractor()
