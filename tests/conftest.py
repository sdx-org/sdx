"""Pytest configuration for the sdx package tests."""

from __future__ import annotations

import os
import warnings

from pathlib import Path

import pytest

from dotenv import dotenv_values, load_dotenv


@pytest.fixture
def env() -> dict[str, str | None]:
    """Return a fixture for the environment variables from .env file."""
    dotenv_path = Path(__file__).parent / '.env'
    if not dotenv_path.exists():
        warnings.warn(
            f"'.env' file not found at {dotenv_path}. Some "
            'tests requiring environment variables might fail or be skipped.'
        )
        return {}

    load_dotenv(dotenv_path=dotenv_path)
    return dotenv_values(dotenv_path)


@pytest.fixture
def test_data_dir() -> Path:
    """Fixture providing the path to the test data directory."""
    return Path(__file__).parent / 'data'


@pytest.fixture
def reports_data_dir(test_data_dir: Path) -> Path:
    """Fixture providing the path to the test reports directory."""
    reports_dir = test_data_dir / 'reports'
    if not reports_dir.exists():
        reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


@pytest.fixture
def api_key_openai(env: dict[str, str | None]) -> str:
    """Fixture providing the OpenAI API key from environment variables."""
    api_key = env.get('OPENAI_API_KEY')

    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')

    if os.getenv('CI') and not api_key:
        pytest.skip(
            'Running in CI environment without OpenAI API key - skipping test'
        )

    if not api_key:
        pytest.skip('OpenAI API key not available - skipping test')

    return api_key
