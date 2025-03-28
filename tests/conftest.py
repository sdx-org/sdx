"""Pytest configuration for the sdx package tests."""

import os

from pathlib import Path

import pytest

from dotenv import load_dotenv


def pytest_configure(config):
    """Configure pytest environment."""
    env_path = Path(__file__).parent / '.env'

    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        tpl_path = Path(__file__).parent / '.env.tpl'
        if tpl_path.exists():
            load_dotenv(dotenv_path=tpl_path)


@pytest.fixture
def test_data_dir():
    """Fixture providing the path to the test data directory."""
    return Path(__file__).parent / 'data'


@pytest.fixture
def reports_data_dir(test_data_dir):
    """Fixture providing the path to the test reports directory."""
    reports_dir = test_data_dir / 'reports'
    if not reports_dir.exists():
        reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


@pytest.fixture
def openai_api_key():
    """Fixture providing the OpenAI API key from environment variables."""
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        pytest.skip('OpenAI API key not available for testing')

    return api_key
