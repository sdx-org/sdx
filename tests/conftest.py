"""Pytest configuration for the hiperhealth package tests."""

# ruff: noqa: E402
from __future__ import annotations

import json
import os
import sys
import types
import warnings

from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).parents[1] / 'research' / 'backend'
sys.path.insert(0, str(BACKEND_DIR))


def _install_magic_stub_if_unavailable() -> None:
    """Fallback stub for environments without native libmagic."""
    try:
        import magic  # noqa: F401
    except ImportError:
        stub = types.ModuleType('magic')

        class _MagicStub:
            def __init__(self, mime: bool = True):
                self.mime = mime

            def from_file(self, filename: str) -> str:
                lower = filename.lower()
                if lower.endswith('.pdf'):
                    return 'application/pdf'
                if lower.endswith('.png'):
                    return 'image/png'
                if lower.endswith('.jpg') or lower.endswith('.jpeg'):
                    return 'image/jpeg'
                return 'application/octet-stream'

            def from_buffer(self, data: bytes) -> str:
                if data.startswith(b'%PDF'):
                    return 'application/pdf'
                if data.startswith(b'\x89PNG\r\n\x1a\n'):
                    return 'image/png'
                if data.startswith(b'\xff\xd8\xff'):
                    return 'image/jpeg'
                return 'application/octet-stream'

        stub.Magic = _MagicStub
        sys.modules['magic'] = stub


_install_magic_stub_if_unavailable()

from app.models.repositories import ResearchRepository
from app.models.ui import Base
from dotenv import dotenv_values, load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def env() -> dict[str, str | None]:
    """Return a fixture for the environment variables from .env file."""
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
    """Fixture providing the OpenAI API key. Skips test if not found."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        pytest.skip('OpenAI API key not available')
    return api_key


@pytest.fixture
def test_data_dir() -> Path:
    """Fixture providing the path to the test data directory."""
    return Path(__file__).parent / 'data'


@pytest.fixture
def reports_pdf_dir(test_data_dir: Path) -> Path:
    """Fixture for the directory containing PDF report files."""
    return test_data_dir / 'reports' / 'pdf_reports'


@pytest.fixture
def reports_image_dir(test_data_dir: Path) -> Path:
    """Fixture for the directory containing image report files."""
    return test_data_dir / 'reports' / 'image_reports'


@pytest.fixture(scope='session')
def patients_json() -> list[dict]:
    """Load the test patients JSON data."""
    path = Path(__file__).parent / 'data' / 'patients' / 'patients.json'
    return json.loads(path.read_text())


@pytest.fixture
def wearable_extractor():
    """Provide a WearableDataFileExtractor instance for tests."""
    from hiperhealth.agents.extraction.wearable import (
        WearableDataFileExtractor,
    )

    return WearableDataFileExtractor()


@pytest.fixture
def medical_extractor():
    """Provide a MedicalReportFileExtractor instance for tests."""
    from hiperhealth.agents.extraction.medical_reports import (
        MedicalReportFileExtractor,
    )

    return MedicalReportFileExtractor()


# Use an in-memory SQLite database for fast, isolated tests
TEST_DB_URL = 'sqlite:///:memory:'
engine = create_engine(TEST_DB_URL, connect_args={'check_same_thread': False})
TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine
)


@pytest.fixture(scope='function')
def db_session():
    """Create a new database session for each test."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope='function')
def test_repo(db_session):
    """Provide a ResearchRepository instance with a test database session."""
    return ResearchRepository(db_session)


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    from app.main import app

    return TestClient(app)
