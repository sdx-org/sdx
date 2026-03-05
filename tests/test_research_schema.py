"""Unit tests for the research schema module."""

from hiperhealth.models.sqla.fhirx import Base
from sqlalchemy import create_engine, inspect


def test_database_schema_creation():
    """Connects to the DB and checks if the new research tables exist."""
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    # Assert that all your new, normalized tables were created
    assert 'patients' in tables
    assert 'consultations' in tables
    assert 'diagnoses' in tables
    assert 'exams' in tables
    assert 'consultation_diagnoses' in tables
    assert 'consultation_exams' in tables
    assert 'patient_consents' in tables
    assert 'patient_consent_audit_logs' in tables
