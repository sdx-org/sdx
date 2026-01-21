"""Test cases for the ResearchRepository class."""

# ruff: noqa: E402
import sys

from pathlib import Path
from uuid import uuid4

BACKEND_DIR = Path(__file__).parents[1] / 'research' / 'backend'
sys.path.insert(0, str(BACKEND_DIR))
from app.models.repositories import ResearchRepository
from sqlalchemy.orm import Session


def test_create_patient_and_consultation(db_session: Session) -> None:
    """Test creating a new patient and their initial consultation."""
    repo = ResearchRepository(db_session)
    patient_data = {
        'meta': {'uuid': str(uuid4()), 'timestamp': '2023-10-26T10:00:00'},
        'patient': {'age': 30, 'gender': 'male'},
    }
    patient = repo.create_patient_and_consultation(patient_data)
    assert patient.uuid == patient_data['meta']['uuid']
    assert patient.age == patient_data['patient']['age']
    assert patient.gender == patient_data['patient']['gender']
    assert len(patient.consultations) == 1
    assert (
        patient.consultations[0].timestamp.isoformat()
        == patient_data['meta']['timestamp']
    )


def test_get_patient_by_uuid(db_session: Session) -> None:
    """Test retrieving a patient by their UUID."""
    repo = ResearchRepository(db_session)
    patient_data = {
        'meta': {'uuid': str(uuid4()), 'timestamp': '2023-10-26T10:00:00'},
        'patient': {'age': 45, 'gender': 'female'},
    }
    created_patient = repo.create_patient_and_consultation(patient_data)
    retrieved_patient = repo.get_patient_by_uuid(created_patient.uuid)
    assert retrieved_patient is not None
    assert retrieved_patient.uuid == created_patient.uuid


def test_list_patients(db_session: Session) -> None:
    """Test listing all patients."""
    repo = ResearchRepository(db_session)
    repo.create_patient_and_consultation(
        {
            'meta': {'uuid': str(uuid4()), 'timestamp': '2023-10-26T10:00:00'},
            'patient': {'age': 35, 'gender': 'male'},
        }
    )
    repo.create_patient_and_consultation(
        {
            'meta': {'uuid': str(uuid4()), 'timestamp': '2023-10-26T11:00:00'},
            'patient': {'age': 50, 'gender': 'female'},
        }
    )
    patients = repo.list_patients()
    assert len(patients) == 2


def test_get_or_create_diagnosis(db_session: Session) -> None:
    """Test getting an existing diagnosis or creating a new one."""
    repo = ResearchRepository(db_session)
    diagnosis_name = 'Hypertension'
    # Create
    new_diag = repo.get_or_create_diagnosis(diagnosis_name)
    assert new_diag.name == diagnosis_name
    # Get
    existing_diag = repo.get_or_create_diagnosis(diagnosis_name)
    assert existing_diag.id == new_diag.id


def test_get_or_create_exam(db_session: Session) -> None:
    """Test getting an existing exam or creating a new one."""
    repo = ResearchRepository(db_session)
    exam_name = 'Blood Pressure'
    # Create
    new_exam = repo.get_or_create_exam(exam_name)
    assert new_exam.name == exam_name
    # Get
    existing_exam = repo.get_or_create_exam(exam_name)
    assert existing_exam.id == new_exam.id


def test_delete_patient(db_session: Session) -> None:
    """Test deleting a patient."""
    repo = ResearchRepository(db_session)
    patient_uuid = str(uuid4())
    patient_data = {
        'meta': {'uuid': patient_uuid, 'timestamp': '2023-10-26T10:00:00'},
        'patient': {'age': 60, 'gender': 'male'},
    }
    repo.create_patient_and_consultation(patient_data)
    assert repo.delete_patient(patient_uuid) is True
    assert repo.get_patient_by_uuid(patient_uuid) is None
