"""Tests for the patient repository."""

import random
import shutil

from pathlib import Path

import pytest

from research.models.repositories import PatientRepository

########### FIXTURES ###########


@pytest.fixture
def patient_repository():
    """Temporary patient repository fixture."""
    # patch DATA_PATH to test data path
    TEST_DATA_PATH = Path(__file__).parent / 'data/patients'
    PATIENTS_DATA_PATH = TEST_DATA_PATH / 'patients.json'
    TEMP_DATA_PATH = TEST_DATA_PATH / 'temp_patients.json'

    shutil.copyfile(PATIENTS_DATA_PATH, TEMP_DATA_PATH)

    # patch DATA_PATH to test data path
    PatientRepository.DATA_PATH = TEMP_DATA_PATH
    temporary_repository = PatientRepository()

    yield temporary_repository

    # clean up, delete TEMP_DATA_PATH
    TEMP_DATA_PATH.unlink()


@pytest.fixture
def patient_id(patient_repository):
    """Patient ID fixture."""
    patients = patient_repository.all()
    return random.choice(patients)['meta']['uuid']


########### TEST CASES ###########


def test_all_patients(patient_repository):
    """Test repository to get all patients."""
    patients = patient_repository.all()
    assert len(patients) > 0
    assert len(patients) == 2


def test_get_patient(patient_repository, patient_id):
    """Test repository to get one patient by id."""
    patient = patient_repository.get(patient_id)
    assert patient is not None
    assert patient['meta']['uuid'] == patient_id


def test_create_patient(patient_repository):
    """Test repository to create a new patient."""
    patient_repository.create({'meta': {'uuid': 'new-patient'}})
    patient = patient_repository.get('new-patient')
    assert patient['meta']['uuid'] == 'new-patient'


def test_update_patient(patient_repository, patient_id):
    """Test repository to update one patient by id."""
    patient = patient_repository.get(patient_id)
    patient_age = patient['patient']['age']
    patient_new_age = patient_age + 10
    patient['patient']['age'] = patient_new_age

    patient_repository.update(patient_id, patient)
    updated_patient = patient_repository.get(patient_id)

    assert updated_patient['patient']['age'] == patient_new_age


def test_delete_patient(patient_repository, patient_id):
    """Test repository to delete one patient by id."""
    patient_repository.delete(patient_id)
    assert patient_repository.get(patient_id) is None
