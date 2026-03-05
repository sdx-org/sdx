"""Tests for patient creation, retrieval, and listing."""

from hiperhealth.models.sqla.fhirx import Base

from tests.conftest import engine


def setup_function():
    """Create the database schema before each test."""
    Base.metadata.create_all(bind=engine)


def teardown_function():
    """Drop the database schema after each test."""
    Base.metadata.drop_all(bind=engine)


def test_create_and_get_patient(test_repo, patients_json):
    """Test creating and retrieving a patient."""
    # Arrange
    patient_data = patients_json[0]
    patient_uuid = patient_data['meta']['uuid']

    # Act
    test_repo.create_patient_and_consultation(patient_data)
    retrieved_patient = test_repo.get_patient_by_uuid(patient_uuid)

    # Assert
    assert retrieved_patient is not None
    assert retrieved_patient.uuid == patient_uuid
    assert retrieved_patient.age == 38


def test_list_patients(test_repo, patients_json):
    """Test listing all patients."""
    # Arrange
    for patient_data in patients_json:
        test_repo.create_patient_and_consultation(patient_data)

    # Act
    all_patients = test_repo.list_patients()

    # Assert
    assert len(all_patients) == len(patients_json)


def test_default_consent_created_with_patient(test_repo, patients_json):
    """Ensure default consent and initial audit entry are created."""
    patient_data = patients_json[0]
    patient_uuid = patient_data['meta']['uuid']
    patient = test_repo.create_patient_and_consultation(patient_data)

    consent = test_repo.get_patient_consent(patient_uuid)
    logs = test_repo.list_consent_audit_logs(patient_uuid)

    assert consent is not None
    assert consent.patient_id == patient.id
    assert consent.allow_diagnostics is True
    assert len(logs) >= 1
    assert logs[0].action in {'consent_created', 'consent_updated'}


def test_update_consent_and_audit(test_repo, patients_json):
    """Ensure consent updates are audited and persisted."""
    patient_data = patients_json[0]
    patient_uuid = patient_data['meta']['uuid']
    test_repo.create_patient_and_consultation(patient_data)

    updated = test_repo.update_patient_consent(
        patient_uuid=patient_uuid,
        updates={'allow_wearable_data': False},
        actor='tester',
        reason='patient opted out of wearable processing',
    )
    logs = test_repo.list_consent_audit_logs(patient_uuid)

    assert updated is not None
    assert updated.allow_wearable_data is False
    assert len(logs) >= 1
    assert any(log.action == 'consent_updated' for log in logs)
