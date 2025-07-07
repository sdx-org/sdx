"""Repositories for reading and saving the web app data."""

from __future__ import annotations

import json

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, TypeVar

T = TypeVar('T')
# Patient type is an alias for now
# TODO: swap for Pydantic in future update
# once we have a better defined schema
Patient = dict[str, Any]


class RepositoryInterface(ABC):
    """Repository interface."""

    @abstractmethod
    def all(self) -> list[T]:
        """Return all records."""
        pass

    @abstractmethod
    def get(self, id) -> Optional[T]:
        """Return a single record."""
        pass

    @abstractmethod
    def create(self, data) -> T:
        """Create a new record."""
        pass

    @abstractmethod
    def update(self, id, data) -> bool:
        """Update a record."""
        pass

    @abstractmethod
    def delete(self, id) -> bool:
        """Delete a record."""
        pass


class PatientRepository(RepositoryInterface):
    """Implement the repository interface for Patient."""

    DATA_PATH = (
        Path(__file__).parent.parent / 'app/data/patients/patients.json'
    )

    patients = list[Patient]

    def __init__(self) -> None:
        """Load patients from file."""
        with self.DATA_PATH.open('r') as f:
            # TODO: swap for parquet in future updates
            self.patients = json.load(f)

    def all(self) -> list[Patient]:
        """Return all patients."""
        return self.patients

    def get(self, id) -> Optional[Patient]:
        """Return a single patient if exists."""
        for patient in self.patients:
            if patient['meta']['uuid'] == id:
                return patient
        return None

    def create(self, data) -> Patient:
        """Create a new patient."""
        self.patients.append(data)
        with open(self.DATA_PATH, 'w') as f:
            json.dump(self.patients, f)
        return data

    def update(self, id, data) -> bool:
        """Update a patient. Returns true if successful."""
        for index, patient in enumerate(self.patients):
            if patient['meta']['uuid'] == id:
                self.patients[index] = data
                with open(self.DATA_PATH, 'w') as f:
                    json.dump(self.patients, f)
                return True

        # return false if patient does not exist
        return False

    def delete(self, id) -> bool:
        """Delete a patient. Returns true if successful."""
        for index, patient in enumerate(self.patients):
            if patient['meta']['uuid'] == id:
                del self.patients[index]
                with open(self.DATA_PATH, 'w') as f:
                    json.dump(self.patients, f)
                return True

        # return false if patient does not exist
        return False
