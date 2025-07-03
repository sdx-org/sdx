"""
Test SQLModel models generated.

Smoke-tests for *all* auto-generated SQLModel tables in
`sdx.models.sqlmodel.fhir`.

The suite verifies that:

1. Metadata creates successfully in an in-memory SQLite engine.
2. Every mapped class:
   • Can be instantiated with minimally required values.
   • Persists and can be queried back with a primary-key lookup.

It does **not** validate domain rules—only that the generated code is
structurally sound and compatible with SQLModel / SQLAlchemy.
"""

from __future__ import annotations

import importlib
import inspect

from datetime import date, datetime
from typing import Any, Dict

import pytest

from sqlmodel import Session, SQLModel, create_engine

_SQLA_TYPE_MAP: Dict[Any, Any] = {
    'String': 'sample',
    'Integer': 1,
    'Float': 1.0,
    'Boolean': True,
    'DateTime': datetime.utcnow(),
    'Date': date.today(),
    'JSON': {},
    'JSONB': {},
}

# In SQLite the JSONB columns are created as JSON → treat identically
_SQLA_FALLBACK_SAMPLE = {}


def _sample_for_column(sa_column) -> Any:
    """Return a hashable sample value for the given SQLAlchemy column."""
    coltype = sa_column.type
    typename = type(coltype).__name__
    if typename in _SQLA_TYPE_MAP:
        return _SQLA_TYPE_MAP[typename]
    # Generic fallback for unknown / user types
    return _SQLA_FALLBACK_SAMPLE


def _iter_sqlmodel_tables():
    """Yield each concrete SQLModel class from the generated module."""
    mod = importlib.import_module('sdx.models.sqlmodel.fhir')
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if (
            issubclass(obj, SQLModel)
            and getattr(obj, '__table__', None) is not None
        ):
            yield obj


@pytest.fixture(scope='session')
def engine():
    """In-memory SQLite engine."""
    return create_engine('sqlite:///:memory:')


@pytest.fixture(scope='session', autouse=True)
def create_all(engine):
    """Create tables once per test session."""
    SQLModel.metadata.create_all(engine)
    yield
    SQLModel.metadata.drop_all(engine)


@pytest.fixture
def db_session(engine):
    """Provide a fresh transactional session per test."""
    with Session(engine) as ses:
        yield ses
        ses.rollback()


def test_metadata_nonempty():
    """The auto-generated metadata must include at least one table."""
    assert SQLModel.metadata.tables, 'No tables were generated'


@pytest.mark.parametrize('table_cls', list(_iter_sqlmodel_tables()))
def test_table_basic_crud(table_cls, db_session):
    """
    Instantiate each auto-generated SQLModel.

    Instantiate each auto-generated SQLModel table with dummy data,
    persist, and query back via primary-key lookup.
    """
    # Build kwargs for non-nullable columns with no default
    kwargs: Dict[str, Any] = {}
    for col in table_cls.__table__.columns:
        if col.default is not None or (col.primary_key and col.autoincrement):
            # column self-generates a value
            continue
        if col.primary_key or not col.nullable:
            kwargs[col.name] = _sample_for_column(col)

    obj = table_cls(**kwargs)  # type: ignore[arg-type]
    db_session.add(obj)
    db_session.commit()

    # Fetch back using identity map
    pk_cols = list(table_cls.__table__.primary_key.columns)
    assert pk_cols, f'{table_cls.__name__} has no primary key'
    pk_values = tuple(getattr(obj, c.name) for c in pk_cols)

    fetched = db_session.get(
        table_cls, pk_values if len(pk_values) > 1 else pk_values[0]
    )
    assert fetched is not None, (
        f'{table_cls.__name__} failed round-trip persistence'
    )
