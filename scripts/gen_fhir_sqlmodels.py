"""
dynamic_sqlmodel.py
Utility to convert an arbitrary Pydantic model into a SQLModel table
at runtime.  Relationships are *not* inferred (add them yourself if
needed), but all scalar attributes are mapped to first-class columns.
"""

from __future__ import annotations

import json

from datetime import date, datetime
from typing import Any, Dict, Tuple, Type, get_args, get_origin

from pydantic import BaseModel, create_model
from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

TYPE_MAP = {
    str: String,
    int: Integer,
    float: Float,
    bool: Boolean,
    datetime: DateTime,
    date: Date,
}

FALLBACK_TYPE = JSONB  # serialise unknown objects as JSONB


def sqlalchemy_type(annotation: Any) -> Any:
    """Return the SQLAlchemy column type for a Pydantic field annotation."""
    origin = get_origin(annotation)

    # Optional[T]  -> origin is Union
    if origin is None:
        return TYPE_MAP.get(annotation, FALLBACK_TYPE)

    # List[T], Dict[K,V], etc.  => JSONB
    return FALLBACK_TYPE


def make_sqlmodel_from_pydantic(
    model_cls: Type[BaseModel],
    *,
    table_name: str | None = None,
) -> Type[SQLModel]:
    """
    Dynamically create a SQLModel subclass with `table=True`
    from a Pydantic model class.

    Args
    ----
    model_cls:
        The source Pydantic model.
    table_name:
        Optional override for `__tablename__`.  Defaults to lower-case
        Pydantic class name.

    Returns
    -------
    SQLModel table class
    """
    field_definitions: Dict[str, Tuple[Any, Any]] = {}
    fields = (
        model_cls.model_fields  # Pydantic v2
        if hasattr(model_cls, 'model_fields')
        else model_cls.__fields__  # v1 fallback
    )

    for name, fld in fields.items():
        annotation = fld.annotation  # type object
        default = fld.default if fld.default is not None else ...

        sa_type = sqlalchemy_type(annotation)
        # Heuristic: field named "id" becomes primary key
        is_pk = name == 'id'

        field_definitions[name] = (
            annotation,
            Field(
                default,
                sa_type=sa_type,
                primary_key=is_pk,
                nullable=not fld.is_required(),
                index=not is_pk,
            ),
        )

    sqlmodel_cls = create_model(  # type: ignore[call-arg]
        model_cls.__name__,
        __base__=SQLModel,
        __tablename__=table_name or model_cls.__name__.lower(),
        __cls_kwargs__={'table': True},
        **field_definitions,
    )

    # Patch __doc__ for clarity
    sqlmodel_cls.__doc__ = f'Auto-generated SQLModel table for {model_cls.__module__}.{model_cls.__name__}'
    return sqlmodel_cls
