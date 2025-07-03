"""Base script for generating models."""

from __future__ import annotations

import inspect
import pkgutil

from types import ModuleType
from typing import Dict, Type

from pydantic import BaseModel
from sdx.schema.fhir import BaseLanguage

# Package paths to scan for Pydantic models
PACKAGE_PATHS = [
    'sdx.schema',
]

IGNORED_CLASSES = [BaseLanguage, BaseModel]


def iter_pydantic_models() -> Dict[str, Type[BaseModel]]:
    """
    Yield (qualified_name, model_cls).

    Yield values for every subclass of BaseModel found in PACKAGE_PATHS.
    """
    discovered: dict[str, Type[BaseModel]] = {}
    for module_path in PACKAGE_PATHS:
        module: ModuleType = __import__(module_path, fromlist=['*'])
        # Walk submodules in case of package
        for _loader, submod_name, _ispkg in pkgutil.walk_packages(
            module.__path__, module.__name__ + '.'
        ):
            submod = __import__(submod_name, fromlist=['*'])
            for name, obj in inspect.getmembers(submod, inspect.isclass):
                if not obj.__module__.startswith('sdx'):
                    continue
                if issubclass(obj, BaseModel) and obj not in IGNORED_CLASSES:
                    discovered[f'{submod.__name__}.{name}'] = obj
    return discovered


def is_concrete_model(model_cls: Type[BaseModel]) -> bool:
    """
    Return True if `model_cls` should be mapped to a table.

    Heuristics
    ----------
    1. The class advertises itself as abstract via `__abstract__ = True`.
    2. Inner `Config` / `model_config` sets `table_abstract = True`.
    3. No own fields ➜ skip (helper alias such as BaseLanguage).
    """
    # Rule 1: explicit marker
    if getattr(model_cls, '__abstract__', False):
        return False

    # Rule 2: honour Pydantic v1 Config or v2 model_config
    cfg = getattr(model_cls, 'Config', None) or getattr(
        model_cls, 'model_config', None
    )
    if cfg and getattr(cfg, 'table_abstract', False):
        return False

    # Rule 3: helper types usually have zero model_fields
    if not model_cls.model_fields:
        return False

    return True
