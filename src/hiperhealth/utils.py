"""
title: HiPerHealth utility functions.
"""

import datetime

from typing import Any


def is_float(value: str) -> bool:
    """
    title: Check if a string represents a decimal number (not a plain integer).
    summary: |-
      Parameters
          ----------
          value : str
              String to evaluate; surrounding whitespace is ignored.

          Returns
          -------
          bool
      True if the string parses as a float and is not a plain integer; False
              otherwise.

          Notes
          -----
          Accepts standard float formats, including scientific notation
      (e.g., ``"1e-3"``). Plain integers (optionally signed) and empty strings
          return ``False``.
    parameters:
      value:
        type: str
        description: Value for value.
    returns:
      type: bool
      description: Return value.
    """
    stripped = value.strip()

    # Empty strings are not floats
    if not stripped:
        return False

    # Reject plain integer strings (e.g., "1", "-2", "+3")
    if stripped.lstrip('+-').isdigit():
        return False

    # Otherwise, validate it parses as a float
    try:
        float(stripped)
        return True
    except ValueError:
        return False


def make_json_serializable(obj: Any) -> Any:
    """
    title: Convert objects to JSON-serializable format recursively.
    parameters:
      obj:
        type: Any
        description: Value for obj.
    returns:
      type: Any
      description: Return value.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    else:
        return obj
