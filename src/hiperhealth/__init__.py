"""
title: hiperhealth.
"""

from importlib import metadata as importlib_metadata


def get_version() -> str:
    """
    title: Return the program version.
    returns:
      type: str
      description: Return value.
    """
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return '0.4.0'  # semantic-release


version = get_version()

__version__ = version
__author__ = 'Ivan Ogasawara'
__email__ = 'ivan.ogasawara@gmail.com'
