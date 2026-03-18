"""
title: Tests for hiperhealth package.
"""

import hiperhealth


def test_version_is_set():
    """
    title: Package version should be a non-empty string.
    """
    assert hiperhealth.__version__
    assert isinstance(hiperhealth.__version__, str)
