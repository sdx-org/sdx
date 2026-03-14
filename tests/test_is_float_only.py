"""
title: Test is_float function independently.
"""

from hiperhealth.utils import is_float


def test_is_float():
    """
    title: Test if string is a float.
    """
    assert is_float('1.0')
    assert is_float('   1.0    ')
    assert is_float('-3.00')
    assert is_float('1.52')
    assert is_float('0.02')
    assert not is_float('1')
    assert not is_float('a')
    assert not is_float('')
    assert not is_float('-3:00')
