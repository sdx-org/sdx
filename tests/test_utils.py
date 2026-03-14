"""
title: Test HiPerHealth utility functions module.
"""

import datetime

from hiperhealth.utils import is_float, make_json_serializable


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


def test_make_json_serializable_date():
    """
    title: Test datetime.date and datetime.datetime objects are serialized.
    """
    d = datetime.date(2023, 1, 1)
    dt = datetime.datetime(2023, 1, 1, 12, 30, 45)

    assert make_json_serializable(d) == d.isoformat()
    assert make_json_serializable(dt) == dt.isoformat()


def test_make_json_serializable_dict():
    """
    title: Test dict serialization with nested types.
    """
    data = {
        'a': datetime.datetime(2023, 1, 1, 12, 0),
        'b': [1, 2, datetime.date(2023, 1, 2)],
        'c': {'nested': datetime.date(2023, 1, 3)},
    }
    result = make_json_serializable(data)
    assert result == {
        'a': '2023-01-01T12:00:00',
        'b': [1, 2, '2023-01-02'],
        'c': {'nested': '2023-01-03'},
    }
