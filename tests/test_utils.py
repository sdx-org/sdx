"""
title: Test HiPerHealth utility functions module.
"""

import datetime

from hiperhealth.utils import (
    _is_sensitive_key,
    _scrub_sensitive_data,
    is_float,
    make_json_serializable,
)


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


def test_is_sensitive_key():
    """
    title: Test detection of sensitive keys.
    """
    # Expected matches
    assert _is_sensitive_key('api_key')
    assert _is_sensitive_key('TOKEN')
    assert _is_sensitive_key('secret_value')
    assert _is_sensitive_key('password123')
    assert _is_sensitive_key('DB_PWD')

    # False positives (should not match)
    assert not _is_sensitive_key('api_endpoint')
    assert not _is_sensitive_key('user_name')
    assert not _is_sensitive_key('configuration')
    assert not _is_sensitive_key('db_port')


def test_scrub_sensitive_data():
    """
    title: Test scrubbing of sensitive data recursively.
    """
    data = {
        'api_key': 'secret123',
        'public_info': 'hello',
        'nested': {
            'token': 'abc',
            'safe': 'data'
        },
        'list_data': [
            {'password': 'pwd', 'id': 1},
            (1, 2, {'secret': 'hidden'})
        ]
    }

    scrubbed = _scrub_sensitive_data(data)

    assert scrubbed['api_key'] == '********'
    assert scrubbed['public_info'] == 'hello'
    assert scrubbed['nested']['token'] == '********'
    assert scrubbed['nested']['safe'] == 'data'
    assert scrubbed['list_data'][0]['password'] == '********'
    assert scrubbed['list_data'][0]['id'] == 1
    # Check that tuple type is preserved
    assert isinstance(scrubbed['list_data'][1], tuple)
    assert scrubbed['list_data'][1][0] == 1
    assert scrubbed['list_data'][1][2]['secret'] == '********'

