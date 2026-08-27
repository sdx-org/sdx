import sys

import pytest

if __name__ == '__main__':
    args = ['tests/test_extraction_wearable.py', '-vv']
    sys.exit(pytest.main(args))
