"""
title: Test the extraction of wearable data.
"""

import io

from pathlib import Path

import pytest

from hiperhealth.agents.extraction.wearable import WearableDataExtractorError

TEST_DATA_PATH = Path(__file__).parent / 'data' / 'wearable'
JSON_FILE = TEST_DATA_PATH / 'wearable_data.json'
CSV_FILE = TEST_DATA_PATH / 'wearable_data.csv'
UNSUPPORTED_FILE = TEST_DATA_PATH / 'invalid_extension.txt'
CORRUPT_FILE = TEST_DATA_PATH / 'giberish.json'


@pytest.fixture
def mock_excel_file(tmp_path):
    """Create a temporary mock Excel file."""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(['name', 'age', 'heart_rate', 'timestamp'])
    ws.append(['John Doe', 30, 70, 1])
    ws.append(['John Doe', 30, 80, 2])
    ws.append(['John Doe', 30, 90, 3])
    ws.append(['Supa User', 99, 60, 1])

    file_path = tmp_path / 'wearable_data.xlsx'
    wb.save(file_path)
    wb.close()
    return file_path


def test_only_supported_files_can_be_extracted(wearable_extractor, mock_excel_file):
    """
    title: Test that only supported files can be extracted.
    parameters:
      wearable_extractor:
        description: Value for wearable_extractor.
      mock_excel_file:
        description: A temporary mock Excel file.
    """
    assert wearable_extractor.is_supported(JSON_FILE)
    assert wearable_extractor.is_supported(CSV_FILE)
    assert wearable_extractor.is_supported(mock_excel_file)
    assert wearable_extractor.is_supported(CORRUPT_FILE)
    assert not wearable_extractor.is_supported(UNSUPPORTED_FILE)


def test_extract_json(wearable_extractor):
    """
    title: Test that JSON file can be extracted.
    parameters:
      wearable_extractor:
        description: Value for wearable_extractor.
    """
    assert wearable_extractor._is_json(JSON_FILE)
    wearable_data = wearable_extractor.extract_wearable_data(JSON_FILE)
    assert wearable_data
    assert len(wearable_data) > 0


def test_extract_csv(wearable_extractor):
    """
    title: Test that CSV file can be extracted.
    parameters:
      wearable_extractor:
        description: Value for wearable_extractor.
    """
    assert wearable_extractor._is_csv(CSV_FILE)
    wearable_data = wearable_extractor.extract_wearable_data(CSV_FILE)
    assert wearable_data
    assert len(wearable_data) > 0


def test_extract_unsupported_file(wearable_extractor):
    """
    title: Test that unsupported file cannot be extracted.
    parameters:
      wearable_extractor:
        description: Value for wearable_extractor.
    """
    with pytest.raises(WearableDataExtractorError):
        wearable_extractor.extract_wearable_data(UNSUPPORTED_FILE)


def test_extract_malformed_file(wearable_extractor):
    """
    title: Test that malformed/corrupted file cannot be extracted.
    parameters:
      wearable_extractor:
        description: Value for wearable_extractor.
    """
    with pytest.raises(WearableDataExtractorError):
        wearable_extractor.extract_wearable_data(CORRUPT_FILE)


def test_support_inmemory_json_file(wearable_extractor):
    """
    title: Test that in-memory file can be extracted.
    parameters:
      wearable_extractor:
        description: Value for wearable_extractor.
    """
    raw_json = b"""
    [
    {
        "name": "John Doe",
        "age": 30,
        "records" : [
            {"2025-05-01": [
                {"heart_rate": 70, "timestamp": 1},
                {"heart_rate": 80, "timestamp": 2},
                {"heart_rate": 90, "timestamp": 3}
                ]
            }
        ]
    },
    {
        "name": "Supa User",
        "age": 99,
        "records": []
    }
    ]
    """

    some_json = io.BytesIO(raw_json)

    assert wearable_extractor.is_supported(some_json)
    assert wearable_extractor._is_json(some_json)
    assert not wearable_extractor._is_csv(some_json)
    wearable_data = wearable_extractor.extract_wearable_data(some_json)

    assert wearable_data
    assert len(wearable_data) == 2


def test_support_inmemory_csv(wearable_extractor):
    """
    title: Test that in-memory CSV can be extracted.
    parameters:
      wearable_extractor:
        description: Value for wearable_extractor.
    """
    raw_csv = b"""
        name,age,heart_rate,timestamp
        John Doe,30,70,1
        John Doe,30,80,2
        John Doe,30,90,3
        Supa User,99,60,1
    """

    some_csv = io.BytesIO(raw_csv.strip())

    assert wearable_extractor.is_supported(some_csv)
    assert wearable_extractor._is_csv(some_csv)
    assert not wearable_extractor._is_json(some_csv)
    wearable_data = wearable_extractor.extract_wearable_data(some_csv)

    assert isinstance(wearable_data, list)
    assert len(wearable_data) == 4
    assert wearable_data[0]['name'] == 'John Doe'
    assert wearable_data[1]['heart_rate'] == 80


def test_extract_excel(wearable_extractor, mock_excel_file):
    """Test that Excel file can be extracted."""
    assert wearable_extractor._is_excel(mock_excel_file)
    wearable_data = wearable_extractor.extract_wearable_data(mock_excel_file)
    assert wearable_data
    assert len(wearable_data) == 4
    assert wearable_data[0]['name'] == 'John Doe'
    assert wearable_data[1]['heart_rate'] == 80


def test_support_inmemory_excel(wearable_extractor):
    """Test that in-memory Excel can be extracted."""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(['name', 'age', 'heart_rate', 'timestamp'])
    ws.append(['John Doe', 30, 70, 1])
    ws.append(['John Doe', 30, 80, 2])
    ws.append(['John Doe', 30, 90, 3])
    ws.append(['Supa User', 99, 60, 1])

    some_excel = io.BytesIO()
    wb.save(some_excel)
    wb.close()
    some_excel.seek(0)

    assert wearable_extractor.is_supported(some_excel)
    assert wearable_extractor._is_excel(some_excel)
    assert not wearable_extractor._is_json(some_excel)
    wearable_data = wearable_extractor.extract_wearable_data(some_excel)

    assert isinstance(wearable_data, list)
    assert len(wearable_data) == 4
    assert wearable_data[0]['name'] == 'John Doe'
    assert wearable_data[-1]['age'] == 99
