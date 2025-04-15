"""Tests for the treatment module."""

from unittest.mock import MagicMock, patch

import pytest

from sdx.treatment import (
    _fhir_dict_to_strings,
    suggest_treatment_from_pdf,
)

from tests.conftest import api_key_openai


class TestFhirDictToStrings:
    """Test suite for the _fhir_dict_to_strings function."""

    def test_empty_dict(self):
        """Test with empty FHIR data."""
        result = _fhir_dict_to_strings({})
        assert result == ['No clinical data provided.']

    def test_simple_dict(self):
        """Test with a simple dictionary of FHIR resources."""
        fhir_data = {
            'Patient': {'id': '123', 'name': 'John Doe', 'gender': 'male'},
            'Observation': {'code': 'blood-pressure', 'value': '120/80'},
        }
        result = _fhir_dict_to_strings(fhir_data)
        assert len(result) == 2
        assert (
            'ResourceType Patient: id: 123; name: John Doe; gender: male'
            in result
        )
        assert (
            'ResourceType Observation: code: blood-pressure; value: 120/80'
            in result
        )

    def test_with_complex_data(self):
        """Test with nested structures in FHIR data."""
        fhir_data = {
            'Patient': {
                'id': '123',
                'contact': {'name': 'Jane', 'phone': '123-456-7890'},
            },
        }
        result = _fhir_dict_to_strings(fhir_data)
        assert len(result) == 1
        assert (
            'ResourceType Patient: id: 123; contact: [complex data]' in result
        )

    def test_with_list_data(self):
        """Test with list data in FHIR resources."""
        fhir_data = {
            'Condition': [
                {'code': 'diabetes', 'severity': 'moderate'},
                {'code': 'hypertension', 'severity': 'mild'},
            ]
        }
        result = _fhir_dict_to_strings(fhir_data)
        assert len(result) == 2
        assert (
            'ResourceType Condition [0]: code: diabetes; severity: moderate'
            in result
        )
        assert (
            'ResourceType Condition [1]: code: hypertension; severity: mild'
            in result
        )


@pytest.mark.skipif(not api_key_openai, reason='OpenAI API key not available')
class TestSuggestTreatment:
    """Test suite for the suggest_treatment_from_pdf function."""

    @patch('sdx.treatment.get_report_data_from_pdf')
    @patch('sdx.treatment.OpenAIGen')
    def test_suggest_treatment_successful(
        self, mock_openai_gen, mock_get_data, api_key_openai
    ):
        """Test successful treatment suggestion flow."""
        if api_key_openai is None:
            pytest.skip('OpenAI API key not available')

        mock_get_data.return_value = {
            'Patient': {'id': '123', 'name': 'John Doe'},
            'Condition': {'code': 'diabetes', 'severity': 'moderate'},
        }

        mock_generator = MagicMock()
        mock_generator.generate.return_value = (
            'Recommended treatment: Insulin therapy and diet management'
        )
        mock_openai_gen.return_value = mock_generator

        expected_result = {'CarePlan': {'activity': 'Insulin therapy'}}
        with patch('sdx.treatment.extract_fhir', return_value=expected_result):
            result = suggest_treatment_from_pdf(
                'dummy_path.pdf', openai_api_key=api_key_openai
            )

        assert result == expected_result
        assert 'Condition' in mock_generator.generate.call_args[1]['query']

    @patch('sdx.treatment.get_report_data_from_pdf')
    def test_suggest_treatment_with_custom_augmenter(
        self, mock_get_data, api_key_openai
    ):
        """Test treatment suggestion with custom augmenter."""
        if api_key_openai is None:
            pytest.skip('OpenAI API key not available')

        mock_get_data.return_value = {
            'Patient': {'id': '123', 'name': 'John Doe'},
        }

        mock_augmenter = MagicMock()
        mock_augmenter.search.return_value = ['Enhanced context']

        with patch('sdx.treatment.OpenAIGen') as mock_openai_gen:
            mock_generator = MagicMock()
            mock_generator.generate.return_value = 'Treatment plan: Rest'
            mock_openai_gen.return_value = mock_generator

            with patch('sdx.treatment.extract_fhir') as mock_extract:
                mock_extract.return_value = {'CarePlan': {'activity': 'Rest'}}

                result = suggest_treatment_from_pdf(
                    'dummy_path.pdf',
                    openai_api_key=api_key_openai,
                    augmenter=mock_augmenter,
                )

        mock_augmenter.search.assert_called_once()
        assert isinstance(result, dict)
        assert 'CarePlan' in result
